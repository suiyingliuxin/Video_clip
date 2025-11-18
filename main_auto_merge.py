import os
import pickle
import numpy as np
import torch
from utils import get_frames, get_batches
from autoshot import TransNetV2Supernet
import ffmpeg
import webrtcvad
import wave
import contextlib
import subprocess
from moviepy import VideoFileClip
from pathlib import Path
from vad_webrtc import extract_audio, detect_voice_activity, separate_vocals_cli
import tempfile
from asr import VideoSubtitleDetector
import re
from doubao import DoubaoCompletenessChecker
import librosa
import json
from datetime import timedelta
results_cache = {}

def load_model(checkpoint_path, device):
    """加载 AutoShot 模型"""
    print(f'Loading model from {checkpoint_path}')
    
    # 初始化模型
    model = TransNetV2Supernet().eval()
    
    # 加载权重
    if os.path.exists(checkpoint_path):
        model_dict = model.state_dict()
        pretrained_dict = torch.load(checkpoint_path, map_location=device)
        
        # 提取权重（兼容不同保存格式）
        if 'net' in pretrained_dict:
            pretrained_dict = pretrained_dict['net']
        
        # 过滤并更新权重
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(f"Model has {len(model_dict)} parameters, loading {len(pretrained_dict)} parameters")
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 移动到设备
    if device == "cuda":
        model = model.cuda()
    
    model.eval()
    return model

def predict_batch(model, batch, device):
    """对单个批次进行预测"""
    # 转换格式: (H, W, C, T) -> (1, C, T, H, W)
    batch = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]).float()
    batch = batch.to(device)
    
    with torch.no_grad():
        output = model(batch)
        if isinstance(output, tuple):
            output = output[0]
        prediction = torch.sigmoid(output[0])
    
    return prediction.detach().cpu().numpy()

def predict_video(video_path, model, device, threshold=0.296):
    """
    对单个视频进行镜头边界检测
    
    参数:
        video_path: 视频文件路径
        model: 加载好的模型
        device: 'cuda' 或 'cpu'
        threshold: 分类阈值，默认 0.296（论文最优值）
    
    返回:
        predictions: 每帧的预测概率 (numpy array)
        shot_boundaries: 镜头边界帧号列表
        scenes: 镜头片段 [[start_frame, end_frame], ...]
    """
    print(f"Processing video: {video_path}")
    
    # 读取视频帧
    frames = get_frames(video_path)
    total_frames = len(frames)
    print(f"Total frames: {total_frames}")
    
    # 分批预测
    predictions = []
    batch_count = 0
    
    for batch in get_batches(frames):
        batch_count += 1
        one_hot = predict_batch(model, batch, device)
        
        # 取中间50帧（避免边界效应）
        predictions.append(one_hot[25:75])
        
        if batch_count % 10 == 0:
            print(f"Processed {batch_count} batches...")
    
    # 拼接所有预测结果
    predictions = np.concatenate(predictions, 0)[:total_frames]
    
    # 应用阈值得到二值边界
    boundaries_binary = (predictions > threshold).astype(np.uint8)
    
    # 提取边界帧号
    shot_boundaries = np.where(boundaries_binary == 1)[0].tolist()
    print(shot_boundaries)
    # 转换为镜头片段格式
    scenes = []
    if len(shot_boundaries) > 0:
        start = 0
        for boundary in shot_boundaries:
            if boundary > start:
                if boundary - start > 0:
                    scenes.append([start, boundary])
            start = boundary + 1
        # 添加最后一个镜头
        if start < total_frames:
            if total_frames - start > 1:
                scenes.append([start, total_frames - 1])
    else:
        # 没有检测到边界，整个视频是一个镜头
        scenes.append([0, total_frames - 1])
    
    print(f"Detected {len(scenes)} shots")
    
    return predictions, shot_boundaries, scenes

def save_results(results, output_path):
    """保存预测结果"""
    with open(output_path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Results saved to {output_path}")

def split_video_by_scenes_with_audio(video_path, scenes, output_dir="output_scenes"):
    """
    使用 ffmpeg 根据帧精确切分视频（保留音频）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取视频信息
    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    fps = eval(video_info['r_frame_rate'])
    
    print(f"视频信息: FPS={fps}")
    print(f"检测到 {len(scenes)} 个场景片段\n")
    
    for idx, (start_frame, end_frame) in enumerate(scenes):
        # 修正：确保结束帧 >= 起始帧
        if end_frame < start_frame:
            print(f"警告: 场景 {idx} 的结束帧 ({end_frame}) 小于起始帧 ({start_frame})，已修正为相同")
            end_frame = start_frame
        
        output_path = os.path.join(output_dir, f"scene_{idx:03d}_frame_{start_frame}-{end_frame}.mp4")
        
        # 计算时间戳（用于音频同步）
        start_time = start_frame / fps
        end_time = end_frame / fps
        duration = end_time - start_time
        
        # 修正：确保音频有最小时长（至少一帧的时长）
        min_duration = float(1.0 / fps)-0.001  # 一帧的时长
        if duration < min_duration:
            duration = min_duration
            end_time = start_time + duration
            print(f"警告: 场景 {idx} 时长过短，已调整为 {duration:.4f}秒")
        
        try:
            # 修正：对于单帧场景使用 eq，多帧场景使用 between
            if start_frame == end_frame:
                vf_filter = f"select='eq(n\\,{start_frame})',setpts=PTS-STARTPTS"
            else:
                vf_filter = f"select='between(n\\,{start_frame}\\,{end_frame-1})',setpts=PTS-STARTPTS"
            
            # 音频过滤器：确保 end_time > start_time
            af_filter = f"atrim={start_time:.6f}:{end_time:.6f},asetpts=PTS-STARTPTS"
            
            (
                ffmpeg
                .input(video_path)
                .output(
                    output_path,
                    vf=vf_filter,
                    af=af_filter,
                    video_bitrate='5M',
                    audio_bitrate='192k'
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            print(f"? 场景 {idx}: 帧 {start_frame}-{end_frame} ({duration:.2f}秒) -> {output_path}")
        
        except ffmpeg.Error as e:
            print(f"? 错误: 处理场景 {idx} 时失败")
            print(f"  起始帧: {start_frame}, 结束帧: {end_frame}")
            print(f"  时间范围: {start_time:.4f}s - {end_time:.4f}s (时长: {duration:.4f}s)")
            print(f"  错误信息: {e.stderr.decode()}")
            print()
    
    print(f"\n完成！所有片段已保存到 {output_dir} 目录")

def get_video_fps(video_path):
    clip = VideoFileClip(video_path)
    fps = clip.fps
    clip.close()
    
    return fps

def has_intersection(interval1, interval2):
    """
    判断两个区间是否相交
    interval1: (start1, end1)
    interval2: (start2, end2)
    返回: True 如果相交, False 如果不相交
    """
    start1, end1 = interval1
    start2, end2 = interval2
    # 两个区间相交的条件: start1 <= end2 and start2 <= end1
    return start1 <= end2 and start2 <= end1


def process_vision_scenes(vision_scenes, audio_frames):
    """
    处理视觉检测区间
    - 如果与音频区间相交,进入待审核
    - 否则直接保留
    """
    check_list=[]
    reserve_list=[]
    for vision_interval in vision_scenes:
        # 检查是否与任何音频区间相交
        has_audio_intersection = False
        
        for audio_interval in audio_frames:
            if has_intersection(vision_interval, audio_interval):
                has_audio_intersection = True
                break
        
        if has_audio_intersection:
            # 进入待审核
            check_list.append(vision_interval)
        else:
            # 直接保留
            reserve_list.append(vision_interval)
    return check_list, reserve_list


def get_detection_result(interval, video_path, detector):
    # ========== 优化:使用滚动缓存策略 ==========
    """获取片段识别结果,如果缓存中没有则识别并缓存"""
    global results_cache
    # 确保interval是tuple类型
    interval_key = tuple(interval) if isinstance(interval, list) else interval
    
    if interval_key not in results_cache:
        result = detector.process_frame_intervals(video_path, [tuple(interval)])[0]
        results_cache[interval_key] = result
    return results_cache[interval_key]


def split_long_clip(video_path, final_list, detector, model, device, threshold=0.296, time_threshold=6):
    """
    对长片段进行二次切分
    
    参数:
        video_path: 视频路径
        final_list: [(start_frame, end_frame), ...] 待处理的片段列表
        detector: VideoSubtitleDetector 实例
        model: AutoShot 视觉模型
        device: 'cuda' 或 'cpu'
        threshold: 视觉检测阈值
        time_threshold: 时长阈值(秒)
    
    返回:
        new_final_list: 切分后的片段列表
    """
    fps = get_video_fps(video_path)
    new_final_list = []
    
    for idx, (start_frame, end_frame) in enumerate(final_list):
        start_time = start_frame / fps
        end_time = end_frame / fps
        duration = end_time - start_time
        
        print(f"\n处理片段 {idx}: 帧 {start_frame}-{end_frame}, 时长 {duration:.2f}秒")
        
        # 情况1: 不超过阈值,直接保留
        if duration <= time_threshold:
            new_final_list.append((start_frame, end_frame))
            print(f"✓ 时长合适,保留")
            continue
        
        # 情况2: 超过阈值,需要切分
        print(f"超过{time_threshold}秒,进行语音识别...")

        # 语音识别
        detection_result = get_detection_result((start_frame, end_frame),video_path,detector)
        # detection_result = detector.process_frame_intervals(video_path, [(start_frame, end_frame)])[0]
        content = detection_result.get('content', '')
        
        # 去除标点符号,统计实际文字数量
        text_only = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', content)
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text_only)  # 只匹配基本汉字
        char_count = len(chinese_chars)
        
        print(f"识别内容: {content}")
        print(f"中文数量: {char_count}")
        
        # 分支1: 文字<=2个,使用视觉+音频切分
        if char_count <= 2:
            print(f"文字较少,使用视觉切分")
            sub_segments = visual_audio_split(
                video_path, 
                start_frame, 
                end_frame, 
                fps,
                model, 
                device, 
                threshold,
                time_threshold
            )
            new_final_list.extend(sub_segments)
        
        # 分支2: 文字>2个,按句子切分
        else:
            print(f"文字较多,按句子切分")
            sub_segments = sentence_split(
                content,
                detection_result.get('segments', []),
                start_frame,
                end_frame,
                fps
            )
            new_final_list.extend(sub_segments)
    
    return new_final_list


def visual_audio_split(video_path, start_frame, end_frame, fps, model, device, threshold, time_threshold):
    """
    视觉+音频切分策略
    1. 先用视觉(AutoShot)切分
    2. 如果还有片段>time_threshold,用音频突变点继续切
    """
    print(f"[视觉切分] 开始...")
    
    # 提取子片段进行视觉检测
    sub_predictions, sub_boundaries, sub_scenes = predict_video_segment(
        video_path, 
        start_frame, 
        end_frame, 
        model, 
        device, 
        threshold
    )

    # 将相对帧号转换为绝对帧号
    absolute_scenes = [(s[0] + start_frame, s[1] + start_frame) for s in sub_scenes]
    
    print(f"[视觉切分] 检测到 {len(absolute_scenes)} 个子片段")
    
    # 检查是否还有超长片段
    result = []
    for sub_start, sub_end in absolute_scenes:
        sub_duration = (sub_end - sub_start) / fps
        
        if sub_duration <= time_threshold:
            result.append((sub_start, sub_end))
            print(f"子片段 {sub_start}-{sub_end} ({sub_duration:.2f}秒) 合适")
        else:
            print(f"子片段 {sub_start}-{sub_end} ({sub_duration:.2f}秒) 仍超长,使用音频切分")
            # 继续用音频突变点切分
            audio_segments = audio_energy_split(
                video_path,
                sub_start,
                sub_end,
                fps,
                time_threshold
            )
            result.extend(audio_segments)
    
    return result


def audio_energy_split(video_path, start_frame, end_frame, fps, time_threshold):
    """
    基于音频能量突变点切分
    """
    print(f"[音频切分] 开始...")
    
    # 提取音频片段
    start_time = start_frame / fps
    end_time = end_frame / fps
    
    # 打开视频获取总时长
    clip = VideoFileClip(video_path)
    video_duration = clip.duration
    
    # 确保时间范围在视频时长内
    start_time = max(0, min(start_time, video_duration))
    end_time = max(start_time, min(end_time, video_duration))
    
    # 检查时间范围是否有效
    if end_time <= start_time:
        print(f"[音频切分] 时间范围无效: {start_time:.2f}s - {end_time:.2f}s")
        clip.close()
        return [(start_frame, end_frame)]

    audio_clip = clip.subclipped(start_time, end_time).audio
    
    if audio_clip is None:
        # 没有音频,强制均匀切分
        print(f"无音频,强制均匀切分")
        return force_split_evenly(start_frame, end_frame, fps, time_threshold)
    
    # 获取音频数据
    audio_array = audio_clip.to_soundarray(fps=22050)
    if len(audio_array.shape) == 2:
        audio_array = audio_array.mean(axis=1)  # 转单声道
    
    clip.close()
    
    # 计算短时能量
    frame_length = 2048
    hop_length = 512
    energy = librosa.feature.rms(y=audio_array, frame_length=frame_length, hop_length=hop_length)[0]
    
    # 检测能量突降点
    energy_drops = []
    threshold_ratio = 0.6  # 能量下降到60%以下认为是突变
    
    for i in range(1, len(energy) - 1):
        # 检测局部最小值且能量明显下降
        if energy[i] < energy[i-1] * threshold_ratio and energy[i] < energy[i+1]:
            time_sec = librosa.frames_to_time(i, sr=22050, hop_length=hop_length)
            energy_drops.append(time_sec)
    
    print(f"[音频切分] 检测到 {len(energy_drops)} 个能量突变点")
    
    if not energy_drops:
        # 没有明显的突变点,强制均匀切分
        print(f"无明显突变,强制均匀切分")
        return force_split_evenly(start_frame, end_frame, fps, time_threshold)
    
    # 在能量突变点处切分
    duration = end_time - start_time
    boundaries = [0] + energy_drops + [duration]
    
    # 贪心选择切分点,尽量接近time_threshold
    result = []
    current_start = 0
    
    for i in range(1, len(boundaries)):
        segment_duration = boundaries[i] - current_start
        
        #确保分割的音频大于time_threshold-2秒
        if segment_duration >= (time_threshold - 2) or i == len(boundaries) - 1:
            # 转换回帧号
            abs_start = int(start_frame + current_start * fps)
            abs_end = int(start_frame + boundaries[i] * fps) - 1
            result.append((abs_start, abs_end))
            print(f"切分: {abs_start}-{abs_end} ({boundaries[i] - current_start:.2f}秒)")
            current_start = boundaries[i]
    
    return result


def sentence_split(content, segments, start_frame, end_frame, fps):
    """
    按句子切分
    segments: whisper返回的带时间戳的segments
    """
    print(f"[句子切分] 开始...")
    
    # 使用正则表达式按句子分割
    sentence_pattern = r'[。！？.!?]+'
    sentences = re.split(sentence_pattern, content)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 只保留中文字符
    chinese_only_sentences = []
    for sentence in sentences:
        # 使用正则表达式提取中文字符（包括扩展区域）
        chinese_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', sentence)
        chinese_text = ''.join(chinese_chars)
        if chinese_text:  # 只添加非空的中文文本
            chinese_only_sentences.append(chinese_text)
    
    sentences = chinese_only_sentences
    
    print(f"[句子切分] 识别到 {len(sentences)} 个句子")
    
    result = []

    # 如果有详细的segments信息(带时间戳)
    if segments and len(segments) > 0:
        # 尝试从segments中提取句子边界
        current_start_frame = start_frame
        current_text = ""
        
        for seg in segments:
            seg_text = seg.get('text', '').strip()
            current_text += seg_text
            
            # 检查是否包含句子结束标记
            if re.search(sentence_pattern, seg_text):
                # 句子结束,创建片段
                seg_end_time = seg.get('end', 0)
                current_end_frame = round(start_frame + seg_end_time * fps)
                
                if current_end_frame > current_start_frame:
                    result.append((current_start_frame, current_end_frame))
                    print(f"句子片段: {current_start_frame}-{current_end_frame}")
                    current_start_frame = current_end_frame + 1
                    current_text = ""
        
        # 添加最后一个片段
        if current_start_frame < end_frame:
            result.append((current_start_frame, end_frame))
            print(f"句子片段: {current_start_frame}-{end_frame}")
    
    # else:
    #     # 没有详细时间戳,均匀分割
    #     print(f"无详细时间戳,按句子数量均匀切分")
    #     total_duration = end_frame - start_frame
    #     sentence_count = len(sentences)
        
    #     for i in range(sentence_count):
    #         seg_start = int(start_frame + (total_duration * i / sentence_count))
    #         seg_end = int(start_frame + (total_duration * (i + 1) / sentence_count))
    #         result.append((seg_start, seg_end))
    #         print(f"句子片段: {seg_start}-{seg_end}")
    
    return result if result else [(start_frame, end_frame)]


def predict_video_segment(video_path, start_frame, end_frame, model, device, threshold):
    """
    对视频的某个片段进行镜头边界检测
    """
    # 读取指定帧范围
    frames = get_frames(video_path)
    segment_frames = frames[start_frame:end_frame+1]
    
    # 预测
    predictions = []
    for batch in get_batches(segment_frames):
        one_hot = predict_batch(model, batch, device)
        predictions.append(one_hot[25:75])
    
    predictions = np.concatenate(predictions, 0)[:len(segment_frames)]
    
    # 检测边界
    boundaries_binary = (predictions > threshold).astype(np.uint8)
    shot_boundaries = np.where(boundaries_binary == 1)[0].tolist()
    
    # 转换为场景
    scenes = []
    if len(shot_boundaries) > 0:
        start = 0
        for boundary in shot_boundaries:
            if boundary > start:
                scenes.append([start, boundary])
            start = boundary + 1
        if start < len(segment_frames):
            scenes.append([start, len(segment_frames) - 1])
    else:
        scenes.append([0, len(segment_frames) - 1])
    
    return predictions, shot_boundaries, scenes


def force_split_evenly(start_frame, end_frame, fps, time_threshold):
    """
    强制均匀切分(兜底策略)
    """
    duration = (end_frame - start_frame) / fps
    n_segments = int(np.ceil(duration / time_threshold))
    
    result = []
    frames_per_segment = (end_frame - start_frame) / n_segments
    
    for i in range(n_segments):
        seg_start = int(start_frame + i * frames_per_segment)
        seg_end = int(start_frame + (i + 1) * frames_per_segment) - 1
        if i == n_segments - 1:
            seg_end = end_frame  # 确保最后一段到达终点
        result.append((seg_start, seg_end))
    
    print(f"强制切分为 {n_segments} 段")
    return result


def format_time_label(seconds):
    """将秒数转换为 HH:MM:SS.mmm 格式"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((seconds - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"

def save_breaks_json(video_name, final_split_list, fps, output_dir="./output_json"):
    """
    保存视频切分结果为JSON格式
    
    参数:
        video_name: 视频名称（不含扩展名）
        final_split_list: [(start_frame, end_frame), ...] 切分后的片段列表
        fps: 视频帧率
        output_dir: JSON输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建breaks列表（只保留切分点，即每个片段的结束帧）
    breaks = []
    for i, (start_frame, end_frame) in enumerate(final_split_list):
        # 跳过最后一个片段的结束帧（因为它是视频末尾）
        if i < len(final_split_list) - 1:
            # 计算切分点的时间（使用片段结束帧）
            break_time = end_frame / fps
            time_label = format_time_label(break_time)
            
            breaks.append({
                "time": round(break_time, 6),
                "label": time_label,
                "type": "strong"
            })
    
    # 构建最终JSON结构
    result = {
        "file": f"{video_name}.mp4",
        "breaks": breaks
    }
    
    # 保存JSON文件
    json_path = os.path.join(output_dir, f"{video_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✓ JSON已保存: {json_path}")
    print(f"  检测到 {len(breaks)} 个切分点")
    
    return json_path

def main():
    # ==================== 配置参数 ====================
    # 1. 视频路径
    VIDEO_DIR = "/home/liuyd/projects/AVT/AutoShot/video_sample/sample"
    
    # 2. 模型权重路径
    CHECKPOINT_PATH = "./autoshot.pth"
    
    # 3. 检测阈值（可调整，0.296 是论文最优值）
    THRESHOLD = 0.296
    
    # 4. 输出目录
    OUTPUT_DIR = "./output_json1"
    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    global results_cache

    # ==================== 初始化 ====================
    # 选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载视觉模型
    model = load_model(CHECKPOINT_PATH, device)
    # 加载语音识别模型
    detector = VideoSubtitleDetector(use_gpu=True)

    #加载豆包检测器
    API_KEY = "sk-mNEEbls0kwB6f2xFN94dtm4gUwXHY1Lmw6aL5bKqaO1HdCwA"
    Base_URL = "https://api.zetatechs.com/v1/chat/completions"
    checker = DoubaoCompletenessChecker(api_key=API_KEY,endpoint=Base_URL )
    
    # ==================== 处理视频 ====================
    # 获取所有视频文件
    video_files = []
    
    if os.path.isfile(VIDEO_DIR):
        # 单个视频文件
        video_files = [VIDEO_DIR]
    elif os.path.isdir(VIDEO_DIR):
        # 目录中的所有 mp4 文件
        video_files = [
            os.path.join(VIDEO_DIR, f) 
            for f in os.listdir(VIDEO_DIR) 
            if f.endswith('.mp4')
        ]
    else:
        raise ValueError(f"Invalid path: {VIDEO_DIR}")
    
    print(f"Found {len(video_files)} video(s) to process\n")
    
    # 存储所有结果
    all_vision_results = {}
    
    # 逐个处理视频
    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(video_files)}] Processing: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        
        # 预测
        predictions, shot_boundaries, vision_scenes = predict_video(
            video_path, model, device, THRESHOLD
        )
        
        # 保存结果
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        all_vision_results[video_name] = {
            'vision_predictions': predictions,        # 每帧的概率
            'vision_shot_boundaries': shot_boundaries, # 边界帧号
            'vision_scenes': vision_scenes              # 镜头片段边界帧
        }
        
        # VAD 检测音频活动区间
        audio_dir = Path("./audio")
        audio_output_path = audio_dir / f"{video_name}.wav"
        audio_file = extract_audio(video_path, str(audio_output_path))
        #分离人声和背景声
        vocals_path = separate_vocals_cli(audio_file)
        audio_segments, audio_frames = detect_voice_activity(vocals_path, video_path, aggressiveness=0)

        all_vision_results[video_name] = {
            'vad_sec': audio_segments,        # 检测到音频活动的秒范围
            'vad_frames': audio_frames,       # 音频活动的帧范围
        }

        check_list, reserve_list = process_vision_scenes(vision_scenes, audio_frames)


        # 用于跟踪合并状态
        i = 0
        while i < len(check_list) - 1:
            start_current, end_current = check_list[i]
            j = i + 1
            
            # 尝试找到所有需要连续合并的片段
            should_merge = False
            merge_end_index = i  # 记录最后需要合并到的索引
            
            while j < len(check_list):
                start_next, end_next = check_list[j]
                
                # 检查时间间隔
                if start_next - end_current > 2:
                    break
                
                # ========== 使用缓存函数获取识别结果 ==========
                current_result = get_detection_result((start_current, end_current),video_path,detector)
                next_result = get_detection_result((start_next, end_next),video_path,detector)
                
                # 检查是否都有内容
                if current_result['content'] == '' or next_result['content'] == '':
                    break
                
                if len(current_result['content']) <3 or len(next_result['content']) <3:
                    break

                # 分割句子
                sentences_current = re.split(r'[。！？.!?]+', current_result['content'])
                sentences_next = re.split(r'[。！？.!?]+', next_result['content'])
                
                # 过滤空字符串
                sentences_current = [s.strip() for s in sentences_current if s.strip()]
                sentences_next = [s.strip() for s in sentences_next if s.strip()]
                
                if not sentences_current or not sentences_next:
                    break
                
                # 获取最后一句和第一句
                last_sentence = sentences_current[-1]
                first_sentence = sentences_next[0]
                combined_sentence = last_sentence + first_sentence
                # 删除中英文的标点符号
                combined_sentence = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', combined_sentence)
                
                # 检查完整性
                last_sentence_results = checker.check_completeness(last_sentence)
                first_sentence_results = checker.check_completeness(first_sentence)
                combined_sentence_results = checker.check_completeness(combined_sentence)
                
                # 检查所有检查是否成功
                if (last_sentence_results['status'] != 'success' or 
                    first_sentence_results['status'] != 'success' or 
                    combined_sentence_results['status'] != 'success'):
                    break
                
                # 判断是否需要合并
                last_complete = last_sentence_results['is_complete']
                first_complete = first_sentence_results['is_complete']
                combined_complete = combined_sentence_results['is_complete']
                
                # 获取置信度
                last_confidence = last_sentence_results.get('confidence', 0)
                first_confidence = first_sentence_results.get('confidence', 0)
                combined_confidence = combined_sentence_results.get('confidence', 0)
                
                need_merge = False
                
                # 情况1: 两个都完整 - 不需要合并
                if last_complete and first_complete:
                    need_merge = False
                    break
                
                # 情况2: 两个都不完整,但合并后完整 - 需要合并
                elif not last_complete and not first_complete:
                    if combined_complete:
                        need_merge = True
                    else:
                        # 合并后仍不完整,可以考虑置信度
                        if last_confidence > 0.7 and first_confidence > 0.7:
                            need_merge = True
                        else:
                            need_merge = False
                            break
                
                # 情况3: 一个完整一个不完整
                else:
                    if not last_complete:
                        # 当前片段最后一句不完整
                        if last_confidence > 0.6 and combined_complete:
                            need_merge = True
                        elif last_confidence <= 0.6:
                            need_merge = False
                            break
                        else:
                            need_merge = False
                            break
                    else:  # not first_complete
                        # 下一个片段第一句不完整
                        if first_confidence > 0.6 and combined_complete:
                            need_merge = True
                        elif first_confidence <= 0.6:
                            need_merge = False
                            break
                        else:
                            need_merge = False
                            break
                
                if need_merge:
                    should_merge = True
                    merge_end_index = j
                    # 更新当前片段的结束时间,继续检查是否需要与下一个合并
                    end_current = end_next
                    j += 1
                else:
                    break
            
            # 根据合并结果添加到reserve_list
            if should_merge:
                # 创建合并后的片段
                merged_interval = (start_current, check_list[merge_end_index][1])
                reserve_list.append(merged_interval)
                # 清空已合并片段的缓存,它们已经不需要了
                for k in range(i, merge_end_index + 1):
                    interval_to_remove = tuple(check_list[k]) if isinstance(check_list[k], list) else check_list[k]
                    if interval_to_remove in results_cache:
                        del results_cache[interval_to_remove]
                # 跳过已经合并的片段
                i = merge_end_index + 1
            else:
                # 不需要合并,保留当前片段
                reserve_list.append((start_current, end_current))
                # # 清空当前片段的缓存
                # current_interval = (start_current, end_current)
                # if current_interval in results_cache:
                #     del results_cache[current_interval]
                i += 1

        # 添加最后一个片段(如果还没有被处理)
        if i == len(check_list) - 1:
            reserve_list.append(check_list[i])

        # 按照每个元素的第一个数字排序
        sorted_list = sorted(reserve_list, key=lambda x: x[0])

        print(sorted_list)

        # 保存JSON格式的切分结果
        fps = get_video_fps(video_path)
        save_breaks_json(
            video_name=video_name,
            final_split_list=sorted_list,
            fps=fps,
            output_dir=OUTPUT_DIR 
        )


if __name__ == "__main__":
    main()