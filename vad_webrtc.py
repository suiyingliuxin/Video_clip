import webrtcvad
import wave
import contextlib
import subprocess
from moviepy import VideoFileClip
import tempfile
from pathlib import Path 
from spleeter.separator import Separator
import gc
import tensorflow as tf
from contextlib import contextmanager

def extract_audio(video_path, output_audio="output.wav"):
    """
    提取音频并转为WAV格式
    """
    subprocess.run([
        'ffmpeg',
        '-i', video_path,           # 输入视频
        '-vn',                       # 不处理视频流
        '-acodec', 'pcm_s16le',     # 音频编码格式
        '-ar', '16000',             # 采样率16kHz（VAD常用）
        '-ac', '1',                  # 单声道
        '-y',                        # 覆盖输出
        output_audio
    ], check=True, capture_output=True)
    
    return output_audio

def get_video_fps(video_path):
    clip = VideoFileClip(video_path)
    fps = clip.fps
    clip.close()
    
    return fps


def detect_voice_activity(audio_path, video_path, aggressiveness=3):
    """
    aggressiveness: 0-3,数值越大，检测越“敏感”
    返回: [(start_time, end_time, start_frame, end_frame), ...] 人声区间列表（包含时间戳和帧号）
    """
    # 获取视频帧率
    fps = get_video_fps(video_path)
    
    vad = webrtcvad.Vad(aggressiveness)
    
    # 读取音频（需要16kHz, 16bit, mono）
    with contextlib.closing(wave.open(audio_path, 'rb')) as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    
    # 30ms窗口滑动检测
    frame_duration = 30  # ms
    frame_size = int(sample_rate * frame_duration / 1000) * 2  # 字节数
    
    voice_segments = []
    frame_segments = []
    is_speech = False
    start_time = 0
    
    for i, offset in enumerate(range(0, len(frames), frame_size)):
        frame = frames[offset:offset + frame_size]
        if len(frame) != frame_size:
            break
        
        timestamp = i * frame_duration / 1000.0  # 转为秒
        
        # VAD判断
        if vad.is_speech(frame, sample_rate):
            if not is_speech:
                start_time = timestamp
                is_speech = True
        else:
            if is_speech:
                # 计算起始和结束帧号
                start_frame = int(start_time * fps)
                end_frame = int(timestamp * fps)
                voice_segments.append((start_time, timestamp))
                frame_segments.append((start_frame,end_frame))
                is_speech = False
    
    # 处理最后一个片段
    if is_speech:
        # 计算起始和结束帧号
        start_frame = int(start_time * fps)
        end_frame = int(timestamp * fps)
        voice_segments.append((start_time, timestamp))
        frame_segments.append((start_frame,end_frame))
    
    return voice_segments, frame_segments


def separate_vocals_cli(audio_path):
    """
    使用命令行Demucs分离人声
    """
    import os
    
    # 创建输出目录
    output_dir = tempfile.mkdtemp()
    # output_dir = "./audio"
    
    # 运行Demucs
    cmd = [
        'demucs',
        '-n', 'htdemucs',  # 使用最新模型
        '--two-stems', 'vocals',  # 只分离人声,速度更快
        '-o', output_dir,
        audio_path
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Demucs输出路径: output_dir/htdemucs/音频名/vocals.wav
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    vocals_path = os.path.join(output_dir, 'htdemucs', audio_name, 'vocals.wav')
    
    # 重采样到16000Hz
    final_path = audio_path.replace('.wav', '_vocals_16k.wav')
    cmd = [
        'ffmpeg',
        '-i', vocals_path,
        '-ar', '16000',
        '-ac', '1',
        '-y',
        final_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 清理临时目录
    import shutil
    shutil.rmtree(output_dir)
    
    return final_path