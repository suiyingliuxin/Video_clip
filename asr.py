import cv2
import numpy as np
from PIL import Image
import pytesseract
from collections import defaultdict
import subprocess
import tempfile
import os
from funasr import AutoModel
import librosa
from transformers import pipeline

class VideoSubtitleDetector:
    def __init__(self, use_gpu=False):
        """
        初始化视频字幕检测器
        
        Args:
            use_gpu: 是否使用GPU加速
        """

        # 加载FunASR模型（一次性加载，避免重复初始化）
        print("加载FunASR模型...")
        
        # 方案1：完整模型（推荐）- 包含VAD、ASR、标点预测
        self.model = AutoModel(
            model="paraformer-zh",           # 中文语音识别模型
            vad_model="fsmn-vad",            # 语音活动检测（加速处理）
            vad_kwargs={"speech_noise_thres": 0.8},  # 默认的vad阈值为0.6
            punc_model="ct-punc",            # 标点预测模型
            spk_model="cam++",               # 说话人识别（可选）
            device="cuda:0" if use_gpu else "cpu"
        )
        
        #加载音频分类模型
        self.classifier = pipeline("audio-classification", 
                    model="MIT/ast-finetuned-audioset-10-10-0.4593")
        
        
        print("✓ FunASR模型加载完成")
    
    def separate_vocals_cli(self, audio_path):
        """
        使用命令行Demucs分离人声
        """
        # 创建输出目录
        output_dir = tempfile.mkdtemp()
        
        # 运行Demucs
        cmd = [
            'demucs',
            '-n', 'htdemucs',
            '--two-stems', 'vocals',
            '-o', output_dir,
            audio_path
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Demucs输出路径
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

    def extract_audio_segment(self, start_frame, end_frame, video_path):
        """
        从视频中提取指定帧范围的音频
        
        Args:
            start_frame: 起始帧
            end_frame: 结束帧
        
        Returns:
            str: 临时音频文件路径
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps
        
        # 创建临时音频文件
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        # 使用ffmpeg提取音频片段
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            temp_audio_path
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        cap.release()
        
        return temp_audio_path
    
    

    def transcribe_audio(self, audio_path, video_path):
        """
        使用FunASR进行语音识别
        """
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            result = self.model.generate(
                input=audio_path,
                batch_size_s=300,
                # 关键: 启用句子级别的时间戳
                sentence_timestamp=True  # ✅ 添加这个参数
            )
            # print(1)
            # print(result)
            # for item in result:
            #     print(item)

            if not result or len(result) == 0:
                cap.release()
                return {
                    'text': '',
                    'language': 'zh',
                    'segments': [],
                    'frames': []
                }
            
            # 获取完整文本
            text = result[0].get('text', '').strip()
            
            import re
            text = re.sub(r'[^\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef0-9\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # ✅ 新的处理逻辑: 使用句子级时间戳
            segments = []
            frames_list = []
            
            # 检查是否有句子级时间戳
            if 'sentence_info' in result[0] and result[0]['sentence_info']:
                # 有句子级时间戳
                for sentence_info in result[0]['sentence_info']:
                    sentence_text = sentence_info.get('text', '').strip()
                    if not sentence_text:
                        continue
                    
                    start_time = sentence_info['start'] / 1000.0  # 毫秒转秒
                    end_time = sentence_info['end'] / 1000.0
                    
                    start_frame = int(start_time * fps)
                    end_frame = int(end_time * fps)
                    
                    segments.append({
                        'start': start_time,
                        'end': end_time,
                        'text': sentence_text  # ✅ 使用分段的文本
                    })
                    
                    frames_list.append([start_frame, end_frame])
            
            cap.release()

            return {
                'text': text,
                'language': 'zh',
                'segments': segments,
                'frames': frames_list,
                'raw_result': result[0]
            }
            
        except Exception as e:
            print(f"FunASR识别失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'text': '',
                'language': 'zh',
                'segments': [],
                'frames': []
            }
        
    def detect_vocal_presence(self, audio_path, threshold=0.6):
        """检测音频中是否存在人声"""
        y, sr = librosa.load(audio_path, sr=16000)
        
        # 提取 MFCC 特征（人声在特定频段有特征）
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # 人声通常在 300Hz - 3400Hz
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        vocal_range_energy = np.mean(spectral_centroid)
        
        # 计算谱通量（人声变化更明显）
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        spectral_flux = np.std(onset_env)
        
        # 简单判断逻辑
        has_vocal = spectral_flux > threshold
        
        return has_vocal
    
    def process_frame_intervals(self, video_path, intervals, sample_rate=10):
        """
        处理帧区间列表
        
        Args:
            intervals: 帧区间列表 [(start, end), ...]
            sample_rate: 每隔多少帧采样一次进行字幕检测
        
        Returns:
            list: 处理结果列表
        """
        results = []
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        for idx, (start_frame, end_frame) in enumerate(intervals):
            print(f"\n处理帧区间 [{idx+1}/{len(intervals)}]: {start_frame} - {end_frame}")
            
            # 提取音频并识别
            audio_path = self.extract_audio_segment(start_frame, end_frame, video_path)
            
            # 可选：如果有人声音，需要人声分离，语音识别
            audio_path = self.separate_vocals_cli(audio_path)

            # 不存在的情况
            if not os.path.exists(audio_path):
                print("没有找到audio_path")
                result = {
                    'type': 'speech',
                    'frame_range': (start_frame, end_frame),
                    'time_range': (start_frame / fps, end_frame / fps),
                    'content': '',
                    'segments': []
                }
                results.append(result)
                continue

            # 使用音频分类模型,判断是人讲话还是其他类型
            result_class = self.classifier(audio_path, top_k=5)
            speech_items = [item for item in result_class if item['label'] == 'Speech']
            if speech_items and speech_items[0]['score'] > 0.6:
                try:
                    # 使用FunASR识别
                    transcription = self.transcribe_audio(audio_path, video_path)
                    
                    # 构建结果
                    if transcription['text'] != "":
                        result = {
                            'type': 'speech',
                            'frame_range': (start_frame, end_frame),
                            'time_range': (start_frame / fps, end_frame / fps),
                            'content': transcription['text'],
                            'language': transcription['language'],
                            'segments': transcription['segments']
                        }
                        print(f"✓ 语音识别: {result['content']}")
                    else:
                        result = {
                            'type': 'speech',
                            'frame_range': (start_frame, end_frame),
                            'time_range': (start_frame / fps, end_frame / fps),
                            'content': '',
                            'language': transcription['language'],
                            'segments': []
                        }
                        print(f"⚠ 未识别到语音内容")
                    
                except Exception as e:
                    print(f"❌ 处理失败: {e}")
                    result = {
                        'type': 'error',
                        'frame_range': (start_frame, end_frame),
                        'time_range': (start_frame / fps, end_frame / fps),
                        'content': '',
                        'error': str(e)
                    }
                finally:
                    # 删除临时音频文件
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                    # # 删除人声分离后的文件
                    # vocals_path = audio_path.replace('.wav', '_vocals_16k.wav')
                    # if os.path.exists(vocals_path):
                    #     os.remove(vocals_path)
            else:
                result = {
                    'type': 'speech',
                    'frame_range': (start_frame, end_frame),
                    'time_range': (start_frame / fps, end_frame / fps),
                    'content': '',
                    'segments': []
                }
                print(f"⚠ 未识别到语音内容")
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            
            results.append(result)

        cap.release()
        
        return results
    