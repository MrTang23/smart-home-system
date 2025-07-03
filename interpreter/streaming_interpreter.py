# 流式同声传译引擎，封装 ASR→Mapper→MT→TTS 的完整流水线
import re

import sounddevice as sd
import numpy as np
import threading
import queue
import time

import webrtcvad

from asr.whisper_asr import WhisperASR
from mapper.asr2mt_mapper import ASR2MTMapper
from mt.feature_injector import FeatureInjectorTranslator
from tts.coqui_tts import CoquiTTS

class StreamingInterpreter:
    def __init__(self,
                 asr_model: str = "small",
                 sample_rate: int = 16000,
                 block_duration: float = 0.5):
        """
        asr_model: Whisper 模型名称
        sample_rate: 音频采样率
        block_duration: 每次从麦克风读取的音频块时长（秒）
        """
        # 模块实例化
        self.asr = WhisperASR(model_name=asr_model, device="cpu")
        self.mapper = ASR2MTMapper(input_dim=768)
        self.translator = FeatureInjectorTranslator(model_name="opus-mt")
        self.tts = CoquiTTS()

        # 流控参数
        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * block_duration)

        # 两条队列：音频采集→识别队列，文本→翻译队列
        self.audio_queue = queue.Queue()
        self.translate_queue = queue.Queue()

        self.last_text = ""
        self.vad = webrtcvad.Vad(2)  # 2: 中等灵敏度
        self.frame_ms = 30  # 每帧时长 30ms
        self.frame_size = int(self.sample_rate * self.frame_ms / 1000)
        self.min_speech_frames = 3  # 至少 3 帧才认为是真正语音
        self.max_silence_frames = int(500 / self.frame_ms)  # 0.5s 静音触发断句

    def start(self):
        """启动音频流与后台线程，进入主循环。"""
        print("同传系统启动中… 讲话后稍等即可听到翻译结果")
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            callback=self._audio_callback
        ):
            threading.Thread(target=self._recognize_thread, daemon=True).start()
            threading.Thread(target=self._translate_thread, daemon=True).start()

            # 主线程保持存活
            while True:
                time.sleep(0.1)

    def _audio_callback(self, indata, frames, time_info, status):
        """回调：每 block_duration 秒将麦克风数据推到音频队列。"""
        if status:
            print("AudioStreamWarning:", status)
        self.audio_queue.put(indata[:, 0].copy())

    def _frame_generator(self):
        """从 self.audio_queue 获取帧（np.ndarray），每帧时长 frame_ms"""
        buffer = np.zeros((0,), dtype=np.float32)
        while True:
            chunk = self.audio_queue.get()
            buffer = np.concatenate([buffer, chunk])
            # 当 buffer 足够一帧时，切出一帧并保留剩余
            while len(buffer) >= self.frame_size:
                frame = buffer[:self.frame_size]
                buffer = buffer[self.frame_size:]
                yield frame

    def _recognize_thread(self):
        speech_frames = 0
        silence_frames = 0
        segment = bytearray()

        for frame in self._frame_generator():
            is_speech = self.vad.is_speech(
                (frame * 32767).astype(np.int16).tobytes(),
                sample_rate=self.sample_rate
            )
            # 结合 RMS 做二次验证
            rms = np.sqrt(np.mean(frame ** 2))
            if is_speech and rms > 0.01:
                speech_frames += 1
                silence_frames = 0
            else:
                if speech_frames > 0:
                    silence_frames += 1

            segment.extend((frame * 32767).astype(np.int16).tobytes())

            # 进入说话状态后，若静音超过阈值则断句
            if speech_frames >= self.min_speech_frames and silence_frames >= self.max_silence_frames:
                audio_segment = np.frombuffer(bytes(segment), np.int16).astype(np.float32) / 32767
                # 调用转录与后续处理
                text = self.asr.transcribe(audio_segment)
                if text and text != self.last_text:
                    self.last_text = text
                    print(f"[识别] {text}")
                    feats = self.asr.extract_features(audio_segment)
                    mapped = self.mapper.map_features(feats)
                    self.translate_queue.put((text, mapped))

                # 重置状态机与缓存，只保留少量尾部预热
                segment = bytearray()
                tail_frames = int(self.sample_rate * 0.2)  # 0.2s 预热
                segment.extend((audio_segment[-tail_frames:] * 32767).astype(np.int16).tobytes())
                speech_frames = silence_frames = 0

    def _translate_thread(self):
        """后台线程：从 translate_queue 取 (text,features) → 翻译 → TTS 播报。"""
        while True:
            text, features = self.translate_queue.get()
            # 翻译
            translated = self.translator.translate(text, features)
            print(f"[翻译] {translated}")

            # 按句号、问号、感叹号等分句
            # 保留标点尾部，确保语气和停顿
            segments = re.split(r'(?<=[。！？.!?])', translated)

            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue

                # 调用 TTS，并捕获可能的“max_decoder_steps”错误
                try:
                    self.tts.speak(seg)
                except RuntimeError as e:
                    print(f"[TTS 错误] 在合成 “{seg}” 时失败：{e}")
                    continue
