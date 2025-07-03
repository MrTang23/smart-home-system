# 封装 WebRTC VAD 工具，基于帧检测是否有语音

import webrtcvad
import numpy as np

class VADWrapper:
    def __init__(self, aggressiveness=2, sample_rate=16000, frame_duration=30):
        """
        aggressiveness：0~3，越高越严格（默认 2）
        frame_duration：帧长（毫秒），可选值为 10, 20, 30
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)

    def is_speech(self, audio: np.ndarray) -> bool:
        """
        判断一帧是否为语音（音频为float32，需要转为int16）
        """
        if len(audio) < self.frame_size:
            return False
        audio = audio[:self.frame_size]
        int16_audio = (audio * 32768).astype(np.int16).tobytes()
        return self.vad.is_speech(int16_audio, self.sample_rate)

    def is_silence_block(self, audio_block: np.ndarray, required_silent_frames=3) -> bool:
        """
        判断一整块音频是否是静音（连续3帧无语音）
        """
        silent_frames = 0
        for i in range(0, len(audio_block) - self.frame_size + 1, self.frame_size):
            frame = audio_block[i:i + self.frame_size]
            if not self.is_speech(frame):
                silent_frames += 1
                if silent_frames >= required_silent_frames:
                    return True
        return False
