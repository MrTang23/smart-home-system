# 本模块封装了 Whisper 模型用于语音识别，并支持输出编码器隐向量供后续模块使用

import whisper
import torch
import numpy as np


class WhisperASR:
    def __init__(self, model_name="tiny", device="cpu", max_len=300):
        """
        初始化 Whisper 模型
        :param model_name: Whisper 模型名 (tiny/base/small/…)
        :param device: 设备 ("cpu" 或 "cuda")
        :param max_len: 隐向量时序最大帧数
        """
        self.device = device
        self.model = whisper.load_model(model_name).to(device)
        # 隐向量维度 (small/medium/large-v3 等版本一致为 768)
        self.dim = self.model.encoder.d_model if hasattr(self.model.encoder, "d_model") else 768
        self.max_len = max_len

    def transcribe(self, audio: np.ndarray, language="zh") -> str:
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        options = whisper.DecodingOptions(
            language=language,
            without_timestamps=True,
            fp16=False
        )
        result = whisper.decode(self.model, mel, options)
        return result.text

    # 提取并预处理 Whisper Encoder 中间隐向量
    # 返回: Tensor of shape [T', 1, D], T' == max_len
    def extract_features(self, audio: np.ndarray) -> torch.Tensor:

        # 生成梅尔谱图并调用 Encoder
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        mel = mel.unsqueeze(0)                          # [1, 80, seq_len]
        with torch.no_grad():
            feats = self.model.encoder(mel)             # [1, T, D]

        # 维度转换：去除 batch 轴并新增 batch=1
        feats = feats.squeeze(0).unsqueeze(1)           # [T, 1, D]

        # 通道归一化（零均值、单位方差）
        mean = feats.mean(dim=0, keepdim=True)
        std  = feats.std(dim=0, keepdim=True).clamp(min=1e-5)
        feats = (feats - mean) / std

        # 定长截断或填充至 max_len
        T, B, D = feats.shape
        if T > self.max_len:
            feats = feats[:self.max_len]
        elif T < self.max_len:
            pad = torch.zeros((self.max_len - T, B, D), device=self.device)
            feats = torch.cat([feats, pad], dim=0)

        return feats
