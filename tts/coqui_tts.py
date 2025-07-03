# 本模块用于将文本转换为语音并实时播放

from TTS.api import TTS
import tempfile
import sounddevice as sd
import soundfile as sf

class CoquiTTS:
    def __init__(self, model_name="tts_models/en/ljspeech/tacotron2-DDC"):
        """
        初始化 TTS 模型
        """
        self.tts = TTS(model_name)

    def speak(self, text):
        """
        合成语音并播放
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            self.tts.tts_to_file(text=text, file_path=tmp.name)
            data, sr = sf.read(tmp.name)
            sd.play(data, sr)
            sd.wait()
