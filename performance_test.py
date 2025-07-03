# compare_performance.py

import soundfile as sf
import numpy as np
import time
import threading
import queue

from asr.whisper_asr import WhisperASR
from mt.easy_nmt import EasyNMTTranslator
from mt.feature_injector import FeatureInjectorTranslator
from tts.coqui_tts import CoquiTTS
import webrtcvad

# 流式测试核心，复用你的 StreamingInterpreter 逻辑，仅剥离播放
class StreamingTester:
    def __init__(self, wav_path,
                 block_duration=0.5,
                 frame_ms=30,
                 min_speech=3,
                 max_silence_ms=500):
        audio, sr = sf.read(wav_path)
        assert sr == 16000, "请使用 16 kHz WAV"
        self.audio = audio.astype(np.float32)
        self.sr = sr
        self.block_size = int(sr * block_duration)

        self.asr    = WhisperASR(model_name="small", device="cpu")
        self.mapper = __import__("mapper.asr2mt_mapper", fromlist=["ASR2MTMapper"])\
                        .ASR2MTMapper(input_dim=768, verbose=False)
        self.trans  = FeatureInjectorTranslator(model_name="opus-mt")

        self.vad    = webrtcvad.Vad(2)
        self.frame_size = int(sr * frame_ms / 1000)
        self.min_speech = min_speech
        self.max_silence = int(max_silence_ms / frame_ms)

        self.audio_q = queue.Queue(maxsize=16)
        self.latencies = []

    def producer(self):
        idx = 0
        while idx < len(self.audio):
            chunk = self.audio[idx:idx+self.block_size]
            ts = time.perf_counter()
            self.audio_q.put((chunk, ts))
            idx += self.block_size

    def frame_generator(self):
        buf = np.zeros(0, dtype=np.float32)
        ts0 = None
        while True:
            chunk, ts = self.audio_q.get()
            buf = np.concatenate([buf, chunk])
            while len(buf) >= self.frame_size:
                frame = buf[:self.frame_size]
                buf   = buf[self.frame_size:]
                yield frame, ts

    def consumer(self):
        speech = silence = 0
        segment = bytearray()
        seg_start = None
        for frame, ts in self.frame_generator():
            if seg_start is None:
                seg_start = time.perf_counter()
                if not self.latencies:
                    self.start0 = ts
            is_s = self.vad.is_speech(
                (frame*32767).astype(np.int16).tobytes(), sample_rate=self.sr)
            rms = np.sqrt((frame**2).mean())
            if is_s and rms>0.01:
                speech += 1; silence = 0
            elif speech>0:
                silence += 1

            segment.extend((frame*32767).astype(np.int16).tobytes())

            if speech>=self.min_speech and silence>=self.max_silence:
                audio_seg = (np.frombuffer(bytes(segment),np.int16)
                             .astype(np.float32)/32767)
                t0 = time.perf_counter()
                txt  = self.asr.transcribe(audio_seg)
                feats= self.asr.extract_features(audio_seg)
                m    = self.mapper.map_features(feats)
                _    = self.trans.translate(txt, m)
                t1 = time.perf_counter()

                self.latencies.append((t0-self.start0, t1-t0))
                # 保留 0.2s 预热
                tail = int(0.2*self.sr)
                segment = bytearray((audio_seg[-tail:]*32767)
                                    .astype(np.int16).tobytes())
                speech = silence = 0
                seg_start = None

            # 结束条件，处理前5句后退出
            if len(self.latencies)>=5:
                break

    def run(self):
        t1 = threading.Thread(target=self.producer, daemon=True)
        t2 = threading.Thread(target=self.consumer, daemon=True)
        t1.start(); t2.start()
        t1.join(); t2.join()
        return self.latencies

def batch_pipeline(wav_path):
    audio, sr = sf.read(wav_path)
    assert sr==16000
    audio = audio.astype(np.float32)

    asr    = WhisperASR(model_name="small", device="cpu")
    mt     = EasyNMTTranslator(model_name="opus-mt")
    tts    = CoquiTTS()

    times = {}
    t0 = time.perf_counter()
    txt = asr.transcribe(audio)
    times["ASR 转录"] = (time.perf_counter()-t0)*1000

    t0 = time.perf_counter()
    en  = mt.translate(txt, source_lang="zh", target_lang="en")
    times["文本翻译"] = (time.perf_counter()-t0)*1000

    t0 = time.perf_counter()
    tts.speak(en)
    times["TTS 合成+播放"] = (time.perf_counter()-t0)*1000

    times["端到端批量"] = sum(times.values())
    return times

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", help="16kHz 单声道 WAV")
    args = parser.parse_args()

    print("== 批量流水测试 ==")
    batch = batch_pipeline(args.wav)
    for k,v in batch.items():
        print(f"{k:<15s}: {v:8.1f} ms")

    print("\n== 流式流水测试 (前5句) ==")
    st = StreamingTester(args.wav)
    lat = st.run()
    first = lat[0][1]*1000
    inc   = [x[1]*1000 for x in lat[1:]]
    avg_inc = sum(inc)/len(inc)
    print(f"首句延迟    : {first:8.1f} ms")
    print(f"后续平均增量: {avg_inc:8.1f} ms")
