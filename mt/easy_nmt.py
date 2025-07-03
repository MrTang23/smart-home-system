# 本模块使用 EasyNMT 执行中英文之间的机器翻译任务

from easynmt import EasyNMT

class EasyNMTTranslator:
    def __init__(self, model_name='opus-mt'):
        self.model = EasyNMT(model_name)

    def translate(self, text, source_lang='zh', target_lang='en'):
        return self.model.translate(text, source_lang=source_lang, target_lang=target_lang)
