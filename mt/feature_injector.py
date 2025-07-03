# 带特征注入的翻译封装
# 将 ASR→Mapper 得到的特征，转换成伪 token，拼接到文本前，再调用 EasyNMT 翻译，最后去掉 token，整个流程封装在此模块。

import torch
from mt.easy_nmt import EasyNMTTranslator

class FeatureInjectorTranslator:
    def __init__(self,
                 model_name: str = 'nllb-200-distilled-600M',
                 token_prefix: str = '<feat_',
                 token_suffix: str = '>'):
        """
        model_name: 传给 EasyNMTTranslator 的模型名
        token_prefix/suffix: 生成伪 token 的前后缀
        """
        self.base = EasyNMTTranslator(model_name)
        self.prefix = token_prefix
        self.suffix = token_suffix

    @staticmethod
    def _make_token(feat: torch.Tensor) -> str:
        """
        从特征张量 feat 生成一个伪 token 字符串
        这里用特征均值 * 1000, 取绝对值, 再 mod 100000 生成一个稳定 id
        """
        # 计算特征全局平均值
        val = feat.mean().item()
        # 放大并取整
        idx = int(abs(val) * 1000) % 100000
        return f"{idx}"

    def translate(self, text: str, features: torch.Tensor,
                  source_lang: str = 'zh', target_lang: str = 'en') -> str:
        """
        将 features 和 text 一起翻译：
        1. 根据 features 生成伪 token
        2. 拼接："<feat_id> " + text
        3. 调用 base.translate
        4. 去除翻译结果中的 token
        返回纯净的翻译文本
        """
        # 生成 token id
        token_id = self._make_token(features)
        token = f"{self.prefix}{token_id}{self.suffix}"

        # 拼接到文本前
        injected = f"{token} {text}"

        # 调用 EasyNMT 翻译
        result = self.base.translate(injected, source_lang=source_lang, target_lang=target_lang)

        # 去掉翻译结果中的 token，并剔除多余空格
        clean = result.replace(token, '').strip()
        return clean
