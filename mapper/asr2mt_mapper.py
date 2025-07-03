# mapper/asr2mt_mapper.py
# 中文注释：ASR→MT 特征映射器（Mapper）模块
#   - 基于 Seq2Seq（Transformer Encoder）结构
#   - 内部使用多头自注意力 + 前馈网络
#   - 在 forward 中打印输入/输出形状和注意力权重示例

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ASR2MTMapper(nn.Module):
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 512,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 verbose: bool = True):
        """
        input_dim:   Whisper 隐向量维度
        hidden_dim:  前馈网络隐藏层大小
        nhead:       多头注意力头数
        num_layers:  Transformer Encoder 层数
        dropout:     Dropout 比例
        verbose:     是否打印调试信息
        """
        super().__init__()
        self.verbose = verbose

        # 构建 num_layers 层的 Transformer Encoder 每层结构
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(
                    embed_dim=input_dim,
                    num_heads=nhead,
                    dropout=dropout,
                    batch_first=False
                ),
                "attn_norm": nn.LayerNorm(input_dim),
                "ff1": nn.Linear(input_dim, hidden_dim),
                "ff2": nn.Linear(hidden_dim, input_dim),
                "ff_norm": nn.LayerNorm(input_dim),
                "dropout": nn.Dropout(dropout)
            })
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [seq_len, batch, input_dim]
        return out: [seq_len, batch, input_dim]
        """
        if self.verbose:
            print(f"[Mapper] 输入形状: {tuple(x.shape)}")

        seq_len, B, D = x.shape
        attn_weights_example = None

        for idx, layer in enumerate(self.layers):
            # 自注意力
            attn_out, attn_w = layer["self_attn"](
                x, x, x,
                need_weights=True,
                average_attn_weights=False
            )
            # attn_w: [B, nhead, seq_len, seq_len]
            if self.verbose and idx == 0:
                # 只打印第一层的注意力权重示例
                w0 = attn_w[0, 0, :5, :5].detach().cpu().numpy()

                plt.imshow(w0, cmap="viridis", aspect="auto")
                plt.title("Layer-1 Head-0 Attention")
                plt.xlabel("Key Position")
                plt.ylabel("Query Position")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig("fig4_attention.png", dpi=300)

                print(f"[Mapper] Layer {idx} 注意力权重示例（B=0,h=0,5×5）:\n{w0}")

            attn_weights_example = attn_w

            # 残差 + 归一化
            x = layer["attn_norm"](x + layer["dropout"](attn_out))

            # 前馈网络
            ff = layer["ff2"](F.relu(layer["ff1"](x)))
            x = layer["ff_norm"](x + layer["dropout"](ff))

        if self.verbose:
            print(f"[Mapper] 输出形状: {tuple(x.shape)}\n")

        # 如果需要外部可视化，这里也可以存 attn_weights_example
        self.last_attn = attn_weights_example
        return x

    def map_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        对外接口：接收 [seq_len, batch, dim]，返回同维度映射后的特征
        """
        return self.forward(x)
