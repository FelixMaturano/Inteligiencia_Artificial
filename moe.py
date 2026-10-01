import torch
import torch.nn as nn
import torch.nn.functional as F
class Expert(nn.Module):

    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


class NoisyTopKRouter(nn.Module):

    def __init__(self, dim, num_experts, k=2, noisy=True):
        super().__init__()
        assert k <= num_experts
        self.k, self.num_experts, self.noisy = k, num_experts, noisy
        self.w_gate = nn.Linear(dim, num_experts, bias=False)
        self.w_noise = nn.Linear(dim, num_experts)
        # Inicio pequeño -> al comienzo el reparto es casi uniforme (no hay expertos "ganadores")
        nn.init.normal_(self.w_gate.weight, std=0.02)
        nn.init.zeros_(self.w_noise.weight)
        nn.init.constant_(self.w_noise.bias, -2.0)  # softplus(-2) ~ 0.13 de ruido inicial

    def forward(self, x):
        clean_logits = self.w_gate(x)                                  # [N, E]
        logits = clean_logits
        if self.noisy and self.training:
            std = F.softplus(self.w_noise(x)) + 1e-2
            logits = clean_logits + torch.randn_like(clean_logits) * std

        top_logits, top_idx = logits.topk(self.k, dim=-1)              # [N, k]
        top_gates = F.softmax(top_logits, dim=-1)                      # [N, k]

        # --- Pérdida auxiliar de balanceo (estilo Switch Transformer) ---
        # f_i = fracción de asignaciones que recibe el experto i   (no diferenciable)
        # P_i = probabilidad media que el router le da al experto i (diferenciable)
        # aux = E * sum(f_i * P_i)  ->  vale 1.0 cuando el reparto es perfectamente uniforme
        probs = F.softmax(clean_logits, dim=-1)
        P = probs.mean(dim=0)
        f = F.one_hot(top_idx, self.num_experts).sum(dim=1).float().mean(dim=0) / self.k
        aux_loss = self.num_experts * (f * P).sum()
        return top_idx, top_gates, aux_loss


class MoE(nn.Module):

    def __init__(self, dim, hidden, num_experts=8, k=2, noisy=True):
        super().__init__()
        self.router = NoisyTopKRouter(dim, num_experts, k, noisy)
        self.experts = nn.ModuleList([Expert(dim, hidden) for _ in range(num_experts)])
        self.last_top_idx = None  # se guarda para analizar el enrutamiento después

    def forward(self, x):
        B, L, D = x.shape
        tokens = x.reshape(-1, D)                                      # [B*L, D]
        top_idx, top_gates, aux_loss = self.router(tokens)
        self.last_top_idx = top_idx.detach()

        out = torch.zeros_like(tokens)
        for e, expert in enumerate(self.experts):
            token_idx, slot = (top_idx == e).nonzero(as_tuple=True)   # tokens que eligieron al experto e
            if token_idx.numel() == 0:
                continue
            y = expert(tokens[token_idx]) * top_gates[token_idx, slot].unsqueeze(-1)
            out.index_add_(0, token_idx, y)
        return out.reshape(B, L, D), aux_loss


class DenseFFN(nn.Module):

    def __init__(self, dim, hidden):
        super().__init__()
        self.net = Expert(dim, hidden)

    def forward(self, x):
        return self.net(x), x.new_zeros(())


class TransformerBlock(nn.Module):

    def __init__(self, dim, n_heads, ffn, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = ffn
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attn_out)
        ffn_out, aux = self.ffn(self.norm2(x))
        x = x + self.drop(ffn_out)
        return x, aux


class VisionMoE(nn.Module):

    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10,
                 dim=128, depth=6, n_heads=4, use_moe=True,
                 num_experts=8, k=2, expert_hidden=256, dropout=0.1, noisy=True):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        # Conv con kernel=stride=patch  ==  cortar en patches + capa lineal sobre cada patch
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.drop = nn.Dropout(dropout)

        def make_ffn():
            if use_moe:
                return MoE(dim, expert_hidden, num_experts, k, noisy)
            return DenseFFN(dim, expert_hidden * k)

        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads, make_ffn(), dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)            # [B, 64, dim]
        x = self.drop(x + self.pos_embed)
        aux_total = 0.0
        for block in self.blocks:
            x, aux = block(x)
            aux_total = aux_total + aux
        logits = self.head(self.norm(x).mean(dim=1))
        return logits, aux_total / len(self.blocks)


def count_params(model):
    """(parámetros totales, parámetros activos por token). En MoE solo k de E expertos trabajan por token."""
    total = sum(p.numel() for p in model.parameters())
    inactive = 0
    for m in model.modules():
        if isinstance(m, MoE):
            per_expert = sum(p.numel() for p in m.experts[0].parameters())
            inactive += per_expert * (m.router.num_experts - m.router.k)
    return total, total - inactive