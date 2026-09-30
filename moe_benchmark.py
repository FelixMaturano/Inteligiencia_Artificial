import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELayer(nn.Module):
    def __init__(self,d_model: int,d_hidden: int,num_experts: int,top_k: int = 2,):
        super().__init__()
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) no puede ser mayor que num_experts ({num_experts})"
            )
        self.num_experts = num_experts
        self.top_k = top_k
        # Grupo de expertos
        self.experts = nn.ModuleList(
            [Expert(d_model, d_hidden) for _ in range(num_experts)]
        )
        # Router (Capa lineal de proyección)
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor):
        # 0. Aplanado de entrada: [batch, seq_len, d_model] -> [n_tokens, d_model]
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        n_tokens = x_flat.shape[0]

        # 1. ROUTER & TOP-K SELECTION
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_probs, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)

        # Renormalización probabilística de los top-k seleccionados
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # 2. PROCESAMIENTO DISPERSO CONDICIONAL
        output_flat = torch.zeros_like(x_flat)

        for expert_id, expert in enumerate(self.experts):
            # Obtener índices de los tokens asignados a este experto
            token_indices, k_indices = torch.where(topk_idx == expert_id)

            if token_indices.numel() == 0:
                continue

            # Extraer subconjunto de tokens para la GPU
            expert_input = x_flat[token_indices]

            # Evaluar el experto solo en su subconjunto
            expert_output = expert(expert_input)

            # Obtener y aplicar los pesos del router
            expert_weights = topk_probs[token_indices, k_indices].unsqueeze(-1)
            expert_output = expert_output * expert_weights

            # Acumular la salida ponderada en las posiciones de memoria originales
            output_flat.index_add_(0, token_indices, expert_output)

        # Restaurar forma original del tensor
        output = output_flat.reshape(orig_shape)

        # 3. LOAD BALANCING LOSS (SWITCH TRANSFORMER)
        top1_idx = topk_idx[:, 0]
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)
        tokens_per_expert.scatter_add_(
            0, top1_idx, torch.ones(n_tokens, device=x.device, dtype=x.dtype)
        )

        fraction_tokens = tokens_per_expert / n_tokens
        router_prob_mean = router_probs.mean(dim=0)

        aux_loss = self.num_experts * torch.sum(fraction_tokens * router_prob_mean)

        return output, aux_loss


def ejecutar_prueba():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ejecutando en dispositivo: {device}")

    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Parámetros del benchmark
    d_model, d_hidden = 512, 2048
    num_experts, top_k = 8, 2
    batch_size, seq_len = 16, 128

    moe = MoELayer(d_model, d_hidden, num_experts, top_k).to(device)
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Warmup
    with torch.inference_mode():
        for _ in range(10):
            _ = moe(x)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Medición
    with torch.inference_mode():
        if device.type == "cuda":
            start_evt, end_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_evt.record()
            for _ in range(100):
                output, aux_loss = moe(x)
            end_evt.record()
            torch.cuda.synchronize()

            tiempo_ms = start_evt.elapsed_time(end_evt) / 100
            memoria_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            t0 = time.time()
            for _ in range(100):
                output, aux_loss = moe(x)
            tiempo_ms = ((time.time() - t0) / 100) * 1000
            memoria_mb = 0

    print("\n --- RESULTADOS DEL BENCHMARK ---")
    print(f" Entrada: {list(x.shape)} ({batch_size * seq_len} tokens)")
    print(f" Salida:  {list(output.shape)}")
    print(f" Loss Auxiliar: {aux_loss.item():.4f}")
    print(f" Tiempo promedio: {tiempo_ms:.3f} ms")
    if device.type == "cuda":
        print(f"VRAM máxima asignada: {memoria_mb:.2f} MB")

if __name__ == "__main__":
    ejecutar_prueba()