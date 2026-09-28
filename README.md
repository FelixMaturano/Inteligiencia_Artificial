# 🧠 Mixture of Experts (MoE) - Implementación en PyTorch y CUDA Benchmark

Este repositorio contiene una implementación educativa de una capa **Sparse Mixture of Experts (MoE)** desarrollada desde cero en **PyTorch**. El modelo utiliza procesamiento disperso real (*True Sparse Routing*), selección Top-$k$ condicional, pérdida auxiliar de balanceo de carga (Switch Transformer) y un módulo de benchmark en hardware GPU CUDA.

---

##  Referencias y Fundamento Teórico

Esta implementación está basada e inspirada en el trabajo de investigación y divugación de:

* 📖 **Artículo de Referencia:** [A Visual Guide to Mixture of Experts (MoE)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts) por **Maarten Grootendorst**.
* 📄 **Paper Científico (Switch Transformers):** Fedus et al., *"Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"*.

---

## Características Clave de la Implementación

* **Procesamiento Disperso Real (True Sparsity):** A diferencia de las aproximaciones conceptuales con máscaras y `einsum`, este código extrae únicamente los tensores asignados a cada experto usando `torch.where` y los acumula de forma eficiente con `index_add_`.
* **Router Top-$k$ Dinámico:** El enrutador asigna los tokens de entrada a los $k$ mejores expertos ($k=2$ por defecto) y renormaliza dinámicamente sus pesos.
* **Redes de Expertos Modernas:** Cada experto es una MLP (Feed-Forward Network) con función de activación **GELU**, estándar en arquitecturas como GPT-4 y Mixtral 8x7B.
* **Loss Auxiliar de Balanceo de Carga:** Previene el *Router Collapse* (colapso del enrutador) penalizando matemáticamente el desbalance en la asignación de tokens entre los expertos.
* **Benchmarking CUDA Asíncrono:** Medición precisa del tiempo de inferencia en milisegundos mediante `torch.cuda.Event` y registro del consumo pico de memoria VRAM.

---

## Arquitectura del Sistema

```text
[ Entrada: (batch, seq_len, d_model) ]
                  │
                  ▼
         [ Flatten Tokens (N_tokens, d_model) ]
                  │
        ┌─────────┴─────────┐
        │   Router Linear   │ ──► Proyección a logits por experto
        └─────────┬─────────┘
                  │
                  ▼
         [ Top-k Softmax & Renormalización ]
                  │
   ┌──────────────┼──────────────┐  (Se seleccionan solo K=2 expertos por token)
   ▼              ▼              ▼
[Experto 0]   [Experto 1] ... [Experto N-1]  ──► Procesan únicamente sus tokens asignados
   │              │              │
   └──────────────┼──────────────┘
                  │
                  ▼
   [ Acumulación index_add_ + Reshape ] ──► [ Salida: (batch, seq_len, d_model) ]