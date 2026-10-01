"""Ejemplo mínimo para ejecutar el modelo MoE sobre CIFAR-10.

El cuaderno es la demostración principal; este script solo sirve como entrada rápida
para ejecutar la misma configuración sin repetir toda la lógica de visualización.
"""

import torch

from moe import VisionMoE, count_params
from train_utils import get_loaders, fit

EPOCHS = 30


def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Dispositivo:', device)

    train_dl, test_dl = get_loaders(batch_size=128, num_workers=0)

    net = VisionMoE(
        dim=128,
        depth=6,
        n_heads=4,
        use_moe=True,
        num_experts=8,
        k=2,
        expert_hidden=256,
    )
    total, active = count_params(net)
    print(f'Parámetros totales: {total:,} | activos por token: {active:,}')

    hist = fit(net, train_dl, test_dl, device, epochs=EPOCHS, lr=1e-3, name='MoE')
    print(
        f"Finished Training | accuracy final en test: {hist['val_acc'][-1]:.2%} "
        f"(mejor: {max(hist['val_acc']):.2%})"
    )


if __name__ == '__main__':
    main()
