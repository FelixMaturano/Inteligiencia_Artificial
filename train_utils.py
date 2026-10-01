"""train_utils.py — datos, entrenamiento y evaluación compartidos.

Este módulo centraliza la lógica de carga y entrenamiento para que el notebook y
el script no repitan la misma pipeline. Mantener la demostración en el cuaderno y
la lógica en un único sitio evita código duplicado.
"""
import time
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

CLASSES = ('Avión', 'Auto', 'Pájaro', 'Gato', 'Ciervo', 'Perro', 'Rana', 'Caballo', 'Barco', 'Camión')
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def get_loaders(batch_size=128, root='./data', num_workers=0):
    """Train con data augmentation (recorte + espejo); test solo normalizado."""
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(MEAN, STD)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    train_ds = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dl, test_dl


@torch.no_grad()
def evaluate(model, loader, device):
    """Devuelve (loss, accuracy, predicciones, etiquetas, confianza)."""
    model.eval()
    loss_sum, preds, labels, confs = 0.0, [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction='sum').item()
        conf, pred = logits.softmax(dim=1).max(dim=1)
        preds.append(pred.cpu()); labels.append(y.cpu()); confs.append(conf.cpu())
    preds, labels, confs = torch.cat(preds), torch.cat(labels), torch.cat(confs)
    return loss_sum / len(labels), (preds == labels).float().mean().item(), preds, labels, confs


def fit(model, train_dl, test_dl, device, epochs=30, lr=1e-3, weight_decay=0.05,
        aux_coef=0.01, label_smoothing=0.1, name='modelo'):
    """AdamW + OneCycle (warmup + decaimiento) + gradient clipping. Devuelve el historial por época."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(train_dl), pct_start=0.15)
    hist = {k: [] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'aux', 'lr', 'time')}

    for epoch in range(1, epochs + 1):
        model.train()
        t0, n, loss_sum, correct, aux_sum = time.time(), 0, 0.0, 0, 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits, aux = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing) + aux_coef * aux

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            n += y.size(0)
            loss_sum += F.cross_entropy(logits.detach(), y, reduction='sum').item()  # CE "limpia" para graficar
            correct += (logits.argmax(1) == y).sum().item()
            aux_sum += float(aux) * y.size(0)

        val_loss, val_acc, *_ = evaluate(model, test_dl, device)
        for key, val in zip(hist, (loss_sum / n, correct / n, val_loss, val_acc, aux_sum / n,
                                   sched.get_last_lr()[0], time.time() - t0)):
            hist[key].append(val)
        print(f"[{name}] época {epoch:02d}/{epochs} | train loss {hist['train_loss'][-1]:.3f} "
              f"acc {hist['train_acc'][-1]:.1%} | val loss {val_loss:.3f} acc {val_acc:.1%} "
              f"| aux {hist['aux'][-1]:.3f} | {hist['time'][-1]:.0f}s")
    return hist