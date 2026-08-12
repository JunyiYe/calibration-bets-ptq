from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 128
    epochs: int = 20
    lr: float = 5e-4
    weight_decay: float = 1e-4
    patience: int = 5
    log_every: int = 1


@dataclass(frozen=True)
class TrainResult:
    best_val_loss: float
    epochs_ran: int


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> TrainResult:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val_loss = float("inf")
    stale_epochs = 0
    epochs_ran = 0
    total_start = time.perf_counter()

    print(
        "[TRAIN] "
        f"epochs={config.epochs} batch_size={config.batch_size} "
        f"lr={config.lr} weight_decay={config.weight_decay} "
        f"patience={config.patience} device={device}",
        flush=True,
    )

    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        epochs_ran = epoch + 1
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            batch = x.shape[0]
            train_loss_sum += float(loss.item()) * batch
            train_count += batch

        train_loss = train_loss_sum / max(1, train_count)
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if should_log_epoch(epochs_ran, config.log_every, improved, stale_epochs, config.patience):
            elapsed = time.perf_counter() - epoch_start
            print(
                "[TRAIN] "
                f"epoch={epochs_ran}/{config.epochs} "
                f"train_loss={train_loss:.8g} val_loss={val_loss:.8g} "
                f"best_val_loss={best_val_loss:.8g} "
                f"improved={int(improved)} stale={stale_epochs}/{config.patience} "
                f"elapsed_sec={elapsed:.1f}",
                flush=True,
            )

        if stale_epochs >= config.patience:
            print(f"[TRAIN] early_stop epoch={epochs_ran} best_val_loss={best_val_loss:.8g}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    total_elapsed = time.perf_counter() - total_start
    print(
        f"[TRAIN] done epochs_ran={epochs_ran} best_val_loss={best_val_loss:.8g} "
        f"total_elapsed_sec={total_elapsed:.1f}",
        flush=True,
    )
    return TrainResult(best_val_loss=best_val_loss, epochs_ran=epochs_ran)


def should_log_epoch(
    epoch: int,
    log_every: int,
    improved: bool,
    stale_epochs: int,
    patience: int,
) -> bool:
    if log_every <= 0:
        return improved or stale_epochs >= patience
    return epoch == 1 or epoch % log_every == 0 or improved or stale_epochs >= patience


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        batch = x.shape[0]
        total_loss += float(loss.item()) * batch
        total_count += batch
    return total_loss / max(1, total_count)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        pred = model(x).detach().cpu().numpy()
        preds.append(pred)
        targets.append(y.numpy())
    return np.concatenate(targets, axis=0), np.concatenate(preds, axis=0)
