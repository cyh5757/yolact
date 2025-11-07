# Copyright (c) 2025
# Ultra-lightweight W&B logger for YOLACT (scalars only by default)
# - No extra forwards, no image/mask/table uploads unless explicitly enabled
# - Safe to import when W&B is unavailable

from __future__ import annotations
import time
from typing import Dict, Optional

import torch

try:
    import wandb  # lazy-optional
    _HAS_WANDB = True
except Exception:
    wandb = None
    _HAS_WANDB = False

from data import cfg


class WandbLogger:
    """
    Defaults:
      - Scalars only (fast)
      - NO overlays, NO tables, NO extra forward calls
    You can opt-in images/tables later via enable_images/enable_tables flags, but
    the methods are no-ops unless explicitly enabled.
    """

    def __init__(
        self,
        project: str,
        name: str,
        args=None,
        enabled: bool = True,
        debug: bool = False,
        log_tables: bool = False,      # ignored unless enable_tables=True
        default_max_images: int = 0,   # 0 = don't log images
        enable_images: bool = False,   # hard switch OFF by default
        enable_tables: bool = False,   # hard switch OFF by default
    ):
        self.enabled = bool(enabled) and _HAS_WANDB
        self.debug = bool(debug)
        self.enable_images = bool(enable_images)
        self.enable_tables = bool(enable_tables)
        self.default_max_images = int(default_max_images)
        self.run = None

        if not self.enabled:
            return

        cfg_name = getattr(cfg, "name", "yolact")
        try:
            self.run = wandb.init(
                project=project,
                name=name or f"{cfg_name}_{int(time.time())}",
                config={
                    "config": cfg_name,
                    "batch_size": getattr(args, "batch_size", None) if args is not None else None,
                    "lr": getattr(args, "lr", None) if args is not None else None,
                    "momentum": getattr(args, "momentum", None) if args is not None else None,
                    "weight_decay": getattr(args, "decay", None) if args is not None else None,
                    "gamma": getattr(args, "gamma", None) if args is not None else None,
                    "max_iter": getattr(cfg, "max_iter", None),
                    "img_size": getattr(cfg, "max_size", None),
                    "train_images": getattr(cfg.dataset, "train_images", None),
                    "valid_images": getattr(cfg.dataset, "valid_images", None),
                    "num_workers": getattr(args, "num_workers", None) if args is not None else None,
                }
            )
        except Exception as e:
            # If init fails, just disable gracefully.
            if self.debug:
                print(f"[W&B] init failed: {e}")
            self.enabled = False
            self.run = None

    # ---- Scalars only (fast path)
    def watch(self, model, log_gradients_every=0):
        # Disabled by default to avoid overhead
        if not (self.enabled and self.run and log_gradients_every and log_gradients_every > 0):
            return
        try:
            wandb.watch(model, log="gradients", log_freq=int(log_gradients_every))
        except Exception as e:
            if self.debug:
                print(f"[W&B] watch() failed: {e}")

    def log_scalars(self, step: int, losses: dict, total_loss: torch.Tensor,
                    lr: float, epoch: int, elapsed_sec: float):
        if not (self.enabled and self.run):
            return
        try:
            scalars = {f"loss/{k}": (v.item() if torch.is_tensor(v) else float(v))
                       for k, v in losses.items()}
            scalars["loss/T"] = (total_loss.item() if torch.is_tensor(total_loss) else float(total_loss))
            scalars["lr"] = float(lr)
            scalars["epoch"] = int(epoch)
            scalars["time/iter_sec"] = float(elapsed_sec)
            wandb.log(scalars, step=int(step))
        except Exception as e:
            if self.debug:
                print(f"[W&B] log_scalars() failed: {e}")

    # ---- Validation metrics (still lightweight)
    def log_val_map(self, epoch: int, iteration: int, val_info: dict, elapsed_sec: float):
        if not (self.enabled and self.run and isinstance(val_info, dict)):
            return
        try:
            log_dict = {
                "val/elapsed_sec": float(elapsed_sec),
                "epoch": int(epoch),
                "iter": int(iteration),
            }
            for head, sub in val_info.items():
                if isinstance(sub, dict):
                    for k, v in sub.items():
                        try:
                            log_dict[f"val/{head}/{k}"] = float(v)
                        except Exception:
                            pass
                else:
                    try:
                        log_dict[f"val/{head}"] = float(sub)
                    except Exception:
                        pass
            wandb.log(log_dict)
        except Exception as e:
            if self.debug:
                print(f"[W&B] log_val_map() failed: {e}")

    # ---- Images/tables are NO-OP unless explicitly enabled ----
    def log_batch_overlays(self, *args, **kwargs):
        # Intentionally disabled by default to avoid extra forward/postprocess
        if not (self.enabled and self.run and self.enable_images and self.default_max_images > 0):
            return
        # If you ever re-enable: implement a *non-forwarding* path only.

    def log_predictions_table(self, *args, **kwargs):
        if not (self.enabled and self.run and self.enable_tables):
            return

    def log_scalar_dict(self, step: int, d: Dict[str, float]):
        if not (self.enabled and self.run):
            return
        try:
            wandb.log(d, step=int(step))
        except Exception as e:
            if self.debug:
                print(f"[W&B] log_scalar_dict() failed: {e}")

    def finish(self):
        if self.enabled and self.run is not None:
            try:
                wandb.finish()
            except Exception as e:
                if self.debug:
                    print(f"[W&B] finish() failed: {e}")
            finally:
                self.run = None
