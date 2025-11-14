#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------
# train.py (YOLACT – EarlyStopping + Best/Last-Checkpoint + Robust mAP pick + Pro scalars)
# -----------------------------

from torch.cpu import is_available
from data import *
from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage, SavePath
from utils.logger import Log
from utils import timer
from layers.modules import MultiBoxLoss
from yolact import Yolact

import os, sys, time, math, random, argparse, datetime, json, shutil, subprocess, platform, csv
from pathlib import Path
from contextlib import nullcontext
from datetime import datetime, timedelta

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import numpy as np

# 평가 스크립트
import eval as eval_script
# W&B
from utils.wandb_logger import WandbLogger

# optional yaml
try:
    import yaml
except Exception:
    yaml = None

# optional psutil for CPU mem
try:
    import psutil
except Exception:
    psutil = None


def str2bool(v):
    if isinstance(v, bool): return v
    return v.lower() in ("yes", "true", "t", "1")


# -------- argparse --------
parser = argparse.ArgumentParser(description='Yolact Training Script')

# 기본 옵션
parser.add_argument('--backbone_path', default=None, type=str,
                    help='Override cfg.backbone.path (absolute or relative).')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint to resume. "interrupt" or "latest" supported.')
parser.add_argument('--start_iter', default=-1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--cuda', default=True, type=str2bool)

parser.add_argument('--lr', '--learning_rate', default=None, type=float)
parser.add_argument('--momentum', default=None, type=float)
parser.add_argument('--decay', '--weight_decay', default=None, type=float)
parser.add_argument('--gamma', default=None, type=float)

parser.add_argument('--save_folder', default='weights/')
parser.add_argument('--log_folder', default='logs/')
parser.add_argument('--config', default=None)
parser.add_argument('--save_interval', default=10000, type=int)

parser.add_argument('--validation_size', default=10, type=int)
parser.add_argument('--validation_epoch', default=2, type=int,
                    help='If -1, no validation.')

parser.add_argument('--keep_latest', dest='keep_latest', action='store_true')
parser.add_argument('--keep_latest_interval', default=100000, type=int)

parser.add_argument('--dataset', default=None, type=str)

parser.add_argument('--no_log', dest='log', action='store_false')
parser.add_argument('--log_gpu', dest='log_gpu', action='store_true')
parser.add_argument('--no_interrupt', dest='interrupt', action='store_false')

parser.add_argument('--batch_alloc', default=None, type=str)
parser.add_argument('--no_autoscale', dest='autoscale', action='store_false')

# 추가: BN freeze CLI 오버라이드
parser.add_argument('--freeze_bn', default=None, type=str2bool,
                    help='Override cfg.freeze_bn (True/False).')

# ---- Optim/Scheduler ----
parser.add_argument('--optim', default=None, choices=['sgd', 'adamw', 'adamwr'],
                    help='Optimizer: sgd | adamw | adamwr (adamwr = AdamW + cosine_restart)')
parser.add_argument('--scheduler', default=None,
                    choices=['step', 'cosine', 'cosine_restart', 'none'],
                    help='LR scheduler: step | cosine | cosine_restart | none (default: from cfg)')

# CosineAnnealingWarmRestarts 파라미터
parser.add_argument('--cosine_t0', type=int, default=None,
                    help='CosineAnnealingWarmRestarts T_0 (post-warmup iterations).')
parser.add_argument('--cosine_tmult', type=int, default=None,
                    help='CosineAnnealingWarmRestarts T_mult.')

# ---- 실험 디렉토리 ----
parser.add_argument('--exp_dir', type=str, default=None,
                    help='실험 저장 루트. 지정하면 표준 트리 생성')

# ---- Best 저장 옵션 ----
parser.add_argument('--save_best', default=True, type=str2bool,
                    help='Validation 성능이 향상되면 best 체크포인트 저장')
parser.add_argument('--best_alias', default='best.pth', type=str,
                    help='가장 최근 best 모델의 별칭 파일명')
parser.add_argument('--prefer_metric', default='mask_only', type=str,
                    choices=['mask_first','box_first','mask_only','box_only'],
                    help='best 판단 시 어떤 지표를 우선할지')

# ---- Last 저장 옵션 ----
parser.add_argument('--save_last', default=True, type=str2bool,
                    help='학습 종료 시점(early stop/정상 종료/interrupt) 가중치 저장')
parser.add_argument('--last_alias', default='last.pth', type=str,
                    help='마지막 가중치 별칭 파일명')

# ---- Early Stopping ----
parser.add_argument('--early_stop', default=True, type=str2bool,
                    help='Enable early stopping based on validation performance.')
parser.add_argument('--patience', default=10, type=int,
                    help='Number of validation epochs without improvement before stopping.')
parser.add_argument('--min_delta', default=0.001, type=float,
                    help='Minimum change in validation metric to qualify as improvement.')
parser.add_argument('--max_nan_streak', default=5, type=int,
                    help='Maximum consecutive NaN losses before stopping.')

# ---- Best Checkpoint 정리 ----
parser.add_argument('--keep_best_n', default=3, type=int,
                    help='Keep only top N best checkpoints (0=keep all).')

# ---- W&B options ----
parser.add_argument('--use_wandb', default=True, type=str2bool,
                    help='Enable Weights & Biases logging.')
parser.add_argument('--wandb_project', default='YOLACT-Cell', type=str)
parser.add_argument('--wandb_name', default=None, type=str,
                    help='Run name; default uses cfg.name + timestamp.')
parser.add_argument('--wandb_images_every', default=200, type=int,
                    help='Log overlay images every N iterations (lightweight).')
parser.add_argument('--wandb_score_thr', default=0.05, type=float,
                    help='Score threshold for overlay predictions.')
parser.add_argument('--wandb_tables', default=False, type=str2bool,
                    help='Debug-heavy tables (disabled by default).')
parser.add_argument('--wandb_debug', default=False, type=str2bool,
                    help='Verbose prints inside W&B logger.')

parser.set_defaults(keep_latest=False, log=True, log_gpu=False,
                    interrupt=True, autoscale=True)
args = parser.parse_args()


# -------- cfg/dataset load + autoscale --------
if args.config is not None:
    set_cfg(args.config)
if args.dataset is not None:
    set_dataset(args.dataset)

if args.autoscale and args.batch_size != 8:
    factor = args.batch_size / 8.0
    print('Scaling parameters by %.2f for batch size %d.' % (factor, args.batch_size))
    cfg.lr *= factor
    cfg.max_iter = max(1, int(round(cfg.max_iter / factor)))
    cfg.lr_steps = [max(1, int(round(s / factor))) for s in cfg.lr_steps]

def replace_from_cfg(name):
    if getattr(args, name) is None:
        setattr(args, name, getattr(cfg, name))
for k in ['lr', 'decay', 'gamma', 'momentum']:
    replace_from_cfg(k)

def replace_from_cfg2(name, default=None):
    if getattr(args, name) is None:
        setattr(args, name, getattr(cfg, name, default))
replace_from_cfg2('optim', 'adamw')
replace_from_cfg2('scheduler', 'cosine')

if args.freeze_bn is not None:
    cfg.freeze_bn = bool(args.freeze_bn)

use_cuda = args.cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
if use_cuda:
    cudnn.benchmark = True

cur_lr = args.lr
loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']


# -------- Early Stopping Class --------
class EarlyStopping:
    """Early Stopping 헬퍼 클래스"""
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.early_stop = False
        
    def __call__(self, score):
        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)
        if improved:
            self.best_score = score
            self.counter = 0
            return False  # 개선됨
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True  # Early stop 발동
            return False


# -------- Best Checkpoint 관리 --------
class BestCheckpointManager:
    """Best 체크포인트 파일 관리 (상위 N개만 유지)"""
    def __init__(self, root_dir, keep_n=3):
        self.root_dir = root_dir
        self.keep_n = keep_n
        self.checkpoints = []  # (score, filepath)
        
    def add(self, score, filepath):
        self.checkpoints.append((score, filepath))
        self.checkpoints.sort(key=lambda x: x[0], reverse=True)
        if self.keep_n > 0 and len(self.checkpoints) > self.keep_n:
            for _, old_path in self.checkpoints[self.keep_n:]:
                if os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                        print(f"[BestMgr] Removed old checkpoint: {old_path}")
                    except Exception as e:
                        print(f"[BestMgr] Failed to remove {old_path}: {e}")
            self.checkpoints = self.checkpoints[:self.keep_n]
    
    def get_best(self):
        return self.checkpoints[0] if self.checkpoints else (float('-inf'), None)


# -------- helpers: serialize cfg/args/env/git --------
def _safe_val(v):
    try:
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
        if isinstance(v, (list, tuple)):
            return [_safe_val(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _safe_val(val) for k, val in v.items()}
        if torch.is_tensor(v):
            return f"Tensor(shape={tuple(v.shape)}, dtype={str(v.dtype)})"
        if 'numpy' in type(v).__module__:
            return getattr(v, 'item', lambda: str(v))()
        return str(v)
    except Exception:
        return str(v)

def _cfg_to_dict(cfg_obj):
    d = {}
    for k, v in vars(cfg_obj).items():
        d[k] = _safe_val(v)
    return d

def _run_quiet(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

def _get_git_info():
    return {
        "git_commit": _run_quiet(["git", "rev-parse", "HEAD"]),
        "git_branch": _run_quiet(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_status_dirty": bool(_run_quiet(["git", "status", "--porcelain"]))
    }

def _get_env_info():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_compiled": str(torch.version.cuda) if torch.version.cuda else None,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    }

def _dump_effective_run(exp_dir, args_obj, cfg_obj, extra=None, fname_prefix="effective"):
    os.makedirs(exp_dir, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "args": {k: _safe_val(v) for k, v in dict(args_obj._get_kwargs()).items()},
        "cfg": _cfg_to_dict(cfg_obj),
        "env": _get_env_info(),
        "git": _get_git_info(),
    }
    if extra:
        payload["extra"] = extra

    json_path = os.path.join(exp_dir, f"{fname_prefix}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    yml_path = os.path.join(exp_dir, f"{fname_prefix}.yaml")
    try:
        if yaml is not None:
            with open(yml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    except Exception as e:
        print(f"[warn] yaml dump failed: {e}")

    return json_path, yml_path


# -------- experiment dirs, tee, initial dumps --------
_paths = None

def _dump_cfg_snapshot(dst_dir:str):
    dump = {k: (v if isinstance(v, (int,float,str,bool,list,dict,type(None)))
                else str(v)) for k,v in vars(cfg).items()}
    ypath = os.path.join(dst_dir, 'config.yaml')
    jpath = os.path.join(dst_dir, 'config.json')
    try:
        if yaml is not None:
            with open(ypath, 'w', encoding='utf-8') as f:
                yaml.safe_dump(dump, f, sort_keys=False, allow_unicode=True)
        else:
            with open(jpath, 'w', encoding='utf-8') as f:
                json.dump(dump, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[warn] config dump failed: {e}")

class TeeStdout:
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, 'a', buffering=1, encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()

def _prepare_experiment_dirs(args, cfg):
    if args.exp_dir is None:
        today = datetime.now().strftime('%Y-%m-%d')
        args.exp_dir = os.path.join('experiments', f'{today}_run')

    exp_dir = args.exp_dir
    eval_dir = os.path.join(exp_dir, 'eval')
    weights_dir = os.path.join(exp_dir, 'weights')
    curves_dir = os.path.join(exp_dir, 'curves')
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(curves_dir, exist_ok=True)

    notes = os.path.join(exp_dir, 'notes.md')
    if not os.path.exists(notes):
        with open(notes, 'w', encoding='utf-8') as f:
            f.write(f"# {cfg.name}\n\n- 생성시각: {datetime.now()}\n")

    # stdout tee → train.log
    TeeStdout(os.path.join(exp_dir, 'train.log'))

    # config snapshot (human-friendly)
    _dump_cfg_snapshot(exp_dir)

    # 재현용 CLI 저장
    try:
        with open(os.path.join(exp_dir, "run_cli.txt"), "w", encoding="utf-8") as f:
            f.write("python " + " ".join(map(str, sys.argv)) + "\n")
    except Exception as e:
        print(f"[warn] failed to write run_cli.txt: {e}")

    return {'exp_dir': exp_dir, 'eval_dir': eval_dir, 'weights_dir': weights_dir, 'curves_dir': curves_dir}

def _epoch_eval_paths(epoch_idx:int):
    export_json = os.path.join(_paths['eval_dir'], f'val_epoch_{epoch_idx:03d}.json')
    metrics_csv = os.path.join(_paths['eval_dir'], 'metrics.csv')
    return export_json, metrics_csv


# -------- CSV helpers / mem & grad stats (NEW) --------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _append_row_csv(csv_path:str, row:dict):
    _ensure_dir(os.path.dirname(csv_path))
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def _mem_stats():
    out = {}
    if torch.cuda.is_available():
        out["vram_alloc_GB"]    = float(torch.cuda.memory_allocated()  / 1e9)
        out["vram_reserved_GB"] = float(torch.cuda.memory_reserved()   / 1e9)
        try:
            out["vram_peak_GB"] = float(torch.cuda.max_memory_allocated() / 1e9)
        except Exception:
            out["vram_peak_GB"] = None
    else:
        out["vram_alloc_GB"] = out["vram_reserved_GB"] = out["vram_peak_GB"] = None

    if psutil is not None:
        vm = psutil.virtual_memory()
        out["cpu_used_GB"] = float((vm.total - vm.available)/ (1024**3))
        out["cpu_total_GB"] = float(vm.total / (1024**3))
    else:
        out["cpu_used_GB"] = out["cpu_total_GB"] = None
    return out

def _global_grad_norm(model: nn.Module):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().float()
            total += float(torch.sum(g*g).item())
    return math.sqrt(total) if total > 0 else 0.0

def _lr_wd_groups(optimizer: optim.Optimizer):
    lrs = [pg.get("lr", None) for pg in optimizer.param_groups]
    wds = [pg.get("weight_decay", None) for pg in optimizer.param_groups]
    return lrs, wds

def iter_with_timer(loader):
    t_fetch = time.time()
    for batch in loader:
        data_time = time.time() - t_fetch
        yield batch, data_time
        t_fetch = time.time()

def _append_ckpt_registry(root_dir:str, tag:str, epoch:int, iteration:int, val_score, path:str, extra:dict=None):
    row = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "epoch": int(epoch),
        "iter": int(iteration),
        "val_score": (float(val_score) if (val_score is not None and math.isfinite(float(val_score))) else ""),
        "path": os.path.relpath(path, root_dir) if os.path.exists(path) else path
    }
    if extra:
        row.update({f"extra_{k}": v for k, v in extra.items()})
    csv_path = os.path.join(root_dir, "checkpoints.csv")
    _append_row_csv(csv_path, row)


# -------- NaN / Inf guards --------
def all_finite_dict(d: dict) -> bool:
    for v in d.values():
        if torch.is_tensor(v):
            if not torch.isfinite(v).all():
                return False
    return True

def log_nonfinite_losses(losses: dict, tag: str = ""):
    bad = []
    for k, v in losses.items():
        if torch.is_tensor(v) and not torch.isfinite(v).all():
            try:
                bad.append(f"{k}={v.detach().float().mean().item():.4g}")
            except Exception:
                bad.append(k)
    if bad:
        print(f"[NaN guard]{(' '+tag) if tag else ''} non-finite: " + ", ".join(bad))


# -------- Minimal ScatterWrapper for validation loss (compat) --------
class ScatterWrapper(object):
    def __init__(self, targets, masks, num_crowds):
        self.targets = targets
        self.masks = masks
        self.num_crowds = num_crowds
    def make_mask(self):
        return [t is not None for t in self.targets]


# -------- Model+Loss wrappers --------
class NetLoss(nn.Module):
    """net + criterion 묶음. AMP는 비활성."""
    def __init__(self, net:Yolact, criterion:MultiBoxLoss, device:torch.device):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.device = device

    def forward(self, images, targets, masks, num_crowds):
        amp_ctx = torch.autocast('cuda', enabled=False) if self.device.type == 'cuda' else nullcontext()
        with amp_ctx:
            preds = self.net(images)
            losses = self.criterion(self.net, preds, targets, masks, num_crowds)
        return losses

class CustomDataParallel(nn.DataParallel):
    """YOLACT 전용 DataParallel: scatter/gather 커스텀"""
    def scatter(self, inputs, kwargs, device_ids):
        devices = ['cuda:' + str(x) for x in device_ids]
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)
        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], [kwargs]*len(devices)

    def gather(self, outputs, output_device):
        out = {}
        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])
        return out

class SingleDeviceWrapper(nn.Module):
    """단일 장치 datum=(images,(targets,masks,num_crowds)) 처리"""
    def __init__(self, netloss_module: NetLoss, device: torch.device):
        super().__init__()
        self.netloss = netloss_module
        self.device = device

    def forward(self, datum):
        images, (targets, masks, num_crowds) = datum
        if isinstance(images, (list, tuple)):
            images = torch.stack([img.to(self.device, non_blocking=(self.device.type=='cuda')) for img in images], dim=0)
        else:
            images = images.to(self.device, non_blocking=(self.device.type=='cuda'))

        def move_to_device(x):
            if isinstance(x, (list, tuple)):
                return [t.to(self.device) if hasattr(t, 'to') else t for t in x]
            return x.to(self.device) if hasattr(x, 'to') else x

        targets    = move_to_device(targets)
        masks      = move_to_device(masks)
        num_crowds = move_to_device(num_crowds)
        return self.netloss(images, targets, masks, num_crowds)


# -------- misc helpers --------
def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    global cur_lr
    cur_lr = new_lr

def gradinator(x):
    x.requires_grad = False
    return x

def prepare_data(datum, devices:list=None, allocation:list=None):
    with torch.no_grad():
        if devices is None:
            devices = ['cuda:0'] if use_cuda else ['cpu']
        if allocation is None:
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)
            allocation.append(args.batch_size - sum(allocation))

        images, (targets, masks, num_crowds) = datum

        cur_idx = 0
        for device_name, alloc in zip(devices, allocation):
            for _ in range(alloc):
                images[cur_idx] = gradinator(images[cur_idx].to(device_name))
                targets[cur_idx] = gradinator(targets[cur_idx].to(device_name))
                masks[cur_idx]   = gradinator(masks[cur_idx].to(device_name))
                cur_idx += 1

        if cfg.preserve_aspect_ratio:
            _, h, w = images[random.randint(0, len(images)-1)].size()
            for idx, (image, target, mask, num_crowd) in enumerate(zip(images, targets, masks, num_crowds)):
                images[idx], targets[idx], masks[idx], num_crowds[idx] = enforce_size(
                    image, target, mask, num_crowd, w, h
                )

        cur_idx = 0
        split_images, split_targets, split_masks, split_numcrowds = [[None for _ in allocation] for _ in range(4)]
        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx]   = torch.stack(images[cur_idx:cur_idx+alloc], dim=0)
            split_targets[device_idx]  = targets[cur_idx:cur_idx+alloc]
            split_masks[device_idx]    = masks[cur_idx:cur_idx+alloc]
            split_numcrowds[device_idx]= num_crowds[cur_idx:cur_idx+alloc]
            cur_idx += alloc

        return split_images, split_targets, split_masks, split_numcrowds


# -------- validation helpers --------
def _dig_first_float(x):
    """dict/list/float 어디에서든 첫 부동소수 값을 안전하게 찾아 반환"""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        for k in ['mAP', 'map', 'AP', 'ap', 'all', 'overall', 'mean', 'avg']:
            if k in x:
                v = _dig_first_float(x[k])
                if v is not None: return v
        for v in x.values():
            r = _dig_first_float(v)
            if r is not None: return r
        return None
    if isinstance(x, (list, tuple)):
        for v in x:
            r = _dig_first_float(v)
            if r is not None: return r
    return None

def _safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def pick_val_score(val_info: dict, prefer: str = 'mask_only') -> float:
    """여러 반환 스키마를 견고하게 지원하는 mAP 선택기"""
    mask_map = (
        _dig_first_float(val_info.get('segm')) or
        _dig_first_float(val_info.get('mask')) or
        _dig_first_float(val_info.get('masks'))
    )
    box_map = (
        _dig_first_float(val_info.get('bbox')) or
        _dig_first_float(val_info.get('box')) or
        _dig_first_float(val_info.get('boxes'))
    )

    if prefer == 'mask_only':
        return float(mask_map) if mask_map is not None else float('-inf')
    if prefer == 'box_only':
        return float(box_map) if box_map is not None else float('-inf')
    if prefer == 'mask_first':
        return float(mask_map) if mask_map is not None else (float(box_map) if box_map is not None else float('-inf'))
    if prefer == 'box_first':
        return float(box_map) if box_map is not None else (float(mask_map) if mask_map is not None else float('-inf'))
    return float(mask_map) if mask_map is not None else (float(box_map) if box_map is not None else float('-inf'))


def compute_validation_loss(net, data_loader, criterion):
    """실제로 validation loss 계산"""
    with torch.no_grad():
        losses, iterations = {}, 0
        for datum in data_loader:
            images, targets, masks, num_crowds = prepare_data(datum)
            out = net(images)
            wrapper = ScatterWrapper(targets, masks, num_crowds)
            _losses = criterion(out, wrapper, wrapper.make_mask())
            for k, v in _losses.items():
                v = v.mean().item()
                losses[k] = losses.get(k, 0.0) + v
            iterations += 1
            if args.validation_size <= iterations * args.batch_size:
                break
        for k in losses:
            losses[k] /= max(1, iterations)
        loss_labels = sum([[k, losses[k]] for k in ['B','C','M','S','P','D','E','I'] if k in losses], [])
        print(('Validation Loss ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)
        return losses


def compute_validation_map(epoch, iteration, yolact_net, dataset, log:Log=None, wandb_logger:WandbLogger=None):
    export_json, metrics_csv = _epoch_eval_paths(epoch)
    eval_args = [
        '--no_bar',
        f'--max_images={args.validation_size}',
        '--top_k=100',
        '--score_threshold=0.05',
        '--fast_nms=True',
        '--cross_class_nms=False',
        '--output_coco_json',
        f'--export_coco={export_json}',
        f'--coco_images_dir={cfg.dataset.valid_images}',
        f'--coco_ann_file={cfg.dataset.valid_info}',
        f'--metrics_csv={metrics_csv}',
        '--compute_aji=False',  # Disable expensive AJI computation during training
        '--profile_infer=False',  # Disable profiler during training validation
        '--model_stats=False',  # Disable model stats during training validation
    ]
    eval_script.parse_args(eval_args)
    eval_script.prep_coco_cats()

    with torch.no_grad():
        yolact_net.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        start = time.time()
        print("\nComputing validation mAP (this may take a while)...", flush=True)
        val_info = eval_script.evaluate(yolact_net, dataset, train_mode=True)
        end = time.time()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)
        if wandb_logger is not None:
            wandb_logger.log_val_map(epoch=epoch, iteration=iteration, val_info=val_info, elapsed_sec=(end - start))
            # 납작한 스칼라로도 몇 개 추가 로깅
            try:
                flat = {}
                def _flatten(prefix, obj):
                    if isinstance(obj, dict):
                        for k,v in obj.items():
                            _flatten(f"{prefix}{k}/", v)
                    elif isinstance(obj, (int,float)) and math.isfinite(float(obj)):
                        flat[prefix[:-1]] = float(obj)
                _flatten("", val_info)
                keep_keys = [k for k in flat.keys() if any(s in k.lower() for s in ["ap50","ap75","map","aps","apm","apl"])]
                wandb_logger.log_scalar_dict(step=iteration, d={f"val_flat/{k}": flat[k] for k in keep_keys})
            except Exception:
                pass
        yolact_net.train()
        return val_info


def setup_eval():
    eval_script.parse_args([
        '--no_bar',
        f'--max_images={args.validation_size}',
        '--top_k=100',
        '--score_threshold=0.05',
        '--fast_nms=True',
        '--cross_class_nms=False'
    ])


# AdamW param groups (Norm/bias WD 제외)
def build_param_groups(model, wd: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndimension() == 1 or n.endswith('.bias') or 'bn' in n.lower() or 'gn' in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay, 'weight_decay': wd},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


# -------- backbone path resolver --------
def _resolve_backbone_path(args, cfg):
    """
    args.save_folder(= exp_dir/weights) 확정 이후 호출.
    args.backbone_path > cfg.backbone.path 순으로 후보 생성 후 존재하는 첫 경로를 반환.
    """
    raw = args.backbone_path if args.backbone_path else getattr(cfg.backbone, 'path', None)
    if not raw:
        raise FileNotFoundError("cfg.backbone.path 가 비어있고 --backbone_path 도 주어지지 않았습니다.")

    candidates = []

    def extend_candidates(name: str):
        if not name:
            return
        if os.path.isabs(name):
            candidates.append(name)
        else:
            candidates.extend([
                os.path.join(args.save_folder, name),                               # exp_dir/weights/name
                os.path.join(Path(__file__).resolve().parent, 'weights', name),     # 프로젝트/weights/name
                os.path.join(os.getcwd(), 'weights', name),                         # CWD/weights/name
                os.path.join(os.getcwd(), name),                                    # CWD/name
            ])

    if args.backbone_path:
        extend_candidates(args.backbone_path)
        if not os.path.isabs(args.backbone_path) and getattr(cfg.backbone, 'path', None):
            extend_candidates(cfg.backbone.path)
    else:
        extend_candidates(cfg.backbone.path)

    for p in candidates:
        if os.path.exists(p):
            return p

    msg = "Backbone checkpoint not found. Tried:\n" + "\n".join(["  - " + c for c in candidates])
    raise FileNotFoundError(msg)


def save_as_best(yolact_net, score: float, epoch: int, iteration: int, root_dir: str, alias_name: str):
    fname = f"best_iter{iteration:07d}_ep{epoch:03d}_score{score:.4f}.pth"
    full_path = os.path.join(root_dir, fname)
    print(f"[BEST] New best score={score:.4f} at iter={iteration}, epoch={epoch} → {full_path}")
    yolact_net.save_weights(full_path)

    # 별칭 파일 갱신
    alias_path = os.path.join(root_dir, alias_name)
    try:
        shutil.copy2(full_path, alias_path)
    except Exception as e:
        print(f"[BEST][warn] alias copy failed: {e}")

    # 시작 설정 사본 옆에 남겨두기 (있으면)
    try:
        stem = os.path.splitext(full_path)[0]
        start_json = os.path.join(os.path.dirname(root_dir), "cfg_start.json")  # exp_dir/cfg_start.json
        start_yaml = os.path.join(os.path.dirname(root_dir), "cfg_start.yaml")
        if os.path.exists(start_json):
            shutil.copy2(start_json, stem + ".cfg_start.json")
        if os.path.exists(start_yaml):
            shutil.copy2(start_yaml, stem + ".cfg_start.yaml")
    except Exception as e:
        print(f"[BEST][warn] start-config copy failed: {e}")

    # 레지스트리 기록
    _append_ckpt_registry(root_dir, tag="best", epoch=epoch, iteration=iteration,
                          val_score=score, path=full_path)

    return full_path, alias_path


def save_as_last(yolact_net, score, epoch, iteration, root_dir, alias_name='last.pth', tag='last'):
    """종료 시점 가중치 저장: tag in {'last','earlystop','interrupt','nan'}"""
    safe_score = (float(score) if (score is not None and math.isfinite(float(score))) else float('nan'))
    fname = f"{tag}_iter{iteration:07d}_ep{epoch:03d}_score{safe_score:.4f}.pth"
    full_path = os.path.join(root_dir, fname)
    print(f"[LAST] Saving {tag} checkpoint: score={safe_score:.4f} at iter={iteration}, epoch={epoch}")
    yolact_net.save_weights(full_path)
    # 별칭 갱신
    alias_path = os.path.join(root_dir, alias_name)
    try:
        shutil.copy2(full_path, alias_path)
    except Exception as e:
        print(f"[LAST][warn] alias copy failed: {e}")

    # 레지스트리 기록
    _append_ckpt_registry(root_dir, tag=tag, epoch=epoch, iteration=iteration,
                          val_score=safe_score, path=full_path)
    return full_path, alias_path


# -------- train --------
def train():
    global _paths

    # 실험 디렉토리 구성
    _paths = _prepare_experiment_dirs(args, cfg)

    # pre-init 스냅샷
    _dump_effective_run(
        _paths['exp_dir'], args, cfg,
        extra={
            "stage": "pre_init",
            "optimizer_choice": (args.optim or 'adamw'),
            "scheduler_choice": (args.scheduler or 'cosine'),
            "autoscale_applied": bool(args.autoscale and args.batch_size != 8),
            "resolved_backbone_path": None
        },
        fname_prefix="effective_pre"
    )

    # save/log 폴더를 실험 트리로 고정
    args.save_folder = _paths['weights_dir']
    args.log_folder  = _paths['exp_dir']
    os.makedirs(args.save_folder, exist_ok=True)

    dataset = COCODetection(
        image_path=cfg.dataset.train_images,
        info_file=cfg.dataset.train_info,
        transform=SSDAugmentation(MEANS)
    )
    val_dataset = None
    if args.validation_epoch > 0:
        setup_eval()
        val_dataset = COCODetection(
            image_path=cfg.dataset.valid_images,
            info_file=cfg.dataset.valid_info,
            transform=BaseTransform(MEANS)
        )

    # 모델
    yolact_net = Yolact().to(device)
    net = yolact_net
    net.train()

    # 로깅
    log = None
    if args.log:
        log = Log(cfg.name, args.log_folder, dict(args._get_kwargs() ),
                  overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)

    wandb_logger = None
    if args.use_wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=(args.wandb_name or getattr(cfg, 'name', 'yolact')),
            args=args, enabled=True, debug=args.wandb_debug,
            log_tables=args.wandb_tables, default_max_images=2
        )
        wandb_logger.watch(yolact_net, log_gradients_every=0)

    timer.disable_all()

    # Early Stopping 초기화
    early_stopper = None
    if args.early_stop:
        early_stopper = EarlyStopping(
            patience=args.patience,
            min_delta=args.min_delta,
            mode='max'  # mAP는 클수록 좋음
        )
        print(f"[EarlyStopping] Enabled with patience={args.patience}, min_delta={args.min_delta}, prefer={args.prefer_metric}")

    # Best Checkpoint Manager
    best_ckpt_mgr = BestCheckpointManager(
        root_dir=args.save_folder,
        keep_n=args.keep_best_n
    )

    # NaN streak 추적
    nan_streak = 0
    # 종료 이유 추적
    training_stop_reason = None  # 'earlystop' | 'nan' | 'interrupt' | None

    # 체크포인트
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    _resolved_backbone_path = None
    if args.resume is not None:
        print(f'Resuming training, loading {args.resume}...')
        yolact_net.load_weights(args.resume)
        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
    else:
        print('Initializing weights...')
        backbone_path = _resolve_backbone_path(args, cfg)
        _resolved_backbone_path = backbone_path
        print(f"[backbone] using: {backbone_path}")
        yolact_net.init_weights(backbone_path=backbone_path)

        # 학습 '시작 설정' 확정 스냅샷
        _dump_effective_run(
            _paths['exp_dir'],
            args,
            cfg,
            extra={"stage": "run_start", "resolved_backbone_path": _resolved_backbone_path},
            fname_prefix="cfg_start"
        )

    # 손실
    criterion = MultiBoxLoss(
        num_classes=cfg.num_classes,
        pos_threshold=cfg.positive_iou_threshold,
        neg_threshold=cfg.negative_iou_threshold,
        negpos_ratio=cfg.ohem_negpos_ratio
    )

    # 배치 할당
    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != args.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' %
                  (args.batch_alloc, args.batch_size))
            exit(-1)

    # NetLoss 래핑 + DP/Single 자동 선택
    netloss = NetLoss(yolact_net, criterion, device).to(device)
    if use_cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(netloss).cuda()
    else:
        net = SingleDeviceWrapper(netloss, device)

    # BN freeze
    yolact_net.freeze_bn(bool(getattr(cfg, 'freeze_bn', False)))

    # Optimizer & Scheduler
    opt_name = (args.optim or 'adamw').lower()
    adamw_wd = getattr(cfg, 'adamw_weight_decay', None)
    wd = (adamw_wd if opt_name in ['adamw', 'adamwr'] and adamw_wd is not None
          else (args.decay if args.decay is not None else 0.0))
    param_groups = build_param_groups(yolact_net, wd=wd)

    if opt_name in ['adamw', 'adamwr']:
        # AdamW / AdamWR 공통 (AdamWR은 스케줄을 cosine_restart로 쓰는 패턴)
        betas = getattr(cfg, 'adamw_betas', (0.9, 0.999))
        eps   = getattr(cfg, 'adamw_eps', 1e-8)
        optimizer = optim.AdamW(param_groups, lr=args.lr, betas=betas, eps=eps)
    elif opt_name == 'sgd':
        optimizer = optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, weight_decay=0.0)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    # Scheduler 설정
    sched_name = (args.scheduler or 'cosine').lower()
    use_cosine         = (sched_name == 'cosine')
    use_cosine_restart = (sched_name == 'cosine_restart')

    scheduler = None
    if use_cosine:
        # 한 번만 쭉 cosine으로 감소 → eta_min까지 수렴
        tmax = max(1, cfg.max_iter - cfg.lr_warmup_until)
        scheduler = CosineAnnealingLR(optimizer, T_max=tmax, eta_min=2e-5)

    elif use_cosine_restart:
        # Warm restarts: 여러 번 cosine 사이클 반복
        total_after_warmup = max(1, cfg.max_iter - cfg.lr_warmup_until)

        T_mult = int(args.cosine_tmult) if args.cosine_tmult is not None else 2

        if args.cosine_t0 is not None:
            T_0 = max(1, int(args.cosine_t0))
        else:
            # 전체 post-warmup 구간에서 대략 3~4 사이클 나오도록 자동 설정
            num_cycles = 4
            geom_sum = sum([T_mult**i for i in range(num_cycles)])
            T_0 = max(1, int(total_after_warmup / geom_sum))

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=2e-5,
        )

    # step 스케줄은 기존 cfg.lr_steps / gamma 로 처리
    # (scheduler=None 이고 step 로직은 루프 안에서 동작)

    _dump_effective_run(
        _paths['exp_dir'], args, cfg,
        extra={
            "stage": "post_opt",
            "optimizer": type(optimizer).__name__,
            "optimizer_params": {
                "lr": args.lr,
                "weight_decay_groups": [pg.get("weight_decay", None) for pg in optimizer.param_groups],
                "opt_name": opt_name,
            },
            "scheduler": (type(scheduler).__name__ if scheduler is not None else None),
            "scheduler_T_max": getattr(scheduler, "T_max", None),
            "scheduler_T_0": getattr(scheduler, "T_0", None),
            "scheduler_T_mult": getattr(scheduler, "T_mult", None),
            "resolved_backbone_path": _resolved_backbone_path,
            "early_stopping": {
                "enabled": args.early_stop,
                "patience": args.patience,
                "min_delta": args.min_delta
            }
        },
        fname_prefix="effective"
    )

    # 상태 변수
    iteration = max(args.start_iter, 0)
    last_time = time.time()
    epoch_size = max(1, len(dataset) // args.batch_size)
    num_epochs = math.ceil(cfg.max_iter / epoch_size)
    step_index = 0

    data_loader = data.DataLoader(
        dataset, args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=detection_collate,
        pin_memory=(device.type == 'cuda')
    )

    save_path = lambda epoch, it: SavePath(cfg.name, epoch, it).get_path(root=args.save_folder)
    time_avg = MovingAverage()
    loss_avgs = {k: MovingAverage(100) for k in loss_types}

    best_score = float('-inf')

    # scalars csv
    scalars_csv_path = os.path.join(_paths['curves_dir'], 'train_scalars.csv')
    _ensure_dir(os.path.dirname(scalars_csv_path))

    print('Begin training!\n')

    try:
        for epoch in range(num_epochs):
            if (epoch + 1) * epoch_size < iteration:
                continue

            # reset CUDA peak mem per epoch (optional)
            if torch.cuda.is_available():
                try: torch.cuda.reset_peak_memory_stats()
                except Exception: pass

            # === measure data_time via wrapper ===
            for datum, data_time in iter_with_timer(data_loader):
                if iteration == (epoch + 1) * epoch_size: break
                if iteration == cfg.max_iter: break

                # delayed settings
                changed = False
                for change in cfg.delayed_settings:
                    if iteration >= change[0]:
                        changed = True
                        cfg.replace(change[1])
                        for avg in loss_avgs.values():
                            avg.reset()
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]
                    _dump_effective_run(
                        _paths['exp_dir'], args, cfg,
                        extra={"stage": f"after_delayed_{iteration}",
                               "resolved_backbone_path": _resolved_backbone_path},
                        fname_prefix=f"effective_iter{iteration:07d}"
                    )

                # warmup
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    set_lr(optimizer, (args.lr - cfg.lr_warmup_init) *
                           (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

                # step lr (cosine 계열이 아닐 때만)
                if not (use_cosine or use_cosine_restart):
                    while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                        step_index += 1
                        set_lr(optimizer, args.lr * (args.gamma ** step_index))

                optimizer.zero_grad(set_to_none=True)

                # === Forward + Loss ===
                fwd_t0 = time.time()
                losses = net(datum)
                losses = {k: (v).mean() for k, v in losses.items()}
                loss = sum(losses.values())
                step_time_forward = time.time() - fwd_t0

                # NaN/Inf guard (forward)
                if (not all_finite_dict(losses)) or (not torch.isfinite(loss)):
                    log_nonfinite_losses(losses, tag=f"(iter {iteration})")
                    optimizer.zero_grad(set_to_none=True)
                    nan_streak += 1
                    if nan_streak >= args.max_nan_streak:
                        print(f"\n[EARLY STOP] Consecutive NaN losses ({nan_streak}) exceeded max_nan_streak={args.max_nan_streak}")
                        training_stop_reason = 'nan'
                        raise KeyboardInterrupt
                    iteration += 1
                    continue
                else:
                    nan_streak = 0

                # Grad norm before clip
                grad_norm_total = None
                try:
                    grad_norm_total = _global_grad_norm(yolact_net)
                except Exception:
                    pass

                # Backward
                loss.backward()

                # grad NaN guard
                bad_grad = False
                for p in yolact_net.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            bad_grad = True; break
                if bad_grad:
                    print(f"[NaN guard] NaN/Inf in gradients (iter {iteration}) — skipping optimizer.step()")
                    optimizer.zero_grad(set_to_none=True)
                    nan_streak += 1
                    if nan_streak >= args.max_nan_streak:
                        print(f"\n[EARLY STOP] Consecutive NaN gradients ({nan_streak}) exceeded max_nan_streak={args.max_nan_streak}")
                        training_stop_reason = 'nan'
                        raise KeyboardInterrupt
                    iteration += 1
                    continue
                else:
                    nan_streak = 0

                # (선택) clip
                clip_max = 1.0
                torch.nn.utils.clip_grad_norm_(yolact_net.parameters(), max_norm=clip_max)

                # step
                step_t0 = time.time()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_time_backward = time.time() - step_t0

                # cosine / cosine_restart 스케줄 step (워밍업 이후)
                if (use_cosine or use_cosine_restart) and iteration > cfg.lr_warmup_until and scheduler is not None:
                    scheduler.step()

                # 시간/평균
                cur_time = time.time()
                elapsed = cur_time - last_time          # 전체 step wall time
                last_time = cur_time
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                for k, v in losses.items():
                    vi = v.item() if torch.is_tensor(v) else float(v)
                    if math.isfinite(vi):
                        loss_avgs[k].add(vi)

                # --- collect scalars ---
                lrs, wds = _lr_wd_groups(optimizer)
                mem = _mem_stats()
                batch_imgs = args.batch_size
                throughput = (batch_imgs / elapsed) if elapsed > 0 else 0.0

                scalars = {
                    "epoch": epoch,
                    "iter": iteration,
                    "lr_group0": lrs[0] if len(lrs)>0 else None,
                    "lr_group1": lrs[1] if len(lrs)>1 else None,
                    "wd_group0": wds[0] if len(wds)>0 else None,
                    "wd_group1": wds[1] if len(wds)>1 else None,
                    "loss_total": float(loss.item()) if torch.is_tensor(loss) else float(loss),
                    **{f"loss_{k}": float(v.item()) for k,v in losses.items() if torch.is_tensor(v)},
                    "data_time_s": float(data_time),
                    "fwd_time_s": float(step_time_forward),
                    "bwd_step_time_s": float(step_time_backward),
                    "step_time_s": float(elapsed),
                    "throughput_img_per_s": float(throughput),
                    "vram_alloc_GB": mem.get("vram_alloc_GB"),
                    "vram_reserved_GB": mem.get("vram_reserved_GB"),
                    "vram_peak_GB": mem.get("vram_peak_GB"),
                    "cpu_used_GB": mem.get("cpu_used_GB"),
                    "cpu_total_GB": mem.get("cpu_total_GB"),
                    "grad_norm": float(grad_norm_total) if grad_norm_total is not None else None,
                }

                # 콘솔 로그
                if iteration % 10 == 0:
                    eta_seconds = max(0, (cfg.max_iter - iteration)) * max(1e-6, time_avg.get_avg())
                    eta_str = str(timedelta(seconds=eta_seconds)).split('.')[0]
                    total = sum([loss_avgs[k].get_avg() for k in losses])
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in losses], [])
                    print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) +
                           ' T: %.3f || ETA: %s || dt: %.3f || tp: %.1f img/s')
                          % tuple([epoch, iteration] + loss_labels +
                                  [total, eta_str, data_time, throughput]),
                          flush=True)

                # 파일 로그 + W&B 스칼라
                if args.log:
                    precision = 5
                    _loss_info = {}
                    for k, v in losses.items():
                        vi = v.item() if torch.is_tensor(v) else float(v)
                        if math.isfinite(vi):
                            _loss_info[k] = round(vi, precision)
                    tval = loss.item() if torch.is_tensor(loss) else float(loss)
                    if math.isfinite(tval):
                        _loss_info['T'] = round(tval, precision)

                    if args.log_gpu:
                        log.log_gpu_stats = (iteration % 10 == 0)
                    log.log('train', loss=_loss_info, epoch=epoch, iter=iteration,
                            lr=round(cur_lr, 10), elapsed=elapsed)
                    # 원래 설정 복구
                    log.log_gpu_stats = args.log_gpu

                if wandb_logger is not None:
                    # 손실/핵심
                    wandb_logger.log_scalars(step=iteration, losses=losses, total_loss=loss,
                                             lr=cur_lr, epoch=epoch, elapsed_sec=elapsed)
                    # 시스템/옵티마 스칼라
                    wandb_logger.log_scalar_dict(step=iteration, d={
                        "sys/data_time_s": scalars["data_time_s"],
                        "sys/step_time_s": scalars["step_time_s"],
                        "sys/fwd_time_s": scalars["fwd_time_s"],
                        "sys/bwd_step_time_s": scalars["bwd_step_time_s"],
                        "sys/throughput_img_per_s": scalars["throughput_img_per_s"],
                        "sys/vram_alloc_GB": scalars["vram_alloc_GB"],
                        "sys/vram_reserved_GB": scalars["vram_reserved_GB"],
                        "sys/vram_peak_GB": scalars["vram_peak_GB"],
                        "sys/cpu_used_GB": scalars["cpu_used_GB"],
                        "opt/grad_norm": scalars["grad_norm"],
                        "opt/lr_group0": scalars["lr_group0"],
                        "opt/lr_group1": scalars["lr_group1"],
                        "opt/wd_group0": scalars["wd_group0"],
                        "opt/wd_group1": scalars["wd_group1"],
                    })
                    if args.wandb_images_every > 0 and (iteration % args.wandb_images_every == 0):
                        wandb_logger.log_batch_overlays(step=iteration, datum=datum, model=yolact_net,
                                                        device=device, tag="train/overlay",
                                                        max_images=2, score_thr=float(args.wandb_score_thr))

                # per-10-iters CSV append (가볍게)
                if iteration % 10 == 0:
                    _append_row_csv(scalars_csv_path, scalars)

                # 저장
                iteration += 1
                if iteration % args.save_interval == 0 and iteration != args.start_iter:
                    if args.keep_latest:
                        latest = SavePath.get_latest(args.save_folder, cfg.name)
                    print('Saving state, iter:', iteration)
                    yolact_net.save_weights(save_path(epoch, iteration))
                    if args.keep_latest and latest is not None:
                        if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                            print('Deleting old save...')
                            try: os.remove(latest)
                            except Exception: pass

            # epoch마다 validation
            if args.validation_epoch > 0:
                if epoch % args.validation_epoch == 0 and epoch > 0 and val_dataset is not None:
                    val_info = compute_validation_map(epoch, iteration, yolact_net, val_dataset,
                                                      log if args.log else None, wandb_logger=wandb_logger)
                    # 현재 점수 계산
                    val_score = pick_val_score(val_info, args.prefer_metric)

                    # Best 체크포인트 저장
                    if args.save_best and isinstance(val_info, dict):
                        if val_score > best_score:
                            best_score = val_score
                            ckpt_path, alias = save_as_best(
                                yolact_net, score=best_score, epoch=epoch,
                                iteration=iteration, root_dir=args.save_folder,
                                alias_name=args.best_alias
                            )
                            best_ckpt_mgr.add(best_score, ckpt_path)
                            if wandb_logger is not None:
                                wandb_logger.log_scalar_dict(step=iteration, d={
                                    'best/score': float(best_score),
                                    'best/iter': int(iteration),
                                    'best/epoch': int(epoch),
                                })

                    # Early Stopping 체크 + 매회 로그
                    if early_stopper is not None and isinstance(val_info, dict):
                        stop_now = early_stopper(val_score)
                        print(f"[ES] epoch={epoch:03d} | val_score={val_score:.4f} | "
                              f"best={early_stopper.best_score:.4f} | "
                              f"patience={early_stopper.counter}/{early_stopper.patience}")
                        if wandb_logger is not None:
                            wandb_logger.log_scalar_dict(step=iteration, d={
                                'es/val_score': float(val_score),
                                'es/best_score': float(early_stopper.best_score),
                                'es/patience_counter': int(early_stopper.counter),
                                'es/patience': int(early_stopper.patience),
                            })
                        if stop_now:
                            print(f"\n[EARLY STOP] No improvement for {args.patience} validation epochs.")
                            print(f"Best score: {early_stopper.best_score:.4f}")
                            training_stop_reason = 'earlystop'
                            raise KeyboardInterrupt  # 학습 종료

        # 마지막 validation (루프 종료 후)
        if args.validation_epoch > 0 and val_dataset is not None:
            val_info = compute_validation_map(epoch, iteration, yolact_net, val_dataset,
                                              log if args.log else None, wandb_logger=wandb_logger)
            val_score = pick_val_score(val_info, args.prefer_metric)
            if args.save_best and isinstance(val_info, dict):
                if val_score > best_score:
                    best_score = val_score
                    ckpt_path, alias = save_as_best(
                        yolact_net, score=best_score, epoch=epoch,
                        iteration=iteration, root_dir=args.save_folder,
                        alias_name=args.best_alias
                    )
                    best_ckpt_mgr.add(best_score, ckpt_path)
                    if wandb_logger is not None:
                        wandb_logger.log_scalar_dict(step=iteration, d={
                            'best/score': float(best_score),
                            'best/iter': int(iteration),
                            'best/epoch': int(epoch),
                        })
        else:
            # 검증 비활성 시 라스트 저장용 점수 없음
            val_score = None

        # 정상 종료 → 라스트 저장
        if args.save_last:
            try:
                save_as_last(yolact_net, score=val_score, epoch=epoch, iteration=iteration,
                             root_dir=args.save_folder, alias_name=args.last_alias, tag='last')
            except Exception as e:
                print(f"[LAST][warn] saving last checkpoint failed: {e}")

    except KeyboardInterrupt:
        # 마지막 1회 검증으로 val_score 확보(가능하면)
        last_val_score = None
        if args.validation_epoch > 0 and val_dataset is not None:
            try:
                print("\n[LAST] Running final validation after interruption...")
                val_info = compute_validation_map(epoch, iteration, yolact_net, val_dataset,
                                                  log if args.log else None, wandb_logger=wandb_logger)
                last_val_score = pick_val_score(val_info, args.prefer_metric)
            except Exception as e:
                print(f"[LAST][warn] final validation failed: {e}")

        # 종료 사유에 따라 태그 결정
        tag = ('earlystop' if training_stop_reason == 'earlystop'
               else ('nan' if training_stop_reason == 'nan' else 'interrupt'))

        if args.save_last:
            try:
                save_as_last(yolact_net, score=last_val_score, epoch=epoch, iteration=iteration,
                             root_dir=args.save_folder, alias_name=args.last_alias, tag=tag)
            except Exception as e:
                print(f"[LAST][warn] saving last checkpoint failed: {e}")

        # 요약/마무리
        if 'early_stopper' in locals() and early_stopper is not None:
            print(f"\n=== Early Stopping Summary ===")
            print(f"Best validation score: {early_stopper.best_score:.4f}")
            print(f"Patience counter: {early_stopper.counter}/{early_stopper.patience}")
        
        if wandb_logger is not None:
            wandb_logger.finish()
        sys.exit(0)

    print("\n=== Training Complete ===")
    if best_score > float('-inf'):
        print(f"Best validation score: {best_score:.4f}")
    
    if wandb_logger is not None:
        wandb_logger.finish()


if __name__ == '__main__':
    train()
