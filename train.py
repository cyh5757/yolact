#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------
# train.py (YOLACT – cell setup, with Best-Checkpoint Saving)
# -----------------------------

from data import *
from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage, SavePath
from utils.logger import Log
from utils import timer
from layers.modules import MultiBoxLoss
from yolact import Yolact

import os, sys, time, math, random, argparse, datetime, json, shutil
from pathlib import Path
from contextlib import nullcontext

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
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


def str2bool(v):
    if isinstance(v, bool): return v
    return v.lower() in ("yes", "true", "t", "1")


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

parser.add_argument('--validation_size', default=5000, type=int)
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
parser.add_argument('--optim', default=None, choices=['sgd', 'adamw'],
                    help='Optimizer: sgd or adamw (default: from cfg)')
parser.add_argument('--scheduler', default=None, choices=['step', 'cosine', 'none'],
                    help='LR scheduler: step | cosine | none (default: from cfg)')

# ---- 실험 디렉토리 ----
parser.add_argument('--exp_dir', type=str, default=None,
                    help='실험 저장 루트. 지정하면 표준 트리 생성')

# ---- Best 저장 옵션 추가 ----
parser.add_argument('--save_best', default=True, type=str2bool,
                    help='Validation 성능이 향상되면 best 체크포인트 저장')
parser.add_argument('--best_alias', default='best.pth', type=str,
                    help='가장 최근 best 모델의 별칭 파일명')
parser.add_argument('--prefer_metric', default='mask_first', type=str,
                    choices=['mask_first','box_first','mask_only','box_only'],
                    help='best 판단 시 어떤 지표를 우선할지')

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

# 설정 주입
if args.config is not None:
    set_cfg(args.config)
if args.dataset is not None:
    set_dataset(args.dataset)

# 배치 스케일링
if args.autoscale and args.batch_size != 8:
    factor = args.batch_size / 8.0
    print('Scaling parameters by %.2f for batch size %d.' % (factor, args.batch_size))
    cfg.lr *= factor
    cfg.max_iter = max(1, int(round(cfg.max_iter / factor)))
    cfg.lr_steps = [max(1, int(round(s / factor))) for s in cfg.lr_steps]

# args에서 비어있으면 config로 대체
def replace_from_cfg(name):
    if getattr(args, name) is None:
        setattr(args, name, getattr(cfg, name))
for k in ['lr', 'decay', 'gamma', 'momentum']:
    replace_from_cfg(k)

# optim / scheduler 기본값 채우기
def replace_from_cfg2(name, default=None):
    if getattr(args, name) is None:
        setattr(args, name, getattr(cfg, name, default))
replace_from_cfg2('optim', 'adamw')
replace_from_cfg2('scheduler', 'cosine')

# BN freeze 오버라이드
if args.freeze_bn is not None:
    cfg.freeze_bn = bool(args.freeze_bn)

# 장치
use_cuda = args.cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
if use_cuda:
    cudnn.benchmark = True

# 현재 lr
cur_lr = args.lr

# DP에서 쓰는 손실 키 순서
loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']

# ===== 실험 디렉토리 준비 =====
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
    # exp_dir 기본값: 날짜_anchor_test
    if args.exp_dir is None:
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        args.exp_dir = os.path.join('experiments', f'{today}_anchor_test')

    exp_dir = args.exp_dir
    eval_dir = os.path.join(exp_dir, 'eval')
    weights_dir = os.path.join(exp_dir, 'weights')
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # notes.md 템플릿
    notes = os.path.join(exp_dir, 'notes.md')
    if not os.path.exists(notes):
        with open(notes, 'w', encoding='utf-8') as f:
            f.write(f"# {cfg.name}\n\n- 목적: Anchors 프리셋 비교(AP_large ↑)\n"
                    f"- pred_scales: {cfg.backbone.pred_scales}\n"
                    f"- 생성시각: {datetime.datetime.now()}\n")

    # stdout tee → train.log
    TeeStdout(os.path.join(exp_dir, 'train.log'))

    # config dump
    _dump_cfg_snapshot(exp_dir)

    return {'exp_dir': exp_dir, 'eval_dir': eval_dir, 'weights_dir': weights_dir}

def _epoch_eval_paths(epoch_idx:int):
    export_json = os.path.join(_paths['eval_dir'], f'val_epoch_{epoch_idx:03d}.json')
    metrics_csv = os.path.join(_paths['eval_dir'], 'metrics.csv')
    return export_json, metrics_csv


# ===== NaN / Inf guards =====
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

# ===== Model+Loss wrappers =====
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

def no_inf_mean(x:torch.Tensor):
    no_inf = [a for a in x if torch.isfinite(a)]
    return (sum(no_inf) / len(no_inf)) if len(no_inf) > 0 else x.mean()

def compute_validation_loss(net, data_loader, criterion):
    global loss_types
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
        loss_labels = sum([[k, losses[k]] for k in loss_types if k in losses], [])
        print(('Validation ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)

def compute_validation_map(epoch, iteration, yolact_net, dataset, log:Log=None, wandb_logger:WandbLogger=None):
    # 매 epoch마다 eval 스크립트 인자 재설정: COCO JSON/CSV 경로를 epoch 기반으로 고정
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
        f'--metrics_csv={metrics_csv}'  # eval.py에 패치되어 있어야 함
    ]
    eval_script.parse_args(eval_args)
    eval_script.prep_coco_cats()
    
    with torch.no_grad():
        yolact_net.eval()
        start = time.time()
        print("\nComputing validation mAP (this may take a while)...", flush=True)
        val_info = eval_script.evaluate(yolact_net, dataset, train_mode=True)
        end = time.time()
        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)
        if wandb_logger is not None:
            wandb_logger.log_val_map(epoch=epoch, iteration=iteration, val_info=val_info, elapsed_sec=(end - start))
        yolact_net.train()
        return val_info  # ← 반환하도록 수정

def setup_eval():
    # 초기 한 번 셋업해도 되지만, compute_validation_map에서 매번 parse_args로 덮어쓴다.
    eval_script.parse_args([
        '--no_bar',
        f'--max_images={args.validation_size}',
        '--top_k=100',
        '--score_threshold=0.05',
        '--fast_nms=True',
        '--cross_class_nms=False'
    ])

# AdamW를 위한 param group (Norm/bias WD 제외)
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

# --------- 추가: 백본 경로 해석기 ----------
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
# -------------------------------------------

# ---------- Best 점수 선택기 ----------
def _safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def pick_val_score(val_info: dict, prefer: str = 'mask_first') -> float:
    """
    val_info에서 best를 판단할 단일 스코어를 고른다.
    지원 키 예:
      - val_info['segm']['mAP'] 또는 val_info['mask']['mAP'] 또는 val_info['mask_mAP']
      - val_info['bbox']['mAP'] 또는 val_info['box_mAP']
    """
    mask_map = (
        _safe_get(val_info, 'segm', 'mAP') or
        _safe_get(val_info, 'mask', 'mAP') or
        _safe_get(val_info, 'mask_mAP') or
        _safe_get(val_info, 'mask', 'map') or
        _safe_get(val_info, 'segm', 'map')
    )
    box_map = (
        _safe_get(val_info, 'bbox', 'mAP') or
        _safe_get(val_info, 'box_mAP') or
        _safe_get(val_info, 'bbox', 'map') or
        _safe_get(val_info, 'box', 'mAP')
    )

    if prefer == 'mask_only':
        return float(mask_map) if mask_map is not None else float('-inf')
    if prefer == 'box_only':
        return float(box_map) if box_map is not None else float('-inf')

    if prefer == 'mask_first':
        return float(mask_map) if mask_map is not None else (float(box_map) if box_map is not None else float('-inf'))
    if prefer == 'box_first':
        return float(box_map) if box_map is not None else (float(mask_map) if mask_map is not None else float('-inf'))

    # default
    return float(mask_map) if mask_map is not None else (float(box_map) if box_map is not None else float('-inf'))

def save_as_best(yolact_net, score: float, epoch: int, iteration: int, root_dir: str, alias_name: str):
    fname = f"best_iter{iteration:07d}_ep{epoch:03d}_score{score:.4f}.pth"
    full_path = os.path.join(root_dir, fname)
    print(f"[BEST] New best score={score:.4f} at iter={iteration}, epoch={epoch} → {full_path}")
    yolact_net.save_weights(full_path)

    # 별칭 파일 갱신
    alias_path = os.path.join(root_dir, alias_name)
    try:
        # 파일 복사로 별칭 갱신 (심볼릭 링크 호환 이슈 피함)
        shutil.copy2(full_path, alias_path)
    except Exception as e:
        print(f"[BEST][warn] alias copy failed: {e}")
    return full_path, alias_path
# ----------------------------------------

def train():
    global _paths

    # 실험 디렉토리 생성 및 표준 트리 구성
    _paths = _prepare_experiment_dirs(args, cfg)
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

    # 체크포인트
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    if args.resume is not None:
        print(f'Resuming training, loading {args.resume}...')
        yolact_net.load_weights(args.resume)
        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
    else:
        print('Initializing weights...')
        backbone_path = _resolve_backbone_path(args, cfg)  # 수정된 부분
        print(f"[backbone] using: {backbone_path}")
        yolact_net.init_weights(backbone_path=backbone_path)

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

    # BN freeze 정책
    if cfg.freeze_bn:
        yolact_net.freeze_bn(True)
    else:
        yolact_net.freeze_bn(False)

    # Optimizer & Scheduler
    opt_name = (args.optim or 'adamw').lower()
    adamw_wd = getattr(cfg, 'adamw_weight_decay', None)
    wd = (adamw_wd if opt_name == 'adamw' and adamw_wd is not None
          else (args.decay if args.decay is not None else 0.0))
    param_groups = build_param_groups(yolact_net, wd=wd)

    if opt_name == 'adamw':
        betas = getattr(cfg, 'adamw_betas', (0.9, 0.999))
        eps   = getattr(cfg, 'adamw_eps', 1e-8)
        optimizer = optim.AdamW(param_groups, lr=args.lr, betas=betas, eps=eps)
    else:
        optimizer = optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, weight_decay=0.0)

    sched_name = (args.scheduler or 'cosine').lower()
    use_cosine = (sched_name == 'cosine')
    scheduler = None
    if use_cosine:
        tmax = max(1, cfg.max_iter - cfg.lr_warmup_until)
        scheduler = CosineAnnealingLR(optimizer, T_max=tmax)

    # 상태 변수
    iteration = max(args.start_iter, 0)
    last_time = time.time()
    epoch_size = max(1, len(dataset) // args.batch_size)
    num_epochs = math.ceil(cfg.max_iter / epoch_size)
    step_index = 0

    data_loader = data.DataLoader(
        dataset, args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,                                # 필요시 True로
        collate_fn=detection_collate,
        pin_memory=(device.type == 'cuda')
    )

    save_path = lambda epoch, it: SavePath(cfg.name, epoch, it).get_path(root=args.save_folder)
    time_avg = MovingAverage()
    loss_avgs = {k: MovingAverage(100) for k in loss_types}

    # ----- best 관리 변수 -----
    best_score = float('-inf')
    best_ckpt_path = None

    print('Begin training!\n')

    try:
        for epoch in range(num_epochs):
            if (epoch + 1) * epoch_size < iteration:
                continue

            for datum in data_loader:
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

                # warmup
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    set_lr(optimizer, (args.lr - cfg.lr_warmup_init) *
                           (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

                # step lr (cosine이 아닐 때만)
                if not use_cosine:
                    while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                        step_index += 1
                        set_lr(optimizer, args.lr * (args.gamma ** step_index))

                optimizer.zero_grad(set_to_none=True)

                # === Forward + Loss ===
                losses = net(datum)                       # dict of tensors
                losses = {k: (v).mean() for k, v in losses.items()}
                loss = sum(losses.values())

                # NaN/Inf guard (forward)
                if (not all_finite_dict(losses)) or (not torch.isfinite(loss)):
                    log_nonfinite_losses(losses, tag=f"(iter {iteration})")
                    optimizer.zero_grad(set_to_none=True)
                    iteration += 1
                    continue

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
                    iteration += 1
                    continue

                # (선택) clip
                torch.nn.utils.clip_grad_norm_(yolact_net.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # cosine 스케줄 step (워밍업 이후)
                if use_cosine and iteration > cfg.lr_warmup_until and scheduler is not None:
                    scheduler.step()

                # 시간/평균
                cur_time = time.time()
                elapsed = cur_time - last_time
                last_time = cur_time
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                for k, v in losses.items():
                    vi = v.item() if torch.is_tensor(v) else float(v)
                    if math.isfinite(vi):
                        loss_avgs[k].add(vi)

                # 콘솔 로그
                if iteration % 10 == 0:
                    eta_seconds = max(0, (cfg.max_iter - iteration)) * max(1e-6, time_avg.get_avg())
                    eta_str = str(datetime.timedelta(seconds=eta_seconds)).split('.')[0]
                    total = sum([loss_avgs[k].get_avg() for k in losses])
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])
                    print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) +
                           ' T: %.3f || ETA: %s || timer: %.3f')
                          % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)

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
                    log.log_gpu_stats = args.log_gpu

                if wandb_logger is not None:
                    wandb_logger.log_scalars(step=iteration, losses=losses, total_loss=loss,
                                             lr=cur_lr, epoch=epoch, elapsed_sec=elapsed)
                    if args.wandb_images_every > 0 and (iteration % args.wandb_images_every == 0):
                        wandb_logger.log_batch_overlays(step=iteration, datum=datum, model=yolact_net,
                                                        device=device, tag="train/overlay",
                                                        max_images=2, score_thr=float(args.wandb_score_thr))

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
                    # ---- best 저장 처리 ----
                    if args.save_best and isinstance(val_info, dict):
                        cur_score = pick_val_score(val_info, args.prefer_metric)
                        if cur_score > best_score:
                            best_score = cur_score
                            best_ckpt_path, alias_path = save_as_best(
                                yolact_net, score=best_score, epoch=epoch,
                                iteration=iteration, root_dir=args.save_folder,
                                alias_name=args.best_alias
                            )
                            if wandb_logger is not None:
                                wandb_logger.log_dict({'best/score': best_score,
                                                       'best/iter': iteration,
                                                       'best/epoch': epoch}, step=iteration)

        # 마지막 validation
        if args.validation_epoch > 0 and val_dataset is not None:
            val_info = compute_validation_map(epoch, iteration, yolact_net, val_dataset,
                                              log if args.log else None, wandb_logger=wandb_logger)
            if args.save_best and isinstance(val_info, dict):
                cur_score = pick_val_score(val_info, args.prefer_metric)
                if cur_score > best_score:
                    best_score = cur_score
                    best_ckpt_path, alias_path = save_as_best(
                        yolact_net, score=best_score, epoch=epoch,
                        iteration=iteration, root_dir=args.save_folder,
                        alias_name=args.best_alias
                    )
                    if wandb_logger is not None:
                        wandb_logger.log_dict({'best/score': best_score,
                                               'best/iter': iteration,
                                               'best/epoch': epoch}, step=iteration)

    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network...')
            SavePath.remove_interrupt(args.save_folder)
            yolact_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
        if wandb_logger is not None:
            wandb_logger.finish()
        sys.exit(0)

    # 최종 저장 + W&B 마무리
    yolact_net.save_weights(save_path(epoch, iteration))
    if wandb_logger is not None:
        wandb_logger.finish()


if __name__ == '__main__':
    train()
