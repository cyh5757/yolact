#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================
# eval.py (YOLACT – extended + optimized)
# - COCO 평가 + 시각화
# - per-image 프로파일(FPS/메모리)
# - Params / FLOPs / FPS 요약 + CSV
# - AJI (Aggregated Jaccard Index) 계산 추가
# - size-bucket(AP_s/m/l) CSV 기록 시 AJI 함께 기록
# - train_mode 최적화 추가
# =============================

from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform
from utils.functions import MovingAverage, ProgressBar, SavePath
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import statistics as stats
try:
    import psutil
except Exception:
    psutil = None

import random
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
import cv2
import matplotlib.pyplot as plt
import csv  # CSV 기록
from typing import Optional, Tuple

# COCOeval for size-bucket AP
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ----------------- Repro -----------------
def set_seed(s=1337):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ----------------- CLI -----------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model', default='weights/ssd300_mAP_77.43_v2.pth', type=str)
    parser.add_argument('--top_k', default=5, type=int)
    parser.add_argument('--cuda', default=True, type=str2bool)
    parser.add_argument('--fast_nms', default=True, type=str2bool)
    parser.add_argument('--cross_class_nms', default=False, type=str2bool)
    parser.add_argument('--display_masks', default=True, type=str2bool)
    parser.add_argument('--display_bboxes', default=True, type=str2bool)
    parser.add_argument('--display_text', default=True, type=str2bool)
    parser.add_argument('--display_scores', default=True, type=str2bool)
    parser.add_argument('--display', dest='display', action='store_true')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str)
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--max_images', default=-1, type=int)

    # COCO 결과 내보내기
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str)
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str)

    parser.add_argument('--config', default=None)
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true')
    parser.add_argument('--web_det_path', default='web/dets/', type=str)
    parser.add_argument('--no_bar', dest='no_bar', action='store_true')
    parser.add_argument('--display_lincomb', default=False, type=str2bool)
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false')
    parser.add_argument('--image', default=None, type=str)
    parser.add_argument('--images', default=None, type=str)
    parser.add_argument('--video', default=None, type=str)
    parser.add_argument('--video_multiframe', default=1, type=int)
    parser.add_argument('--score_threshold', default=0.0, type=float)
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--detect', default=False, dest='detect', action='store_true')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true')

    # 외부 COCO 경로 오버라이드 & 단일 COCO 결과 파일
    parser.add_argument('--coco_images_dir', default=None, type=str)
    parser.add_argument('--coco_ann_file', default=None, type=str)
    parser.add_argument('--export_coco', default=None, type=str)

    # metrics.csv (AP_s/m/l + AJI 같이 기록)
    parser.add_argument('--metrics_csv', default=None, type=str)

    # Inference profiler options
    parser.add_argument('--profile_infer', type=str2bool, default=True)
    parser.add_argument('--profile_warmup', type=int, default=2)
    parser.add_argument('--profile_csv', type=str, default=None)

    # Model Stats (Params / FLOPs / Throughput)
    parser.add_argument('--model_stats', type=str2bool, default=True)
    parser.add_argument('--flops_hw', type=str, default=None)   # 예: 700x700
    parser.add_argument('--flops_use_dummy', type=str2bool, default=True)
    parser.add_argument('--stats_csv', type=str, default=None)

    # AJI 옵션
    parser.add_argument('--compute_aji', type=str2bool, default=True,
                        help='평가 중 AJI 계산(세포/핵 데이터 권장)')
    parser.add_argument('--aji_mask_thresh', type=float, default=0.5,
                        help='예측 마스크를 이진화할 임계값')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False,
                        shuffle=False, benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False,
                        crop=True, detect=False, display_fps=False, emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True
    if args.export_coco and not args.output_coco_json:
        args.output_coco_json = True
    if args.seed is not None:
        random.seed(args.seed)


# ----------------- Globals -----------------
iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {}
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})


def prep_coco_cats():
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inv[coco_cat_id] = transformed_cat_id


def get_coco_cat(transformed_cat_id):  # [0, num_classes)
    return coco_cats[transformed_cat_id]


def get_transformed_cat(coco_cat_id):
    return coco_cats_inv[coco_cat_id]


# ----------------- Visualization -----------------
def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')

    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.tensor(img_numpy, device=device, dtype=torch.float32)
    else:
        img_gpu = (img.to(device=device, dtype=torch.float32) / 255.0) if torch.is_tensor(img) \
                  else torch.tensor(img, device=device, dtype=torch.float32) / 255.0
        h, w, _ = img_gpu.shape

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb=args.display_lincomb,
                        crop_masks=args.crop, score_threshold=args.score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:args.top_k]
        if cfg.eval_mask_branch:
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].detach().to('cpu').numpy() for x in t[:3]]

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    def get_color(j, on_gpu_device=None):
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        color = COLORS[color_idx]
        if not undo_transform:
            color = (color[2], color[1], color[0])
        return color

    if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        masks = masks[:num_dets_to_consider, :, :, None]
        colors = torch.cat([torch.tensor(get_color(j), device=device, dtype=torch.float32).view(1, 1, 1, 3) / 255.0
                            for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
        inv_alph_masks = masks * (-mask_alpha) + 1

        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    if args.display_fps:
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1
        fps_str = fps_str or ""
        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]
        img_gpu[:text_h + 8, :text_w + 8] *= 0.6

    img_numpy = (img_gpu * 255).byte().to('cpu').numpy()

    if args.display_fps:
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]
        cv2.putText(img_numpy, fps_str, cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 1, cv2.LINE_AA)

    if num_dets_to_consider == 0:
        return img_numpy

    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]
            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class
                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1
                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]
                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return img_numpy


# ----------------- Eval Utils -----------------
def _mask_iou(mask1, mask2, iscrowd=False):
    with timer.env('Mask IoU'):
        ret = mask_iou(mask1, mask2, iscrowd)
    return ret.cpu()


def _bbox_iou(bbox1, bbox2, iscrowd=False):
    with timer.env('BBox IoU'):
        ret = jaccard(bbox1, bbox2, iscrowd)
    return ret.cpu()


class Detections:
    def __init__(self):
        self.bbox_data = []
        self.mask_data = []

    def add_bbox(self, image_id: int, category_id: int, bbox: list, score: float):
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        bbox = [round(float(x) * 10) / 10 for x in bbox]
        self.bbox_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'bbox': bbox,
            'score': float(score)
        })

    def add_mask(self, image_id: int, category_id: int, segmentation: np.ndarray, score: float):
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')
        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'segmentation': rle,
            'score': float(score)
        })

    def dump(self):
        # bbox/mask 분리 저장
        for data, path in [(self.bbox_data, args.bbox_det_file), (self.mask_data, args.mask_det_file)]:
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'w') as f:
                    json.dump(data, f)

        # 단일 COCO 파일(보통 mask 중심)
        if getattr(args, 'export_coco', None):
            results = self.mask_data
            os.makedirs(os.path.dirname(args.export_coco), exist_ok=True)
            with open(args.export_coco, 'w') as f:
                json.dump(results, f)
            print(f'[OK] Exported COCO-style results to: {args.export_coco}')


# ----------------- AJI -----------------
class AJIAccumulator:
    """
    Dataset-level AJI 누적기.
    표준 AJI(= greedily matched pairs의 intersection 합 / union 합(+unmatched preds))
    """
    def __init__(self, thr: float = 0.5):
        self.num_sum = 0.0
        self.den_sum = 0.0
        self.thr = float(thr)

    @staticmethod
    def _pairwise_iou(gt_bin, pr_bin):
        """
        gt_bin: (Ng, H, W) bool
        pr_bin: (Np, H, W) bool
        return: (Ng, Np) IoU matrix
        """
        if gt_bin.size == 0 or pr_bin.size == 0:
            return np.zeros((gt_bin.shape[0], pr_bin.shape[0]), dtype=np.float32)
        Ng, H, W = gt_bin.shape
        Np = pr_bin.shape[0]
        gt_flat = gt_bin.reshape(Ng, -1).astype(np.uint8)
        pr_flat = pr_bin.reshape(Np, -1).astype(np.uint8)
        inter = (gt_flat[:, None, :] & pr_flat[None, :, :]).sum(axis=2).astype(np.float64)
        area_g = gt_flat.sum(axis=1).astype(np.float64)[:, None]
        area_p = pr_flat.sum(axis=1).astype(np.float64)[None, :]
        union = area_g + area_p - inter
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = np.where(union > 0.0, inter / union, 0.0)
        return iou

    def add_image(self, gt_masks_hw: np.ndarray, pred_masks_hw: np.ndarray):
        """
        gt_masks_hw: (Ng, H, W) {0,1}
        pred_masks_hw: (Np, H, W) float or {0,1}; 내부에서 이진화
        """
        if pred_masks_hw.dtype != np.uint8 and pred_masks_hw.dtype != np.bool_:
            pred_masks_hw = (pred_masks_hw >= self.thr).astype(np.uint8)
        else:
            pred_masks_hw = (pred_masks_hw > 0).astype(np.uint8)
        gt_masks_hw = (gt_masks_hw > 0).astype(np.uint8)

        Ng = gt_masks_hw.shape[0]
        Np = pred_masks_hw.shape[0]
        if Ng == 0 and Np == 0:
            return
        if Ng == 0:
            # GT가 없다면 분모에 pred 영역만 들어감 (분자 0)
            self.num_sum += 0.0
            self.den_sum += float(pred_masks_hw.sum())
            return
        if Np == 0:
            # Pred가 없다면 분자 0, 분모에 gt 영역만
            self.num_sum += 0.0
            self.den_sum += float(gt_masks_hw.sum())
            return

        iou = self._pairwise_iou(gt_masks_hw, pred_masks_hw)

        # Greedy matching: 매번 최대 IoU 쌍을 선택 (IoU==0이면 종료)
        used_g = set()
        used_p = set()
        inter_sum = 0.0
        union_sum = 0.0

        while True:
            # 남은 쌍 중 최대 IoU 찾기
            max_i, max_j, max_v = -1, -1, 0.0
            for gi in range(Ng):
                if gi in used_g: continue
                for pj in range(Np):
                    if pj in used_p: continue
                    v = iou[gi, pj]
                    if v > max_v:
                        max_v = v; max_i = gi; max_j = pj
            if max_v <= 0.0:  # 더 매칭할 게 없음
                break

            # 매칭된 쌍의 inter/union 재계산
            g = gt_masks_hw[max_i].astype(np.uint8)
            p = pred_masks_hw[max_j].astype(np.uint8)
            inter = int((g & p).sum())
            union = int((g | p).sum())
            inter_sum += inter
            union_sum += union
            used_g.add(max_i); used_p.add(max_j)

        # 남은 unmatched GT는 분모에 영역을 더함 (분자는 0)
        for gi in range(Ng):
            if gi not in used_g:
                union_sum += int(gt_masks_hw[gi].sum())

        # 남은 unmatched Pred도 분모에 영역을 더함
        for pj in range(Np):
            if pj not in used_p:
                union_sum += int(pred_masks_hw[pj].sum())

        self.num_sum += inter_sum
        self.den_sum += max(union_sum, 1.0)

    def value(self):
        if self.den_sum <= 0.0:
            return 0.0
        return float(self.num_sum / self.den_sum)


def prep_metrics(ap_data, dets, img, gt, gt_masks, h, w, num_crowd, image_id,
                 detections: Detections = None,
                 aji_acc: AJIAccumulator = None):
    device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')

    # transform 후 실제 이미지 크기 (img: (C,H,W))
    img_h, img_w = img.shape[1], img.shape[2]

    crowd_boxes = crowd_masks = crowd_classes = None

    # --- GT 준비 ---
    with timer.env('Prepare gt'):
        gt_boxes = torch.tensor(gt[:, :4], dtype=torch.float32, device=device)
        gt_boxes[:, [0, 2]] *= w
        gt_boxes[:, [1, 3]] *= h
        # transform 후 크기로 스케일링
        gt_boxes[:, [0, 2]] *= (img_w / w)
        gt_boxes[:, [1, 3]] *= (img_h / h)
        gt_classes = list(gt[:, 4].astype(int))

        if not isinstance(gt_masks, torch.Tensor):
            gt_masks = torch.tensor(gt_masks, dtype=torch.float32, device=device)
        else:
            gt_masks = gt_masks.detach().clone().to(torch.float32).to(device)

        # img_h, img_w 기준으로 reshape
        if len(gt_masks.shape) == 3:
            num_masks, mask_h, mask_w = gt_masks.shape
            gt_masks_vec = gt_masks.view(num_masks, mask_h * mask_w)
        else:
            gt_masks_vec = gt_masks.view(-1, img_h * img_w)

        if num_crowd > 0:
            split = lambda x: (x[-num_crowd:], x[:-num_crowd])
            crowd_boxes, gt_boxes = split(gt_boxes)
            crowd_masks, gt_masks_vec = split(gt_masks_vec)
            crowd_classes, gt_classes = split(gt_classes)

    with timer.env('Postprocess'):
        classes, scores, boxes, masks = postprocess(
            dets, img_w, img_h, crop_masks=args.crop, score_threshold=args.score_threshold
        )
        if classes.size(0) == 0:
            # AJI도 0/분모만 더해짐 (unmatched gt만 있을 때는 evaluate()에서 다음 이미지로)
            if aji_acc is not None:
                aji_acc.add_image(gt_masks.view(-1, img_h, img_w).detach().cpu().numpy(),
                                  np.zeros((0, img_h, img_w), dtype=np.uint8))
            return

        classes = list(classes.detach().to('cpu').numpy().astype(int))
        if isinstance(scores, list):
            box_scores = list(scores[0].detach().to('cpu').numpy().astype(float))
            mask_scores = list(scores[1].detach().to('cpu').numpy().astype(float))
        else:
            scores = list(scores.detach().to('cpu').numpy().astype(float))
            box_scores = scores
            mask_scores = scores

        masks_flat = masks.to(device=device).view(-1, img_h * img_w)
        boxes = boxes.to(device=device)

    # --- JSON 덤프 ---
    if args.output_coco_json:
        with timer.env('JSON Output'):
            boxes_np = boxes.detach().to('cpu').numpy()
            masks_np = masks_flat.view(-1, img_h, img_w).detach().to('cpu').numpy()
            for i in range(masks_np.shape[0]):
                if (boxes_np[i, 3] - boxes_np[i, 1]) * (boxes_np[i, 2] - boxes_np[i, 0]) > 0:
                    detections.add_bbox(image_id, classes[i], boxes_np[i, :], box_scores[i])
                    detections.add_mask(image_id, classes[i], masks_np[i, :, :], mask_scores[i])

    # ====== 평가 경로 (AP 계산) ======
    with timer.env('Eval Setup'):
        num_pred = len(classes)

        bbox_iou_cache = jaccard(boxes.float(), gt_boxes.float()).cpu()
        mask_iou_cache = mask_iou(masks_flat, gt_masks_vec).cpu()

        if num_crowd > 0 and (crowd_boxes is not None) and (crowd_masks is not None):
            crowd_bbox_iou_cache = jaccard(boxes.float(), crowd_boxes.float(), iscrowd=True).cpu()
            crowd_mask_iou_cache = mask_iou(masks_flat, crowd_masks, iscrowd=True).cpu()
        else:
            crowd_bbox_iou_cache = crowd_mask_iou_cache = None

        box_indices = sorted(range(num_pred), key=lambda i: -box_scores[i])
        mask_indices = sorted(box_indices, key=lambda i: -mask_scores[i])

        iou_types = [
            ('box',
             lambda i, j: bbox_iou_cache[i, j].item(),
             lambda i, j: crowd_bbox_iou_cache[i, j].item() if crowd_bbox_iou_cache is not None else 0.0,
             lambda i: box_scores[i], box_indices),

            ('mask',
             lambda i, j: mask_iou_cache[i, j].item(),
             lambda i, j: crowd_mask_iou_cache[i, j].item() if crowd_mask_iou_cache is not None else 0.0,
             lambda i: mask_scores[i], mask_indices)
        ]

    timer.start('Main loop')
    for _class in set(classes + gt_classes):
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])

        for iouIdx, iou_threshold in enumerate(iou_thresholds):
            for iou_type, iou_func, crowd_func, score_func, indices in iou_types:
                gt_used = [False] * len(gt_classes)
                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_for_class)

                for i in indices:
                    if classes[i] != _class:
                        continue
                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    for j in range(len(gt_classes)):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue
                        iou = iou_func(i, j)
                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j
                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(score_func(i), True)
                    else:
                        matched_crowd = False
                        if num_crowd > 0 and crowd_classes is not None:
                            for j in range(len(crowd_classes)):
                                if crowd_classes[j] != _class:
                                    continue
                                iou = crowd_func(i, j)
                                if iou > iou_threshold:
                                    matched_crowd = True
                                    break
                        if not matched_crowd:
                            ap_obj.push(score_func(i), False)
    timer.stop('Main loop')

    # ====== AJI 누적 ======
    if aji_acc is not None:
        gt_np = gt_masks.view(-1, img_h, img_w).detach().cpu().numpy()
        pr_np = masks_flat.view(-1, img_h, img_w).detach().cpu().numpy()
        aji_acc.add_image(gt_np, pr_np)


class APDataObject:
    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0
    def push(self, score: float, is_true: bool):
        self.data_points.append((score, is_true))
    def add_gt_positives(self, num_positives: int):
        self.num_gt_positives += num_positives
    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0
    def get_ap(self) -> float:
        if self.num_gt_positives == 0:
            return 0
        self.data_points.sort(key=lambda x: -x[0])
        precisions, recalls = [], []
        num_true, num_false = 0, 0
        for score, is_true in self.data_points:
            if is_true: num_true += 1
            else:       num_false += 1
            precision = num_true / (num_true + num_false)
            recall    = num_true / self.num_gt_positives
            precisions.append(precision)
            recalls.append(recall)
        for i in range(len(precisions) - 1, 0, -1):
            if precisions[i] > precisions[i - 1]:
                precisions[i - 1] = precisions[i]
        y_range = [0] * 101
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]
        return sum(y_range) / len(y_range)


# ----------------- Single/Folder Eval -----------------
def badhash(x):
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = ((x >> 16) ^ x) & 0xFFFFFFFF
    return x


def evalimage(net: Yolact, path: str, save_path: str = None):
    device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
    frame = torch.from_numpy(cv2.imread(path)).to(device=device, dtype=torch.float32)
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)
    img_numpy = prep_display(preds, frame, None, None, undo_transform=False)
    if save_path is None:
        img_numpy = img_numpy[:, :, (2, 1, 0)]
        plt.imshow(img_numpy)
        plt.title(path)
        plt.show()
    else:
        cv2.imwrite(save_path, img_numpy)


def evalimages(net: Yolact, input_folder: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    print()
    for p in Path(input_folder).glob('*'):
        if p.is_dir():
            continue
        path = str(p)
        name = os.path.basename(path)
        name = '.'.join(name.split('.')[:-1]) + '.png'
        out_path = os.path.join(output_folder, name)
        evalimage(net, path, out_path)
        print(path + ' -> ' + out_path)
    print('Done.')


# ----------------- COCOeval Size Buckets -----------------
def _eval_size_buckets(ann_file: str, det_file: str, iouType: str = 'segm'):
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(det_file)
    ce = COCOeval(coco_gt, coco_dt, iouType=iouType)

    # 전체 AP (0.5:0.95)
    ce.evaluate(); ce.accumulate(); ce.summarize()
    ap_all = float(ce.stats[0])

    # size buckets
    buckets = {
        'small':  [0, 32**2],
        'medium': [32**2, 96**2],
        'large':  [96**2, 1e10]
    }
    out = {}
    for lbl, rng in buckets.items():
        ce2 = COCOeval(coco_gt, coco_dt, iouType=iouType)
        ce2.params.areaRng = [rng]
        ce2.params.areaRngLbl = [lbl]
        ce2.evaluate(); ce2.accumulate()
        prec = ce2.eval['precision']  # [TxRxKxAxM]
        out[lbl] = float(np.nanmean(prec))
    return ap_all, out['small'], out['medium'], out['large']


# ----------------- Inference Profiler -----------------
class InferProfiler:
    def __init__(self, device='cpu'):
        self.device = device
        self.has_cuda = (device == 'cuda' and torch.cuda.is_available())
        self.proc = psutil.Process() if psutil is not None else None
        self.rows = []  # per-image metrics

    def reset_gpu_peak(self):
        if self.has_cuda:
            torch.cuda.reset_peak_memory_stats()

    def gpu_mem_peak_mb(self):
        if not self.has_cuda:
            return 0.0, 0.0
        alloc = torch.cuda.max_memory_allocated() / (1024**2)
        reserv = torch.cuda.max_memory_reserved() / (1024**2)
        return float(alloc), float(reserv)

    def ram_rss_mb(self):
        if self.proc is None: return 0.0
        return float(self.proc.memory_info().rss) / (1024**2)

    def record(self, image_id, latency_s, ram_mb_before, ram_mb_after):
        gpu_alloc_mb, gpu_res_mb = self.gpu_mem_peak_mb()
        self.rows.append({
            'image_id': image_id,
            'latency_ms': round(latency_s * 1000, 3),
            'gpu_alloc_peak_mb': round(gpu_alloc_mb, 2),
            'gpu_reserved_peak_mb': round(gpu_res_mb, 2),
            'ram_rss_before_mb': round(ram_mb_before, 2),
            'ram_rss_after_mb': round(ram_mb_after, 2),
        })

    def summary(self):
        if not self.rows:
            return {}
        L = [r['latency_ms'] for r in self.rows]
        alloc = [r['gpu_alloc_peak_mb'] for r in self.rows]
        reserv = [r['gpu_reserved_peak_mb'] for r in self.rows]
        return {
            'count': len(self.rows),
            'latency_ms_avg': round(sum(L)/len(L), 3),
            'latency_ms_p50': round(stats.median(L), 3),
            'latency_ms_p95': round(np.percentile(L, 95), 3) if len(L) >= 3 else None,
            'throughput_img_per_s': round(1000.0/(sum(L)/len(L)), 2),
            'gpu_alloc_peak_mb_avg': round(sum(alloc)/len(alloc), 2) if alloc else 0.0,
            'gpu_reserved_peak_mb_avg': round(sum(reserv)/len(reserv), 2) if reserv else 0.0,
        }

    def to_csv(self, path):
        if not self.rows: return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        keys = list(self.rows[0].keys())
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)


# ----------------- Params / FLOPs -----------------
def count_params(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def _parse_hw(s: Optional[str], default: Optional[Tuple[int, int]] = None):
    if not s:
        return default
    if 'x' in s.lower():
        h, w = s.lower().split('x')
        return int(h), int(w)
    v = int(s)
    return (v, v)

def try_measure_flops(model, hw=(550, 550), device='cpu'):
    """
    FLOPs 계산을 thop 또는 fvcore로 시도. 실패 시 None 반환.
    - 모델 forward 만 카운트 (postprocess 제외)
    """
    model_device = next(model.parameters()).device
    H, W = hw
    dummy = torch.randn(1, 3, H, W, device=model_device)

    # 1) thop
    try:
        import thop
        macs, _params = thop.profile(model, inputs=(dummy,), verbose=False)
        return float(macs) / 1e9
    except Exception:
        pass

    # 2) fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        with torch.no_grad():
            flops = FlopCountAnalysis(model, dummy).total()
        return float(flops) / 1e9
    except Exception:
        pass

    return None

def print_model_stats(model, hw=None, device='cpu', stats_csv=None, fps_from_prof=None):
    total = count_params(model, trainable_only=False)
    trainable = count_params(model, trainable_only=True)

    # HxW 결정
    if hw is None:
        side = getattr(cfg, 'max_size', None)
        if side is None:
            side = 550
        hw = (int(side), int(side))

    flops_g = try_measure_flops(model, hw=hw, device=device)

    print('\n[MODEL STATS]')
    print(f' - Params (total): {total:,}  ({total/1e6:.3f} M)')
    print(f' - Params (trainable): {trainable:,}  ({trainable/1e6:.3f} M)')
    if flops_g is not None:
        print(f' - FLOPs: {flops_g:.3f} G  @ input {hw[0]}x{hw[1]}')
    else:
        print(f' - FLOPs: (계산 실패)  @ input {hw[0]}x{hw[1]}  → thop 또는 fvcore 설치 권장')

    if fps_from_prof is not None:
        print(f' - Throughput (eval): {fps_from_prof:.2f} images/sec')

    # CSV 기록 (옵션)
    if stats_csv:
        os.makedirs(os.path.dirname(stats_csv), exist_ok=True)
        row = {
            'config': getattr(cfg, 'name', 'NA'),
            'input_h': hw[0],
            'input_w': hw[1],
            'params_m': round(total/1e6, 6),
            'params_trainable_m': round(trainable/1e6, 6),
            'flops_g': (round(flops_g, 6) if flops_g is not None else None),
            'throughput_img_per_s': (round(float(fps_from_prof), 4) if fps_from_prof is not None else None),
        }
        write_header = not os.path.exists(stats_csv) or os.path.getsize(stats_csv) == 0
        with open(stats_csv, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f'[MODEL STATS] saved CSV → {stats_csv}')


# ----------------- Main Evaluation -----------------
def evaluate(net: Yolact, dataset, train_mode=False):
    """
    Returns:
      maps: dict 형태. 예) {
        'box':  {'AP': ..., 'AP50': ..., 'AP75': ..., ...},
        'mask': {'AP': ..., 'AP50': ..., 'AP75': ..., ...},
        'AJI':  0.73,   # 추가
        '_profile': { ... }  # (선택) 프로파일 요약
      }
    """
    # COCO 카테고리 매핑 준비
    try:
        if not coco_cats:
            prep_coco_cats()
    except NameError:
        prep_coco_cats()

    maps = {}

    device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')

    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    # ===== 수정 1: train_mode일 때 더 가벼운 실행 =====
    if train_mode:
        print("[Val] Running in train_mode (lighter evaluation)", flush=True)
        timer.disable_all()  # 타이머 오버헤드 제거
        # train_mode에서는 프로파일링과 AJI를 비활성화
        original_profile = args.profile_infer
        original_aji = args.compute_aji
        args.profile_infer = False
        args.compute_aji = False

    # 1) 단발 이미지/폴더 모드는 시각화/저장만
    if args.image is not None:
        if ':' in args.image:
            inp, out = args.image.split(':')
            evalimage(net, inp, out)
        else:
            evalimage(net, args.image)
        return maps

    if args.images is not None:
        inp, out = args.images.split(':')
        evalimages(net, inp, out)
        return maps

    # 2) 반복 평가 준비
    frame_times = MovingAverage()
    dataset_size = len(dataset) if args.max_images < 0 else min(args.max_images, len(dataset))
    progress_bar = ProgressBar(30, dataset_size)
    
    if not train_mode:
        print()

    if not args.display and not args.benchmark:
        global ap_data
        ap_data = {
            'box':  [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds],
            'mask': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds],
        }
        detections = Detections()
        aji_acc = AJIAccumulator(thr=args.aji_mask_thresh) if args.compute_aji else None
    else:
        timer.disable('Load Data')
        aji_acc = None

    # 3) 프로파일러
    profiler = InferProfiler(device='cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu') if args.profile_infer else None
    warmup_skip = args.profile_warmup if args.profile_infer else 0

    dataset_indices = list(range(len(dataset)))
    if args.shuffle:
        random.shuffle(dataset_indices)
    elif not args.no_sort:
        hashed = [badhash(x) for x in dataset.ids]
        dataset_indices.sort(key=lambda x: hashed[x])
    dataset_indices = dataset_indices[:dataset_size]

    try:
        for it, image_idx in enumerate(dataset_indices):
            # ===== 수정 2: 진행상황 더 자주 출력 =====
            if train_mode and it % 5 == 0:
                print(f"\r[Val] Processing {it+1}/{dataset_size}...", end='', flush=True)
            
            timer.reset()
            
            # ===== 수정 3: CUDA 캐시 정리 빈도 감소 =====
            if torch.cuda.is_available() and it > 0 and it % 20 == 0:  # 5 → 20으로 변경
                torch.cuda.empty_cache()
            
            with timer.env('Load Data'):
                img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)

                if torch.is_tensor(img):
                    th, tw = int(img.shape[-2]), int(img.shape[-1])
                    if (h is not None and w is not None) and (th != h or tw != w):
                        if it == 0 and not train_mode:  # 첫 이미지만 경고 (train_mode가 아닐 때만)
                            print(f"\n[WARN] eval: orig_hw={h}x{w} but transformed={th}x{tw}")
                        h, w = th, tw

                batch = img.unsqueeze(0).to(device=device, dtype=torch.float32)

            # ===== 수정 4: 프로파일링 간소화 =====
            ram_before = None
            if profiler and not train_mode:  # train_mode에서는 스킵
                ram_before = profiler.ram_rss_mb()
                profiler.reset_gpu_peak()
                if profiler.has_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

            with timer.env('Network Extra'):
                preds = net(batch)

            if profiler and not train_mode and ram_before is not None:
                if profiler.has_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                if it >= warmup_skip:
                    ram_after = profiler.ram_rss_mb()
                    profiler.record(
                        image_id=int(dataset.ids[image_idx]) if hasattr(dataset, 'ids') else it,
                        latency_s=(t1 - t0),
                        ram_mb_before=ram_before,
                        ram_mb_after=ram_after
                    )

            # 후처리/메트릭
            if args.display:
                img_numpy = prep_display(preds, img, h, w)
            elif args.benchmark:
                with timer.env('Postprocess'):
                    postprocess(preds, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)
            else:
                # ===== 수정 5: AJI는 train_mode에서 비활성화 =====
                aji_to_use = None if train_mode else aji_acc
                prep_metrics(ap_data, preds, img, gt, gt_masks, h, w, num_crowd,
                             dataset.ids[image_idx], detections, aji_acc=aji_to_use)

            if it > 1:
                frame_times.add(timer.total_time())

            # 진행 표시
            if args.display:
                if it > 1:
                    print('Avg FPS: %.4f' % (1 / frame_times.get_avg()))
                plt.imshow(img_numpy[:, :, (2, 1, 0)])
                plt.title(str(dataset.ids[image_idx]))
                plt.show()
            elif not args.no_bar and not train_mode:  # train_mode에서는 간소화
                fps = 1 / frame_times.get_avg() if it > 1 and frame_times.get_avg() > 0 else 0
                progress = (it + 1) / dataset_size * 100
                progress_bar.set_val(it + 1)
                print('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                      % (repr(progress_bar), it + 1, dataset_size, progress, fps), end='')
        
        # ===== 수정 6: train_mode 완료 메시지 =====
        if train_mode:
            print(f"\n[Val] Processed {dataset_size} images", flush=True)

        # 4) 결과 수집/반환
        if not args.display and not args.benchmark:
            if not train_mode:
                print()
            if args.output_coco_json:
                if not train_mode:
                    print('Dumping detections...')
                detections.dump()

            # mAP 계산
            maps = calc_map(ap_data)

            # ===== 수정 7: AJI는 명시적으로 계산되었을 때만 =====
            if aji_acc is not None and not train_mode:
                aji_val = aji_acc.value()
                maps['AJI'] = round(aji_val, 4)
                print(f"[AJI] dataset AJI = {aji_val:.4f}")

            # ===== 수정 8: train_mode에서는 size_bucket 스킵 =====
            if not train_mode and args.output_coco_json and args.export_coco and args.coco_ann_file and args.metrics_csv:
                try:
                    ap_all, ap_s, ap_m, ap_l = _eval_size_buckets(args.coco_ann_file, args.export_coco, iouType='segm')
                    row = {
                        'config': cfg.name,
                        'pred_scales': str(getattr(cfg.backbone, 'pred_scales', 'NA')),
                        'AP_all': round(ap_all * 100, 2),
                        'AP_small': round(ap_s * 100, 2),
                        'AP_medium': round(ap_m * 100, 2),
                        'AP_large': round(ap_l * 100, 2),
                        'AJI': (round(aji_acc.value()*100, 2) if aji_acc is not None else None),
                    }
                    os.makedirs(os.path.dirname(args.metrics_csv), exist_ok=True)
                    write_header = not os.path.exists(args.metrics_csv) or os.path.getsize(args.metrics_csv) == 0
                    with open(args.metrics_csv, 'a', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=list(row.keys()))
                        if write_header:
                            w.writeheader()
                        w.writerow(row)
                    print(f"[eval] Wrote metrics to {args.metrics_csv}")
                except Exception as e:
                    print(f"[WARN] metrics_csv write failed: {e}")

        elif args.benchmark:
            print('\n\nStats for the last frame:')
            timer.print_stats()
            avg_seconds = frame_times.get_avg()
            print('Average: %5.2f fps, %5.2f ms' % (1 / avg_seconds if avg_seconds > 0 else 0, 1000 * avg_seconds))
            maps = {}

        # 5) 프로파일 요약
        if isinstance(maps, dict):
            if args.profile_infer and profiler and profiler.rows and dataset_size > 0 and not train_mode:
                summ = profiler.summary()
                print("\n[PROFILE] images=%d | latency_avg=%.3f ms | p50=%.3f | p95=%s | thrpt=%.2f img/s | "
                      "gpu_alloc_peak_avg=%.2f MB | gpu_reserved_peak_avg=%.2f MB" %
                      (summ.get('count', 0),
                       summ.get('latency_ms_avg', 0.0),
                       summ.get('latency_ms_p50', 0.0),
                       '%.3f' % summ['latency_ms_p95'] if summ.get('latency_ms_p95') is not None else 'n/a',
                       summ.get('throughput_img_per_s', 0.0),
                       summ.get('gpu_alloc_peak_mb_avg', 0.0),
                       summ.get('gpu_reserved_peak_mb_avg', 0.0)),
                      flush=True)
                if args.profile_csv:
                    try:
                        profiler.to_csv(args.profile_csv)
                        print(f"[PROFILE] saved CSV → {args.profile_csv}")
                    except Exception as e:
                        print(f"[PROFILE][WARN] failed to write CSV: {e}")
                maps['_profile'] = summ

    except KeyboardInterrupt:
        print('\n[Val] Validation interrupted by user')
        if train_mode:
            # train_mode에서는 상위로 전파하여 학습이 중단되도록
            raise
        # 그 외에는 부분 결과라도 반환

    # ===== train_mode 종료 시 원래 설정 복원 =====
    if train_mode:
        args.profile_infer = original_profile
        args.compute_aji = original_aji

    return maps


def calc_map(ap_data):
    print('Calculating mAP...', flush=True)
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]
    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]
                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold * 100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values()) - 1))

    print_maps(all_maps)
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps


def print_maps(all_maps):
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n: ('-------+' * n)
    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()


# ----------------- Main -----------------
if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    if args.dataset is not None:
        set_dataset(args.dataset)

    os.makedirs('results', exist_ok=True)

    device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
    if args.resume and not args.display:
        with open(args.ap_data_file, 'rb') as f:
            ap_data = pickle.load(f)
        calc_map(ap_data)
        exit()

    # Dataset 생성: 외부 경로 오버라이드 우선
    if args.image is None and args.video is None and args.images is None:
        if args.coco_images_dir is not None and args.coco_ann_file is not None:
            dataset = COCODetection(args.coco_images_dir, args.coco_ann_file,
                                    transform=BaseTransform(), has_gt=True)
            print(f'[INFO] Using dataset override:\n'
                  f'       images_dir = {args.coco_images_dir}\n'
                  f'       ann_file   = {args.coco_ann_file}')
            prep_coco_cats()
        else:
            dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info,
                                    transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
            prep_coco_cats()
    else:
        dataset = None

    print('Loading model...', end='')
    net = Yolact()
    net.load_weights(args.trained_model)
    net.eval()
    net = net.to(device)
    print(' Done.')

    # FLOPs 입력 해상도 결정
    desired_hw = _parse_hw(args.flops_hw, default=None)
    if desired_hw is None:
        side = getattr(cfg, 'max_size', None)
        desired_hw = (int(side), int(side)) if side is not None else (550, 550)

    # FLOPs/Params를 평가 전에 먼저 찍기 (FPS는 없음)
    if args.model_stats and args.flops_use_dummy:
        print_model_stats(net, hw=desired_hw, device=device.type, stats_csv=args.stats_csv, fps_from_prof=None)

    # 실행
    if args.images is not None:
        # 폴더 시각화 모드
        inp, out = args.images.split(':')
        os.makedirs(out, exist_ok=True)
        for p in Path(inp).glob("*"):
            if p.is_dir():
                continue
            evalimage(net, str(p), os.path.join(out, p.stem + ".png"))
        results = {}
    else:
        results = evaluate(net, dataset) if dataset is not None else {}

    # 평가 끝난 후, 프로파일에서 Throughput을 가져와 Stats와 함께 출력/CSV 기록
    if args.model_stats:
        fps = None
        if isinstance(results, dict) and '_profile' in results and 'throughput_img_per_s' in results['_profile']:
            fps = results['_profile']['throughput_img_per_s']
        print_model_stats(net, hw=desired_hw, device=device.type, stats_csv=args.stats_csv, fps_from_prof=fps)