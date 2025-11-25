#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLACT Loss 활성화 상태 확인 스크립트

사용법:
    python scripts/check_active_losses.py [--config CONFIG_NAME]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data import cfg, set_cfg
import argparse

def check_active_losses():
    """현재 설정에서 활성화된 Loss 확인"""
    active = []
    inactive = []
    
    # Box Loss
    if cfg.train_boxes:
        active.append(('B', 'Box Localization Loss', 'Smooth L1', cfg.bbox_alpha))
    else:
        inactive.append(('B', 'Box Localization Loss', 'Smooth L1', None))
    
    # Confidence Loss (항상 활성화)
    if cfg.use_focal_loss:
        if cfg.use_sigmoid_focal_loss:
            loss_type = 'Focal Loss (Sigmoid)'
        elif cfg.use_objectness_score:
            loss_type = 'Focal Loss (Objectness)'
        else:
            loss_type = 'Focal Loss (Softmax)'
    else:
        if cfg.use_objectness_score:
            loss_type = 'Objectness Loss'
        else:
            loss_type = 'OHEM (Cross Entropy)'
    active.append(('C', 'Class Confidence Loss', loss_type, cfg.conf_alpha))
    
    # Mask Loss
    if cfg.train_masks:
        if cfg.mask_type == 0:  # direct
            mask_loss_type = 'Binary Cross Entropy'
        else:  # lincomb
            mask_loss_type = 'BCE/Smooth L1 (LinComb)'
        active.append(('M', 'Mask Loss', mask_loss_type, cfg.mask_alpha))
    else:
        inactive.append(('M', 'Mask Loss', 'BCE/Smooth L1', None))
    
    # Prototype Loss
    if cfg.mask_proto_loss is not None:
        proto_loss_type = f'Prototype Loss ({cfg.mask_proto_loss})'
        active.append(('P', 'Prototype Loss', proto_loss_type, None))
    else:
        inactive.append(('P', 'Prototype Loss', 'L1/Disjoint', None))
    
    # Diversity Loss
    if cfg.mask_proto_coeff_diversity_loss:
        active.append(('D', 'Coefficient Diversity Loss', 'Cosine Similarity', 
                      cfg.mask_proto_coeff_diversity_alpha))
    else:
        inactive.append(('D', 'Coefficient Diversity Loss', 'Cosine Similarity', None))
    
    # Class Existence Loss
    if cfg.use_class_existence_loss:
        active.append(('E', 'Class Existence Loss', 'BCE with Logits', 
                      cfg.class_existence_alpha))
    else:
        inactive.append(('E', 'Class Existence Loss', 'BCE with Logits', None))
    
    # Semantic Segmentation Loss
    if cfg.use_semantic_segmentation_loss:
        active.append(('S', 'Semantic Segmentation Loss', 'BCE with Logits', 
                      cfg.semantic_segmentation_alpha))
    else:
        inactive.append(('S', 'Semantic Segmentation Loss', 'BCE with Logits', None))
    
    # MaskIoU Loss
    if cfg.use_maskiou:
        active.append(('I', 'Mask IoU Loss', 'Smooth L1', cfg.maskiou_alpha))
    else:
        inactive.append(('I', 'Mask IoU Loss', 'Smooth L1', None))
    
    # 출력
    print("=" * 80)
    print(f"Config: {cfg.name}")
    print("=" * 80)
    print("\n✅ 활성화된 Loss:")
    print("-" * 80)
    for key, name, loss_type, alpha in active:
        alpha_str = f"α={alpha}" if alpha is not None else ""
        print(f"  [{key}] {name:30s} | {loss_type:25s} {alpha_str}")
    
    print("\n❌ 비활성화된 Loss:")
    print("-" * 80)
    for key, name, loss_type, alpha in inactive:
        print(f"  [{key}] {name:30s} | {loss_type:25s}")
    
    print("\n" + "=" * 80)
    print(f"총 {len(active)}개 Loss 활성화, {len(inactive)}개 Loss 비활성화")
    print("=" * 80)
    
    # 총 Loss 가중치 합계
    total_alpha = sum([alpha for _, _, _, alpha in active if alpha is not None])
    print(f"\n활성화된 Loss 가중치 합계: {total_alpha:.3f}")
    
    return active, inactive

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check active losses in YOLACT config')
    parser.add_argument('--config', type=str, default=None,
                       help='Config name to use (e.g., cell_yolact_im700_config)')
    args = parser.parse_args()
    
    if args.config:
        set_cfg(args.config)
    
    check_active_losses()

