# YOLACT Loss 분석 및 정리 가이드

## 현재 설정에서 실제 사용되는 Loss

### ✅ 활성화된 Loss

1. **B: Box Localization Loss**
   - **활성화 조건**: `cfg.train_boxes = True`
   - **현재 상태**: ✅ 활성화
   - **Loss 함수**: Smooth L1 Loss
   - **가중치**: `cfg.bbox_alpha = 1.5`
   - **용도**: 바운딩 박스 위치 예측 정확도 향상

2. **C: Class Confidence Loss**
   - **활성화 조건**: 항상 활성화 (필수)
   - **현재 상태**: ✅ 활성화
   - **Loss 함수**: 
     - `cfg.use_focal_loss = False` → **OHEM (Online Hard Example Mining)** 사용
     - `cfg.use_focal_loss = True` → Focal Loss 사용
   - **가중치**: `cfg.conf_alpha = 1`
   - **용도**: 클래스 분류 정확도 향상

3. **M: Mask Loss**
   - **활성화 조건**: `cfg.train_masks = True`
   - **현재 상태**: ✅ 활성화
   - **Loss 함수**: 
     - `mask_type = lincomb` → Binary Cross Entropy 또는 Smooth L1
     - `mask_type = direct` → Binary Cross Entropy
   - **가중치**: `cfg.mask_alpha = 6.125` (yolact_base_config 기준)
   - **용도**: 마스크 예측 정확도 향상

4. **S: Semantic Segmentation Loss**
   - **활성화 조건**: `cfg.use_semantic_segmentation_loss = True`
   - **현재 상태**: ✅ 활성화 (yolact_base_config에서 상속)
   - **Loss 함수**: Binary Cross Entropy with Logits
   - **가중치**: `cfg.semantic_segmentation_alpha = 1`
   - **용도**: 픽셀 단위 클래스 예측 (멀티태스크 학습)

### ❌ 비활성화된 Loss

1. **P: Prototype Loss**
   - **활성화 조건**: `cfg.mask_proto_loss is not None`
   - **현재 상태**: ❌ 비활성화 (`mask_proto_loss = None`)
   - **Loss 함수**: L1 또는 Disjoint Loss
   - **용도**: 프로토타입 마스크의 스파스성 유도

2. **D: Coefficient Diversity Loss**
   - **활성화 조건**: `cfg.mask_proto_coeff_diversity_loss = True`
   - **현재 상태**: ❌ 비활성화
   - **Loss 함수**: 코사인 유사도 기반
   - **용도**: 같은 인스턴스는 유사한 계수, 다른 인스턴스는 다른 계수 유도

3. **E: Class Existence Loss**
   - **활성화 조건**: `cfg.use_class_existence_loss = True`
   - **현재 상태**: ❌ 비활성화
   - **Loss 함수**: Binary Cross Entropy with Logits
   - **용도**: 이미지에 존재하는 클래스 예측 (멀티태스크 학습)

4. **I: Mask IoU Loss**
   - **활성화 조건**: `cfg.use_maskiou = True`
   - **현재 상태**: ❌ 비활성화 (`use_maskiou = False`)
   - **Loss 함수**: Smooth L1 Loss
   - **용도**: 마스크 IoU 예측 정확도 향상

## Loss 정리 방법

### 1. 코드 레벨 정리 (권장)

#### Step 1: 사용하지 않는 Loss 함수 제거 또는 주석 처리

`layers/modules/multibox_loss.py`에서:

```python
# 비활성화된 Loss 계산 부분을 조건문으로 명확히 구분
# 또는 완전히 제거 (코드 가독성 향상)

# 예시: Prototype Loss
if cfg.mask_proto_loss is not None:
    if cfg.mask_proto_loss == 'l1':
        losses['P'] = torch.mean(torch.abs(proto_data)) / self.l1_expected_area * self.l1_alpha
    elif cfg.mask_proto_loss == 'disj':
        losses['P'] = -torch.mean(torch.max(F.log_softmax(proto_data, dim=-1), dim=-1)[0])
# else: 이 부분은 실행되지 않으므로 제거 가능
```

#### Step 2: Loss 반환 딕셔너리 정리

```python
# forward 메서드에서 반환하는 losses 딕셔너리를 정리
# 사용하지 않는 Loss 키는 반환하지 않도록 수정

def forward(self, net, predictions, targets, masks, num_crowds):
    losses = {}
    
    # 활성화된 Loss만 계산
    if cfg.train_boxes:
        losses['B'] = ...  # Box Loss
    
    if cfg.train_masks:
        losses['M'] = ...  # Mask Loss
    
    # Confidence Loss는 항상 활성화
    losses['C'] = ...  # Confidence Loss
    
    # 선택적 Loss
    if cfg.use_semantic_segmentation_loss:
        losses['S'] = ...  # Semantic Loss
    
    if cfg.use_maskiou:
        losses['I'] = ...  # MaskIoU Loss
    
    # 사용하지 않는 Loss는 계산하지 않음
    # - P (Prototype): mask_proto_loss가 None이면 계산 안 함
    # - D (Diversity): mask_proto_coeff_diversity_loss가 False면 계산 안 함
    # - E (Existence): use_class_existence_loss가 False면 계산 안 함
    
    return losses
```

### 2. 설정 레벨 정리

#### `data/config.py`에서 명시적으로 비활성화

```python
cell_yolact_im700_config = yolact_im700_config.copy({
    'name': 'cell_yolact_im700',
    
    # ... 기존 설정 ...
    
    # Loss 관련 명시적 설정
    'train_boxes': True,                    # Box Loss 활성화
    'train_masks': True,                    # Mask Loss 활성화
    
    # 비활성화할 Loss 명시
    'use_focal_loss': False,                # OHEM 사용 (Focal Loss 비활성화)
    'use_maskiou': False,                   # MaskIoU Loss 비활성화
    'use_class_existence_loss': False,       # Class Existence Loss 비활성화
    'use_semantic_segmentation_loss': False, # Semantic Loss 비활성화 (필요시)
    'mask_proto_loss': None,                 # Prototype Loss 비활성화
    'mask_proto_coeff_diversity_loss': False, # Diversity Loss 비활성화
    
    # Loss 가중치 조정
    'conf_alpha': 1,
    'bbox_alpha': 1.5,
    'mask_alpha': 6.125,
})
```

### 3. 로깅 레벨 정리

#### `train.py`에서 Loss 로깅 정리

```python
# train.py의 loss_types 리스트를 실제 사용되는 Loss만 포함하도록 수정

# 현재: loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']
# 수정: 실제 활성화된 Loss만 포함

active_loss_types = []
if cfg.train_boxes:
    active_loss_types.append('B')
active_loss_types.append('C')  # 항상 활성화
if cfg.train_masks:
    active_loss_types.append('M')
if cfg.use_semantic_segmentation_loss:
    active_loss_types.append('S')
if cfg.use_maskiou:
    active_loss_types.append('I')
# P, D, E는 조건에 따라 추가

loss_types = active_loss_types
```

### 4. 검증 스크립트 작성

#### Loss 활성화 상태 확인 스크립트

```python
# scripts/check_active_losses.py

from data import cfg

def check_active_losses():
    """현재 설정에서 활성화된 Loss 확인"""
    active = []
    inactive = []
    
    # Box Loss
    if cfg.train_boxes:
        active.append('B: Box Localization Loss')
    else:
        inactive.append('B: Box Localization Loss')
    
    # Confidence Loss (항상 활성화)
    active.append('C: Class Confidence Loss')
    
    # Mask Loss
    if cfg.train_masks:
        active.append('M: Mask Loss')
    else:
        inactive.append('M: Mask Loss')
    
    # Prototype Loss
    if cfg.mask_proto_loss is not None:
        active.append('P: Prototype Loss')
    else:
        inactive.append('P: Prototype Loss')
    
    # Diversity Loss
    if cfg.mask_proto_coeff_diversity_loss:
        active.append('D: Coefficient Diversity Loss')
    else:
        inactive.append('D: Coefficient Diversity Loss')
    
    # Class Existence Loss
    if cfg.use_class_existence_loss:
        active.append('E: Class Existence Loss')
    else:
        inactive.append('E: Class Existence Loss')
    
    # Semantic Segmentation Loss
    if cfg.use_semantic_segmentation_loss:
        active.append('S: Semantic Segmentation Loss')
    else:
        inactive.append('S: Semantic Segmentation Loss')
    
    # MaskIoU Loss
    if cfg.use_maskiou:
        active.append('I: Mask IoU Loss')
    else:
        inactive.append('I: Mask IoU Loss')
    
    print("=" * 60)
    print("활성화된 Loss:")
    print("=" * 60)
    for loss in active:
        print(f"  ✅ {loss}")
    
    print("\n" + "=" * 60)
    print("비활성화된 Loss:")
    print("=" * 60)
    for loss in inactive:
        print(f"  ❌ {loss}")
    
    print("\n" + "=" * 60)
    print(f"총 {len(active)}개 Loss 활성화, {len(inactive)}개 Loss 비활성화")
    print("=" * 60)

if __name__ == '__main__':
    check_active_losses()
```

## 현재 설정 요약

### 활성화된 Loss (4개)
- ✅ **B**: Box Localization Loss (Smooth L1, α=1.5)
- ✅ **C**: Class Confidence Loss (OHEM, α=1)
- ✅ **M**: Mask Loss (BCE/Smooth L1, α=6.125)
- ✅ **S**: Semantic Segmentation Loss (BCE with Logits, α=1)

### 비활성화된 Loss (4개)
- ❌ **P**: Prototype Loss
- ❌ **D**: Coefficient Diversity Loss
- ❌ **E**: Class Existence Loss
- ❌ **I**: Mask IoU Loss

## 권장 사항

1. **코드 정리**: 사용하지 않는 Loss 계산 코드는 주석 처리하거나 제거
2. **설정 명시**: config.py에서 모든 Loss 관련 설정을 명시적으로 지정
3. **로깅 최적화**: 실제 사용되는 Loss만 로깅하여 가독성 향상
4. **검증 스크립트**: 학습 전 활성화된 Loss 확인 스크립트 실행

## Loss 가중치 튜닝 가이드

현재 활성화된 Loss의 가중치:
- `bbox_alpha = 1.5`: Box Loss 가중치
- `conf_alpha = 1`: Confidence Loss 가중치
- `mask_alpha = 6.125`: Mask Loss 가중치
- `semantic_segmentation_alpha = 1`: Semantic Loss 가중치

튜닝 시 고려사항:
- Box와 Mask Loss의 균형 조정
- Semantic Loss는 멀티태스크 학습용이므로 필요시 비활성화 가능
- 각 Loss의 스케일이 다르므로 가중치 조정 시 주의

