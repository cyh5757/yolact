import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from math import sqrt
import albumentations as A

from data import cfg, MEANS, STD


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, masks=None, boxes=None, labels=None):
        for t in self.transforms:
            img, masks, boxes, labels = t(img, masks, boxes, labels)
        return img, masks, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, masks=None, boxes=None, labels=None):
        return self.lambd(img, masks, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, masks=None, boxes=None, labels=None):
        return image.astype(np.float32), masks, boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, masks=None, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, masks, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, masks=None, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, masks, boxes, labels


class Pad(object):
    """
    Pads the image to the input width and height, filling the
    background with mean and putting the image in the top-left.

    Note: this expects im_w <= width and im_h <= height
    """
    def __init__(self, width, height, mean=MEANS, pad_gt=True):
        self.mean = mean
        self.width = width
        self.height = height
        self.pad_gt = pad_gt

    def __call__(self, image, masks, boxes=None, labels=None):
        im_h, im_w, depth = image.shape

        expand_image = np.zeros(
            (self.height, self.width, depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[:im_h, :im_w] = image

        if self.pad_gt:
            expand_masks = np.zeros(
                (masks.shape[0], self.height, self.width),
                dtype=masks.dtype)
            expand_masks[:, :im_h, :im_w] = masks
            masks = expand_masks

        return expand_image, masks, boxes, labels


class Resize(object):
    """If preserve_aspect_ratio is true, this resizes to an approximate area of max_size * max_size"""

    @staticmethod
    def calc_size_preserve_ar(img_w, img_h, max_size):
        """Resulting width*height ≈ max_size^2 while maintaining AR."""
        ratio = sqrt(img_w / img_h)
        w = max_size * ratio
        h = max_size / ratio
        return int(w), int(h)

    def __init__(self, resize_gt=True):
        self.resize_gt = resize_gt
        self.max_size = cfg.max_size
        self.preserve_aspect_ratio = cfg.preserve_aspect_ratio

    def __call__(self, image, masks, boxes, labels=None):
        img_h, img_w, _ = image.shape

        if self.preserve_aspect_ratio:
            width, height = Resize.calc_size_preserve_ar(img_w, img_h, self.max_size)
        else:
            width, height = self.max_size, self.max_size

        # 이미지 리사이즈 (보간은 기본/선형)
        image = cv2.resize(image, (width, height))

        if self.resize_gt:
            # ----- 마스크 리사이즈 -----
            # (N, H, W) -> (H, W, N)로 바꿔서 OpenCV에 넣기
            masks = masks.transpose((1, 2, 0))  # (H, W, N_objs)

            cv_limit = 512
            if masks.shape[2] <= cv_limit:
                resized = cv2.resize(
                    np.ascontiguousarray(masks),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )
                if resized.ndim == 2:  # 단일 채널이면 (H,W)로 나옴
                    resized = resized[:, :, None]
                masks = resized
            else:
                chunks = []
                for i in range(0, masks.shape[2], cv_limit):
                    chunk = np.ascontiguousarray(masks[:, :, i:i + cv_limit])
                    res = cv2.resize(
                        chunk,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST
                    )
                    if res.ndim == 2:
                        res = res[:, :, None]
                    chunks.append(res)
                masks = np.concatenate(chunks, axis=2)

            # (H, W, N) -> (N, H, W)
            masks = masks.transpose((2, 0, 1))

            # ----- 박스 스케일 조정 (절대좌표인 상태) -----
            boxes[:, [0, 2]] *= (width / img_w)
            boxes[:, [1, 3]] *= (height / img_h)

        # ----- 너무 작은 박스 제거 -----
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        scale = cfg.max_size / 1024.0  # 원래 1024 기준이었다면
        min_w = cfg.discard_box_width * scale
        min_h = cfg.discard_box_height * scale
        keep = (w > min_w) & (h > min_h)
        masks = masks[keep]
        boxes = boxes[keep]

        if isinstance(labels, dict) and 'labels' in labels:
            labels['labels'] = labels['labels'][keep]
            labels['num_crowds'] = int((labels['labels'] < 0).sum())

        return image, masks, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, masks, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, masks, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, masks=None, boxes=None, labels=None):
        # 채널 셔플은 사용 안 함
        return image, masks, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, masks, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, masks=None, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, masks, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, masks, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, masks=None, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), masks, boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, masks=None, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), masks, boxes, labels


class RandomSampleCrop(object):
    """
    SSD-style random sample crop
    """
    def __init__(self):
        self.sample_options = (
            None,
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        )

    def __call__(self, image, masks, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            idx = np.random.randint(0, len(self.sample_options))
            mode = self.sample_options[idx]
            if mode is None:
                return image, masks, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                overlap = jaccard_numpy(boxes, rect)

                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                mask = m1 * m2

                num_crowds = labels['num_crowds']
                crowd_mask = np.zeros(mask.shape, dtype=np.int32)

                if num_crowds > 0:
                    crowd_mask[-num_crowds:] = 1

                if not mask.any() or np.sum(1 - crowd_mask[mask]) == 0:
                    continue

                current_masks = masks[mask, :, :].copy()
                current_boxes = boxes[mask, :].copy()

                labels['labels'] = labels['labels'][mask]
                current_labels = labels

                if num_crowds > 0:
                    labels['num_crowds'] = np.sum(crowd_mask[mask])

                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[:2]

                current_masks = current_masks[:, rect[1]:rect[3], rect[0]:rect[2]]

                return current_image, current_masks, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, masks, boxes, labels):
        if random.randint(2):
            return image, masks, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        expand_masks = np.zeros(
            (masks.shape[0], int(height * ratio), int(width * ratio)),
            dtype=masks.dtype)
        expand_masks[:, int(top):int(top + height),
                     int(left):int(left + width)] = masks
        masks = expand_masks

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, masks, boxes, labels


class RandomMirror(object):
    def __call__(self, image, masks, boxes, labels):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            masks = masks[:, :, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, masks, boxes, labels


class RandomFlip(object):
    def __call__(self, image, masks, boxes, labels):
        height, _, _ = image.shape
        if random.randint(2):
            image = image[::-1, :]
            masks = masks[:, ::-1, :]
            boxes = boxes.copy()
            boxes[:, 1::2] = height - boxes[:, 3::-2]
        return image, masks, boxes, labels


class RandomRot90(object):
    def __call__(self, image, masks, boxes, labels):
        old_height, old_width, _ = image.shape
        k = random.randint(4)
        image = np.rot90(image, k)
        masks = np.array([np.rot90(mask, k) for mask in masks])
        boxes = boxes.copy()
        for _ in range(k):
            boxes = np.array([[box[1], old_width - 1 - box[2], box[3], old_width - 1 - box[0]] for box in boxes])
            old_width, old_height = old_height, old_width
        return image, masks, boxes, labels



class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, masks, boxes, labels):
        im = image.copy()
        im, masks, boxes, labels = self.rand_brightness(im, masks, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, masks, boxes, labels = distort(im, masks, boxes, labels)
        return self.rand_light_noise(im, masks, boxes, labels)


class PrepareMasks(object):
    """
    Prepares the gt masks for use_gt_bboxes by cropping with the gt box
    and downsampling the resulting mask to mask_size, mask_size.
    """

    def __init__(self, mask_size, use_gt_bboxes):
        self.mask_size = mask_size
        self.use_gt_bboxes = use_gt_bboxes

    def __call__(self, image, masks, boxes, labels=None):
        if not self.use_gt_bboxes:
            return image, masks, boxes, labels

        height, width, _ = image.shape

        new_masks = np.zeros((masks.shape[0], self.mask_size ** 2))

        for i in range(len(masks)):
            x1, y1, x2, y2 = boxes[i, :]
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height
            x1, y1, x2, y2 = (int(x1), int(y1), int(x2), int(y2))

            cropped_mask = masks[i, y1:(y2 + 1), x1:(x2 + 1)]
            scaled_mask = cv2.resize(cropped_mask, (self.mask_size, self.mask_size))

            new_masks[i, :] = scaled_mask.reshape(1, -1)

        new_masks[new_masks > 0.5] = 1
        new_masks[new_masks <= 0.5] = 0

        return image, new_masks, boxes, labels


class BackboneTransform(object):
    """
    Transforms a BGR image made of floats in the range [0, 255] to whatever
    input the current backbone network needs.
    """
    def __init__(self, transform, mean, std, in_channel_order):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.transform = transform

        self.channel_map = {c: idx for idx, c in enumerate(in_channel_order)}
        self.channel_permutation = [self.channel_map[c] for c in transform.channel_order]

    def __call__(self, img, masks=None, boxes=None, labels=None):
        img = img.astype(np.float32)

        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255

        img = img[:, :, self.channel_permutation]

        return img.astype(np.float32), masks, boxes, labels


class BaseTransform(object):
    """ Transform to be used when evaluating. """

    def __init__(self, mean=MEANS, std=STD):
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            Resize(resize_gt=True),
            ToPercentCoords(),
            PrepareMasks(cfg.mask_size, cfg.use_gt_bboxes),
            BackboneTransform(cfg.backbone.transform, mean, std, 'BGR')
        ])

    def __call__(self, img, masks=None, boxes=None, labels=None):
        return self.augment(img, masks, boxes, labels)


import torch.nn.functional as F


class FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for super speed.
    """

    def __init__(self):
        super().__init__()

        self.mean = torch.Tensor(MEANS).float()[None, :, None, None]
        self.std = torch.Tensor(STD).float()[None, :, None, None]
        self.transform = cfg.backbone.transform

    def forward(self, img):
        self.mean = self.mean.to(img.device)
        self.std = self.std.to(img.device)

        if cfg.preserve_aspect_ratio:
            _, h, w, _ = img.size()
            img_size = Resize.calc_size_preserve_ar(w, h, cfg.max_size)
            img_size = (img_size[1], img_size[0])
        else:
            img_size = (cfg.max_size, cfg.max_size)

        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, img_size, mode='bilinear', align_corners=False)

        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255

        if self.transform.channel_order != 'RGB':
            raise NotImplementedError

        img = img[:, (2, 1, 0), :, :].contiguous()
        return img


def do_nothing(img=None, masks=None, boxes=None, labels=None):
    return img, masks, boxes, labels


def enable_if(condition, obj):
    return obj if condition else do_nothing


class SSDAugmentation(object):
    """ Transform to be used when training. """

    def __init__(self, mean=MEANS, std=STD):
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            # enable_if(cfg.augment_photometric_distort, PhotometricDistort()),
            # enable_if(cfg.augment_expand, Expand(mean)),
            # enable_if(cfg.augment_random_sample_crop, RandomSampleCrop()),
            # enable_if(cfg.augment_random_mirror, RandomMirror()),
            # enable_if(cfg.augment_random_flip, RandomFlip()),
            # enable_if(cfg.augment_random_flip, RandomRot90()),
            Resize(),
            # enable_if(not cfg.preserve_aspect_ratio, Pad(cfg.max_size, cfg.max_size, mean)),
            ToPercentCoords(),
            PrepareMasks(cfg.mask_size, cfg.use_gt_bboxes),
            BackboneTransform(cfg.backbone.transform, mean, std, 'BGR')
        ])

    def __call__(self, img, masks, boxes, labels):
        return self.augment(img, masks, boxes, labels)


class AlbumentationsImageAugment(object):
    def __init__(self, policy="medium", p=1.0):
        """
        Args:
            policy (str): 'base', 'light', 'medium', 'heavy' 중 선택
            p (float): 전체 파이프라인이 적용될 확률
        """
        self.policy = policy
        self.p = p
        self.transform = self.get_transforms(policy)

    def get_transforms(self, policy):
        transforms_list = []

        # -----------------------------
        # 공통: 픽셀 레벨 변환 정의
        # -----------------------------
        pixel_transforms = [
            A.RandomBrightnessContrast(
                brightness_limit=0.5,
                contrast_limit=0.4,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
            A.RGBShift(
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=1.0
            ),
        ]

        # ----------------------------------------------------------
        # 1. Base: 아무것도 안 함 (Albumentations 파이프라인만 통과)
        # ----------------------------------------------------------
        if policy == 'base':
            return A.Compose(
                [],
                bbox_params=self.get_bbox_params(),
                p=self.p
            )

        # ----------------------------------------------------------
        # 2. Light: 픽셀 레벨 변환만 적용 (기하학 X)
        # ----------------------------------------------------------
        if policy == 'light':
            transforms_list.extend(pixel_transforms)

        # ----------------------------------------------------------
        # 3. Medium: Light + 안전한 기하학적 변환
        #    - Flip + 약한 Shift/Scale/Rotate
        # ----------------------------------------------------------
        elif policy == 'medium':
            transforms_list.extend([
                # 좌우/상하 플립
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),

                # 크롭 대신 Shift/Scale/Rotate로 기하학적 다양성 확보
                A.ShiftScaleRotate(
                    shift_limit=0.0625,   # ±6.25% 이동
                    scale_limit=0.2,      # ±20% 확대/축소
                    rotate_limit=15,      # ±15도 회전
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,              # 빈 공간을 검은색으로 채움 (형광 이미지에 적합)
                    p=0.5
                ),

                # 픽셀 변환은 OneOf로 적당히 랜덤하게
                A.OneOf(pixel_transforms, p=0.8),
            ])

        # ----------------------------------------------------------
        # 4. Heavy: Medium + 강한 기하학/노이즈/왜곡
        # ----------------------------------------------------------
        elif policy == 'heavy':
            transforms_list.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),

                # 더 강한 Shift/Scale/Rotate
                A.ShiftScaleRotate(
                    shift_limit=0.1,      # ±10% 이동
                    scale_limit=0.3,      # ±30% 확대/축소
                    rotate_limit=30,      # ±30도 회전
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.7
                ),

                # 픽셀 변환 강화
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.6,
                        contrast_limit=0.5,
                        p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=25,
                        sat_shift_limit=35,
                        val_shift_limit=25,
                        p=1.0
                    ),
                    A.RGBShift(
                        r_shift_limit=30,
                        g_shift_limit=30,
                        b_shift_limit=30,
                        p=1.0
                    ),
                ], p=0.8),

                # 이미지 품질 저하 (노이즈, 블러)
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ], p=0.3),

                # 공간 왜곡 (너무 세게 쓰면 박스/마스크가 찌그러지므로 p 낮게)
                A.OneOf([
                    A.OpticalDistortion(distort_limit=0.3, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                ], p=0.1),
            ])

        # safety: 혹시라도 transforms_list가 비어 있으면 no-op로
        return A.Compose(
            transforms_list,
            bbox_params=self.get_bbox_params(),
            p=self.p
        )

    def get_bbox_params(self):
        return A.BboxParams(
            format="pascal_voc",
            min_visibility=0.1,  # 박스 영역 10% 이상 살아있어야 유지
            label_fields=["bbox_labels"]
        )

    def __call__(self, image, masks, boxes, labels):
        # 박스 없으면 그냥 패스
        if boxes is None or len(boxes) == 0:
            return image, masks, boxes, labels

        orig_image, orig_masks, orig_boxes, orig_labels = image, masks, boxes, labels

        # masks: (N, H, W) -> list로 변환
        mask_list = [masks[i] for i in range(masks.shape[0])] if masks is not None else []

        # labels 처리
        if isinstance(labels, dict) and "labels" in labels:
            bbox_labels = labels["labels"]
        else:
            bbox_labels = np.ones((boxes.shape[0],), dtype=np.int32)

        bboxes = boxes.astype(np.float32).tolist()

        try:
            transformed = self.transform(
                image=image,
                masks=mask_list,
                bboxes=bboxes,
                bbox_labels=bbox_labels,
            )
        except ValueError as e:
            print(f"[Aug Fail] {e}")
            return orig_image, orig_masks, orig_boxes, orig_labels

        out_img = transformed["image"]
        new_bboxes = np.array(transformed["bboxes"], dtype=np.float32)
        new_labels = np.array(transformed["bbox_labels"], dtype=bbox_labels.dtype)
        new_masks_list = transformed["masks"]

        # -------- 정합성 검사 --------
        if len(new_bboxes) == 0:
            return orig_image, orig_masks, orig_boxes, orig_labels

        if len(new_masks_list) != len(new_bboxes):
            return orig_image, orig_masks, orig_boxes, orig_labels

        if len(new_masks_list) > 0:
            masks_arr = np.stack(new_masks_list, axis=0).astype(np.float32)

            # 너무 작아진 마스크 제거
            areas = masks_arr.reshape(masks_arr.shape[0], -1).sum(axis=1)
            keep = areas > 5

            if keep.sum() == 0:
                return orig_image, orig_masks, orig_boxes, orig_labels

            masks_arr = masks_arr[keep]
            new_bboxes = new_bboxes[keep]
            new_labels = new_labels[keep]

            masks_arr = (masks_arr > 0.5).astype(np.float32)
            out_masks = masks_arr
        else:
            out_masks = masks

        # labels dict 업데이트
        if isinstance(labels, dict):
            labels_out = labels.copy()
            labels_out["labels"] = new_labels
            labels_out["num_crowds"] = int((labels_out["labels"] < 0).sum())
        else:
            labels_out = new_labels  # 여기서도 new_labels로 바꿔주는 게 깔끔

        return out_img, out_masks, new_bboxes, labels_out


class SSD_ALBU_Augmentation(object):
    """ 
    Ablation Study를 위한 메인 Augmentation 클래스 
    """
    def __init__(self, mean=MEANS, std=STD, policy="medium"):
        """
        policy: 'base', 'light', 'medium', 'heavy'
        """
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            
            # 재정립된 Albumentations 모듈
            AlbumentationsImageAugment(policy=policy, p=1.0),
            
            # 이후 기존 파이프라인 (Resize -> 포맷 변경)
            Resize(),
            ToPercentCoords(),
            PrepareMasks(cfg.mask_size, cfg.use_gt_bboxes),
            BackboneTransform(cfg.backbone.transform, mean, std, 'BGR'),
        ])

    def __call__(self, img, masks, boxes, labels):
        return self.augment(img, masks, boxes, labels)