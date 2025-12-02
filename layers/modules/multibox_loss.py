# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box_utils import match, log_sum_exp, decode, center_size, crop, elemwise_mask_iou, elemwise_box_iou

from data import cfg, mask_type, activation_func


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    """

    def __init__(self, num_classes, pos_threshold, neg_threshold, negpos_ratio):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes

        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.negpos_ratio = negpos_ratio

        # If you output a proto mask with this area, your l1 loss will be l1_alpha
        # Note that the area is relative (so 1 would be the entire image)
        self.l1_expected_area = 20 * 20 / 70 / 70
        self.l1_alpha = 0.1

        if cfg.use_class_balanced_conf:
            self.class_instances = None
            self.total_instances = 0

    def forward(self, net, predictions, targets, masks, num_crowds):
        """Multibox Loss
        """

        loc_data = predictions["loc"]
        conf_data = predictions["conf"]
        mask_data = predictions["mask"]
        priors = predictions["priors"]

        if cfg.mask_type == mask_type.lincomb:
            proto_data = predictions["proto"]

        score_data = predictions["score"] if cfg.use_mask_scoring else None
        inst_data = predictions["inst"] if cfg.use_instance_coeff else None

        labels = [None] * len(targets)  # Used in sem segm loss

        batch_size = loc_data.size(0)
        num_priors = priors.size(0)
        num_classes = self.num_classes

        # Match priors (default boxes) and ground truth boxes
        # These tensors will be created with the same device as loc_data
        loc_t = loc_data.new(batch_size, num_priors, 4)
        gt_box_t = loc_data.new(batch_size, num_priors, 4)  # matched gt boxes (point-form)
        conf_t = loc_data.new(batch_size, num_priors).long()
        idx_t = loc_data.new(batch_size, num_priors).long()

        if cfg.use_class_existence_loss:
            class_existence_t = loc_data.new(batch_size, num_classes - 1)

        maskiou_targets = None  # for mask IoU loss

        for idx in range(batch_size):
            truths = targets[idx][:, :-1].data
            labels[idx] = targets[idx][:, -1].data.long()

            if cfg.use_class_existence_loss:
                # Construct a one-hot vector for each object and collapse it into an existence vector with max
                # Also it's fine to include the crowd annotations here
                class_existence_t[idx, :] = torch.eye(
                    num_classes - 1, device=conf_t.get_device()
                )[labels[idx]].max(dim=0)[0]

            # Split the crowd annotations because they come bundled in
            cur_crowds = num_crowds[idx]
            if cur_crowds > 0:
                split = lambda x: (x[-cur_crowds:], x[:-cur_crowds])
                crowd_boxes, truths = split(truths)

                # We don't use the crowd labels or masks
                _, labels[idx] = split(labels[idx])
                _, masks[idx] = split(masks[idx])
            else:
                crowd_boxes = None

            match(
                self.pos_threshold,
                self.neg_threshold,
                truths,
                priors.data,
                labels[idx],
                crowd_boxes,
                loc_t,
                conf_t,
                idx_t,
                idx,
                loc_data[idx],
            )

            # truths 는 [num_objs, 4], idx_t 가 matching 된 인덱스
            gt_box_t[idx, :, :] = truths[idx_t[idx]]

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        idx_t = Variable(idx_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)

        losses = {}

        # --------------------------------------------------
        # B: Localization Loss (GIoU-based + size-aware weight)
        # --------------------------------------------------
        if cfg.train_boxes:
            # positive offsets (encoded)
            loc_p = loc_data[pos_idx].view(-1, 4)  # [N_pos, 4] offsets
            loc_t_pos = loc_t[pos_idx].view(-1, 4)  # [N_pos, 4] encoded gt offsets (for conf_objectness_loss)

            # 실제 GT boxes (point-form, 0~1 normalized)
            pos_gt_boxes = gt_box_t[pos_idx].view(-1, 4)  # [N_pos, 4]

            # 해당 positive 에 대응하는 prior boxes
            priors_batch = priors.unsqueeze(0).expand(batch_size, num_priors, 4)
            pos_priors = priors_batch[pos_idx].view(-1, 4)  # [N_pos, 4]

            # decode offsets → predicted boxes (point-form)
            boxes_pred = decode(loc_p, pos_priors, cfg.use_yolo_regressors)  # [N_pos, 4]

            # size-aware weight
            with torch.no_grad():
                size_w_b = self._get_size_weights(pos_gt_boxes)  # [N_pos]

            # GIoU loss (1 - GIoU)
            bbox_loss_raw = self.giou_loss(boxes_pred, pos_gt_boxes)  # [N_pos]

            losses["B"] = (bbox_loss_raw * size_w_b).sum() * cfg.bbox_alpha

            # 이후 conf_objectness_loss 에서 쓸 수 있도록 encoded gt offsets 넘겨주기
            loc_t_encoded = loc_t_pos
        else:
            loc_p = None
            loc_t_encoded = None

        # --------------------
        # M: Mask Loss
        # --------------------
        if cfg.train_masks:
            if cfg.mask_type == mask_type.direct:
                if cfg.use_gt_bboxes:
                    pos_masks = []
                    for idx in range(batch_size):
                        pos_masks.append(masks[idx][idx_t[idx, pos[idx]]])
                    masks_t = torch.cat(pos_masks, 0)
                    masks_p = mask_data[pos, :].view(-1, cfg.mask_dim)
                    losses["M"] = (
                        F.binary_cross_entropy(
                            torch.clamp(masks_p, 0, 1), masks_t, reduction="sum"
                        )
                        * cfg.mask_alpha
                    )
                else:
                    losses["M"] = self.direct_mask_loss(
                        pos_idx, idx_t, loc_data, mask_data, priors, masks
                    )
            elif cfg.mask_type == mask_type.lincomb:
                ret = self.lincomb_mask_loss(
                    pos,
                    idx_t,
                    loc_data,
                    mask_data,
                    priors,
                    proto_data,
                    masks,
                    gt_box_t,
                    score_data,
                    inst_data,
                    labels,
                )
                if cfg.use_maskiou:
                    loss, maskiou_targets = ret
                else:
                    loss = ret
                losses.update(loss)

                if cfg.mask_proto_loss is not None:
                    if cfg.mask_proto_loss == "l1":
                        losses["P"] = (
                            torch.mean(torch.abs(proto_data))
                            / self.l1_expected_area
                            * self.l1_alpha
                        )
                    elif cfg.mask_proto_loss == "disj":
                        losses["P"] = -torch.mean(
                            torch.max(F.log_softmax(proto_data, dim=-1), dim=-1)[0]
                        )

        # Confidence loss
        if cfg.use_focal_loss:
            if cfg.use_sigmoid_focal_loss:
                losses["C"] = self.focal_conf_sigmoid_loss(conf_data, conf_t)
            elif cfg.use_objectness_score:
                losses["C"] = self.focal_conf_objectness_loss(conf_data, conf_t)
            else:
                losses["C"] = self.focal_conf_loss(conf_data, conf_t)
        else:
            if cfg.use_objectness_score:
                losses["C"] = self.conf_objectness_loss(
                    conf_data,
                    conf_t,
                    batch_size,
                    loc_p,
                    loc_t_encoded,
                    priors,
                )
            else:
                losses["C"] = self.ohem_conf_loss(conf_data, conf_t, pos, batch_size)

        # Mask IoU Loss
        if cfg.use_maskiou and maskiou_targets is not None:
            losses["I"] = self.mask_iou_loss(net, maskiou_targets)

        # These losses also don't depend on anchors
        if cfg.use_class_existence_loss:
            losses["E"] = self.class_existence_loss(
                predictions["classes"], class_existence_t
            )
        if cfg.use_semantic_segmentation_loss:
            losses["S"] = self.semantic_segmentation_loss(
                predictions["segm"], masks, labels
            )

        # Divide all losses by the number of positives.
        # Don't do it for loss[P] because that doesn't depend on the anchors.
        total_num_pos = num_pos.data.sum().float().clamp(min=1.0)
        for k in losses:
            # 원본 규칙 유지:
            #  - P, E, S 는 image/global → batch_size 로 나눔
            #  - 나머지(B, C, M, D, I 등)는 total_num_pos 로 나눔
            if k not in ("P", "E", "S"):
                losses[k] /= total_num_pos
            else:
                losses[k] /= batch_size

        # Loss Key:
        #  - B: Box Localization Loss
        #  - C: Class Confidence Loss
        #  - M: Mask Loss
        #  - P: Prototype Loss
        #  - D: Coefficient Diversity Loss
        #  - E: Class Existence Loss
        #  - S: Semantic Segmentation Loss
        #  - I: Mask IoU Loss
        return losses

    # --------------------------------------------------
    # GIoU loss (for B)
    # --------------------------------------------------
    def giou_loss(self, boxes_pred, boxes_targ, eps=1e-7):
        """
        GIoU Loss
        boxes_pred, boxes_targ: [N, 4], (xmin, ymin, xmax, ymax), 0~1 normalized
        return: [N] = 1 - GIoU
        """
        x1_p, y1_p, x2_p, y2_p = (
            boxes_pred[:, 0],
            boxes_pred[:, 1],
            boxes_pred[:, 2],
            boxes_pred[:, 3],
        )
        x1_t, y1_t, x2_t, y2_t = (
            boxes_targ[:, 0],
            boxes_targ[:, 1],
            boxes_targ[:, 2],
            boxes_targ[:, 3],
        )

        area_p = (x2_p - x1_p).clamp(min=0) * (y2_p - y1_p).clamp(min=0)
        area_t = (x2_t - x1_t).clamp(min=0) * (y2_t - y1_t).clamp(min=0)

        # intersection
        x1_i = torch.max(x1_p, x1_t)
        y1_i = torch.max(y1_p, y1_t)
        x2_i = torch.min(x2_p, x2_t)
        y2_i = torch.min(y2_p, y2_t)

        w_i = (x2_i - x1_i).clamp(min=0)
        h_i = (y2_i - y1_i).clamp(min=0)
        inter = w_i * h_i

        # union
        union = area_p + area_t - inter
        union = union.clamp(min=eps)

        iou = inter / union

        # outer box
        x1_c = torch.min(x1_p, x1_t)
        y1_c = torch.min(y1_p, y1_t)
        x2_c = torch.max(x2_p, x2_t)
        y2_c = torch.max(y2_p, y2_t)

        w_c = (x2_c - x1_c).clamp(min=0)
        h_c = (y2_c - y1_c).clamp(min=0)
        area_c = (w_c * h_c).clamp(min=eps)

        giou = iou - (area_c - union) / area_c
        giou = torch.clamp(giou, min=-1.0, max=1.0)

        return 1.0 - giou

    # --------------------------------------------------
    # size-aware weight helper (bbox / mask 공용)
    # --------------------------------------------------
    def _get_size_weights(
        self,
        boxes,
        small_thr=0.005,  # area < 0.005 → small
        large_thr=0.01,   # area > 0.01 → large
        small_w=0.5,
        med_w=1.5,
        large_w=2.0,
    ):
        """
        boxes: [N, 4], point-form (xmin, ymin, xmax, ymax), 0~1 normalized
        return: [N] size weight
        """
        if boxes.numel() == 0:
            return torch.tensor([], device=boxes.device)

        widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        areas = widths * heights  # [N]

        weights = torch.ones_like(areas, device=boxes.device)

        small_mask = areas < small_thr
        large_mask = areas > large_thr
        medium_mask = (~small_mask) & (~large_mask)

        weights[small_mask] = small_w
        weights[medium_mask] = med_w
        weights[large_mask] = large_w

        return weights.detach()

    def class_existence_loss(self, class_data, class_existence_t):
        return cfg.class_existence_alpha * F.binary_cross_entropy_with_logits(
            class_data, class_existence_t, reduction="sum"
        )

    def semantic_segmentation_loss(
        self, segment_data, mask_t, class_t, interpolation_mode="bilinear"
    ):
        # Note num_classes here is without the background class so cfg.num_classes-1
        batch_size, num_classes, mask_h, mask_w = segment_data.size()
        loss_s = 0

        for idx in range(batch_size):
            cur_segment = segment_data[idx]
            cur_class_t = class_t[idx]

            with torch.no_grad():
                downsampled_masks = F.interpolate(
                    mask_t[idx].unsqueeze(0),
                    (mask_h, mask_w),
                    mode=interpolation_mode,
                    align_corners=False,
                ).squeeze(0)
                downsampled_masks = downsampled_masks.gt(0.5).float()

                # Construct Semantic Segmentation
                segment_t = torch.zeros_like(cur_segment, requires_grad=False)
                for obj_idx in range(downsampled_masks.size(0)):
                    segment_t[cur_class_t[obj_idx]] = torch.max(
                        segment_t[cur_class_t[obj_idx]], downsampled_masks[obj_idx]
                    )

            loss_s += F.binary_cross_entropy_with_logits(
                cur_segment, segment_t, reduction="sum"
            )

        return loss_s / mask_h / mask_w * cfg.semantic_segmentation_alpha

    def ohem_conf_loss(self, conf_data, conf_t, pos, num):
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        if cfg.ohem_use_most_confident:
            # i.e. max(softmax) along classes > 0
            batch_conf = F.softmax(batch_conf, dim=1)
            loss_c, _ = batch_conf[:, 1:].max(dim=1)
        else:
            # i.e. -softmax(class 0 confidence)
            loss_c = log_sum_exp(batch_conf) - batch_conf[:, 0]

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes
        loss_c[conf_t < 0] = 0  # filter out neutrals (conf_t = -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg[pos] = 0
        neg[conf_t < 0] = 0  # Filter out neutrals

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction="none")

        if cfg.use_class_balanced_conf:
            # Lazy initialization
            if self.class_instances is None:
                self.class_instances = torch.zeros(
                    self.num_classes, device=targets_weighted.device
                )

            classes, counts = targets_weighted.unique(return_counts=True)

            for _cls, _cnt in zip(classes.cpu().numpy(), counts.cpu().numpy()):
                self.class_instances[_cls] += _cnt

            self.total_instances += targets_weighted.size(0)

            weighting = 1 - (self.class_instances[targets_weighted] / self.total_instances)
            weighting = torch.clamp(weighting, min=1 / self.num_classes)

            # If you do the math, the average weight of self.class_instances is this
            avg_weight = (self.num_classes - 1) / self.num_classes

            loss_c = (loss_c * weighting).sum() / avg_weight
        else:
            loss_c = loss_c.sum()

        return cfg.conf_alpha * loss_c

    def focal_conf_loss(self, conf_data, conf_t):
        """
        Focal loss as described in https://arxiv.org/pdf/1708.02002.pdf
        """
        conf_t = conf_t.view(-1)  # [batch_size*num_priors]
        conf_data = conf_data.view(
            -1, conf_data.size(-1)
        )  # [batch_size*num_priors, num_classes]

        # Ignore neutral samples (class < 0)
        keep = (conf_t >= 0).float()
        conf_t[conf_t < 0] = 0  # so that gather doesn't drum up a fuss

        logpt = F.log_softmax(conf_data, dim=-1)
        logpt = logpt.gather(1, conf_t.unsqueeze(-1))
        logpt = logpt.view(-1)
        pt = logpt.exp()

        background = (conf_t == 0).float()
        at = (1 - cfg.focal_loss_alpha) * background + cfg.focal_loss_alpha * (
            1 - background
        )

        loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt

        return cfg.conf_alpha * (loss * keep).sum()

    def focal_conf_sigmoid_loss(self, conf_data, conf_t):
        """
        Focal loss but using sigmoid like the original paper.
        """
        num_classes = conf_data.size(-1)

        conf_t = conf_t.view(-1)  # [batch_size*num_priors]
        conf_data = conf_data.view(
            -1, num_classes
        )  # [batch_size*num_priors, num_classes]

        # Ignore neutral samples (class < 0)
        keep = (conf_t >= 0).float()
        conf_t[conf_t < 0] = 0  # can't mask with -1, so filter that out

        # one-hot emb
        conf_one_t = torch.eye(num_classes, device=conf_t.get_device())[conf_t]
        conf_pm_t = conf_one_t * 2 - 1  # -1 if background, +1 if fg for that class

        logpt = F.logsigmoid(conf_data * conf_pm_t)
        pt = logpt.exp()

        at = cfg.focal_loss_alpha * conf_one_t + (1 - cfg.focal_loss_alpha) * (
            1 - conf_one_t
        )
        at[..., 0] = 0  # background alpha = 0

        loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt
        loss = keep * loss.sum(dim=-1)

        return cfg.conf_alpha * loss.sum()

    def focal_conf_objectness_loss(self, conf_data, conf_t):
        """
        class[0] = objectness, sigmoid focal on obj, CE on class for pos
        """

        conf_t = conf_t.view(-1)  # [batch_size*num_priors]
        conf_data = conf_data.view(
            -1, conf_data.size(-1)
        )  # [batch_size*num_priors, num_classes]

        # Ignore neutral samples (class < 0)
        keep = (conf_t >= 0).float()
        conf_t[conf_t < 0] = 0

        background = (conf_t == 0).float()
        at = (1 - cfg.focal_loss_alpha) * background + cfg.focal_loss_alpha * (
            1 - background
        )

        logpt = F.logsigmoid(conf_data[:, 0]) * (1 - background) + F.logsigmoid(
            -conf_data[:, 0]
        ) * background
        pt = logpt.exp()

        obj_loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt

        # class loss (for pos only)
        pos_mask = conf_t > 0
        conf_data_pos = (conf_data[:, 1:])[pos_mask]
        conf_t_pos = conf_t[pos_mask] - 1

        class_loss = F.cross_entropy(conf_data_pos, conf_t_pos, reduction="sum")

        return cfg.conf_alpha * (class_loss + (obj_loss * keep).sum())

    def conf_objectness_loss(self, conf_data, conf_t, batch_size, loc_p, loc_t, priors):
        """
        class[0] = p(obj)*p(IoU) as in YOLO
        """

        conf_t = conf_t.view(-1)  # [batch_size*num_priors]
        conf_data = conf_data.view(
            -1, conf_data.size(-1)
        )  # [batch_size*num_priors, num_classes]

        pos_mask = conf_t > 0
        neg_mask = conf_t == 0

        obj_data = conf_data[:, 0]
        obj_data_pos = obj_data[pos_mask]
        obj_data_neg = obj_data[neg_mask]

        # negative obj loss
        obj_neg_loss = -F.logsigmoid(-obj_data_neg).sum()

        with torch.no_grad():
            pos_priors = (
                priors.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .reshape(-1, 4)[pos_mask, :]
            )

            boxes_pred = decode(loc_p, pos_priors, cfg.use_yolo_regressors)
            boxes_targ = decode(loc_t, pos_priors, cfg.use_yolo_regressors)

            iou_targets = elemwise_box_iou(boxes_pred, boxes_targ)

        obj_pos_loss = (
            -iou_targets * F.logsigmoid(obj_data_pos)
            - (1 - iou_targets) * F.logsigmoid(-obj_data_pos)
        )
        obj_pos_loss = obj_pos_loss.sum()

        # class loss
        conf_data_pos = (conf_data[:, 1:])[pos_mask]
        conf_t_pos = conf_t[pos_mask] - 1

        class_loss = F.cross_entropy(conf_data_pos, conf_t_pos, reduction="sum")

        return cfg.conf_alpha * (class_loss + obj_pos_loss + obj_neg_loss)

    def direct_mask_loss(self, pos_idx, idx_t, loc_data, mask_data, priors, masks):
        """ Crops the gt masks using the predicted bboxes, scales them down, and outputs the BCE loss. """
        loss_m = 0
        for idx in range(mask_data.size(0)):
            with torch.no_grad():
                cur_pos_idx = pos_idx[idx, :, :]
                cur_pos_idx_squeezed = cur_pos_idx[:, 1]

                # Shape: [num_priors, 4], decoded predicted bboxes
                pos_bboxes = decode(
                    loc_data[idx, :, :], priors.data, cfg.use_yolo_regressors
                )
                pos_bboxes = pos_bboxes[cur_pos_idx].view(-1, 4).clamp(0, 1)
                pos_lookup = idx_t[idx, cur_pos_idx_squeezed]

                cur_masks = masks[idx]
                pos_masks = cur_masks[pos_lookup, :, :]

                # Convert bboxes to absolute coordinates
                num_pos, img_height, img_width = pos_masks.size()

                from ..box_utils import sanitize_coordinates  # 보통 여기서 import 되어 있음

                # Take care of all the bad behavior that can be caused by out of bounds coordinates
                x1, x2 = sanitize_coordinates(
                    pos_bboxes[:, 0], pos_bboxes[:, 2], img_width
                )
                y1, y2 = sanitize_coordinates(
                    pos_bboxes[:, 1], pos_bboxes[:, 3], img_height
                )

                # Crop each gt mask with the predicted bbox and rescale to the predicted mask size
                scaled_masks = []
                for jdx in range(num_pos):
                    tmp_mask = pos_masks[jdx, y1[jdx] : y2[jdx], x1[jdx] : x2[jdx]]

                    # Restore any dimensions we've left out because our bbox was 1px wide
                    while tmp_mask.dim() < 2:
                        tmp_mask = tmp_mask.unsqueeze(0)

                    new_mask = F.adaptive_avg_pool2d(
                        tmp_mask.unsqueeze(0), cfg.mask_size
                    )
                    scaled_masks.append(new_mask.view(1, -1))

                mask_t = torch.cat(scaled_masks, 0).gt(0.5).float()  # Threshold

            pos_mask_data = mask_data[idx, cur_pos_idx_squeezed, :]
            loss_m += (
                F.binary_cross_entropy(
                    torch.clamp(pos_mask_data, 0, 1), mask_t, reduction="sum"
                )
                * cfg.mask_alpha
            )

        return loss_m

    def coeff_diversity_loss(self, coeffs, instance_t):
        """
        coeffs     should be size [num_pos, num_coeffs]
        instance_t should be size [num_pos] and be values from 0 to num_instances-1
        """
        num_pos = coeffs.size(0)
        instance_t = instance_t.view(-1)  # juuuust to make sure

        coeffs_norm = F.normalize(coeffs, dim=1)
        cos_sim = coeffs_norm @ coeffs_norm.t()

        inst_eq = (
            instance_t[:, None].expand_as(cos_sim)
            == instance_t[None, :].expand_as(cos_sim)
        ).float()

        # Rescale to be between 0 and 1
        cos_sim = (cos_sim + 1) / 2

        # If they're the same instance, use cosine distance, else use cosine similarity
        loss = (1 - cos_sim) * inst_eq + cos_sim * (1 - inst_eq)

        # Only divide by num_pos once because we're summing over a num_pos x num_pos tensor
        return cfg.mask_proto_coeff_diversity_alpha * loss.sum() / num_pos

    def lincomb_mask_loss(
        self,
        pos,
        idx_t,
        loc_data,
        mask_data,
        priors,
        proto_data,
        masks,
        gt_box_t,
        score_data,
        inst_data,
        labels,
        interpolation_mode="bilinear",
    ):
        mask_h = proto_data.size(1)
        mask_w = proto_data.size(2)

        process_gt_bboxes = (
            cfg.mask_proto_normalize_emulate_roi_pooling or cfg.mask_proto_crop
        )

        if cfg.mask_proto_remove_empty_masks:
            # Make sure to store a copy of this because we edit it to get rid of all-zero masks
            pos = pos.clone()

        loss_m = 0
        loss_d = 0  # Coefficient diversity loss

        maskiou_t_list = []
        maskiou_net_input_list = []
        label_t_list = []

        # Dice weight (없으면 0으로 가정)
        lambda_dice = getattr(cfg, "mask_dice_alpha", 0.0)

        for idx in range(mask_data.size(0)):
            with torch.no_grad():
                downsampled_masks = F.interpolate(
                    masks[idx].unsqueeze(0),
                    (mask_h, mask_w),
                    mode=interpolation_mode,
                    align_corners=False,
                ).squeeze(0)
                downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()

                if cfg.mask_proto_binarize_downsampled_gt:
                    downsampled_masks = downsampled_masks.gt(0.5).float()

                if cfg.mask_proto_remove_empty_masks:
                    # Get rid of gt masks that are so small they get downsampled away
                    very_small_masks = downsampled_masks.sum(dim=(0, 1)) <= 0.0001
                    for i in range(very_small_masks.size(0)):
                        if very_small_masks[i]:
                            pos[idx, idx_t[idx] == i] = 0

                if cfg.mask_proto_reweight_mask_loss:
                    # Ensure that the gt is binary
                    if not cfg.mask_proto_binarize_downsampled_gt:
                        bin_gt = downsampled_masks.gt(0.5).float()
                    else:
                        bin_gt = downsampled_masks

                    gt_foreground_norm = bin_gt / (
                        torch.sum(bin_gt, dim=(0, 1), keepdim=True) + 0.0001
                    )
                    gt_background_norm = (1 - bin_gt) / (
                        torch.sum(1 - bin_gt, dim=(0, 1), keepdim=True) + 0.0001
                    )

                    mask_reweighting = (
                        gt_foreground_norm * cfg.mask_proto_reweight_coeff
                        + gt_background_norm
                    )
                    mask_reweighting *= mask_h * mask_w

            cur_pos = pos[idx]
            pos_idx_t = idx_t[idx, cur_pos]

            if process_gt_bboxes:
                # Note: this is in point-form
                if cfg.mask_proto_crop_with_pred_box:
                    pos_gt_box_t = decode(
                        loc_data[idx, :, :], priors.data, cfg.use_yolo_regressors
                    )[cur_pos]
                else:
                    pos_gt_box_t = gt_box_t[idx, cur_pos]

            if pos_idx_t.size(0) == 0:
                continue

            proto_masks = proto_data[idx]
            proto_coef = mask_data[idx, cur_pos, :]
            if cfg.use_mask_scoring:
                mask_scores = score_data[idx, cur_pos, :]

            if cfg.mask_proto_coeff_diversity_loss:
                if inst_data is not None:
                    div_coeffs = inst_data[idx, cur_pos, :]
                else:
                    div_coeffs = proto_coef

                loss_d += self.coeff_diversity_loss(div_coeffs, pos_idx_t)

            # If we have over the allowed number of masks, select a random sample
            old_num_pos = proto_coef.size(0)
            if old_num_pos > cfg.masks_to_train:
                perm = torch.randperm(proto_coef.size(0))
                select = perm[: cfg.masks_to_train]

                proto_coef = proto_coef[select, :]
                pos_idx_t = pos_idx_t[select]

                if process_gt_bboxes:
                    pos_gt_box_t = pos_gt_box_t[select, :]
                if cfg.use_mask_scoring:
                    mask_scores = mask_scores[select, :]

            num_pos = proto_coef.size(0)
            mask_t = downsampled_masks[:, :, pos_idx_t]
            label_t = labels[idx][pos_idx_t]

            # Size: [mask_h, mask_w, num_pos]
            pred_masks = proto_masks @ proto_coef.t()
            pred_masks = cfg.mask_proto_mask_activation(pred_masks)

            # size-aware weight (per mask)
            size_weight = None
            if process_gt_bboxes:
                with torch.no_grad():
                    size_weight = self._get_size_weights(pos_gt_box_t)  # [num_pos]

            # double loss (optional)
            if cfg.mask_proto_double_loss:
                if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                    pre_loss_double = F.binary_cross_entropy(
                        torch.clamp(pred_masks, 0, 1),
                        mask_t,
                        reduction="sum",
                    )
                else:
                    pre_loss_double = F.smooth_l1_loss(
                        pred_masks, mask_t, reduction="sum"
                    )
                loss_m += cfg.mask_proto_double_loss_alpha * pre_loss_double

            # crop if needed
            if cfg.mask_proto_crop:
                pred_masks_c = crop(pred_masks, pos_gt_box_t)
                mask_t_c = mask_t
            else:
                pred_masks_c = pred_masks
                mask_t_c = mask_t

            # BCE(or L1) loss, per-pixel
            if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                bce_loss = F.binary_cross_entropy(
                    torch.clamp(pred_masks_c, 0, 1),
                    mask_t_c,
                    reduction="none",
                )
            else:
                bce_loss = F.smooth_l1_loss(
                    pred_masks_c,
                    mask_t_c,
                    reduction="none",
                )

            if cfg.mask_proto_normalize_mask_loss_by_sqrt_area:
                gt_area = torch.sum(mask_t_c, dim=(0, 1), keepdim=True)
                bce_loss = bce_loss / (torch.sqrt(gt_area) + 0.0001)

            if cfg.mask_proto_reweight_mask_loss:
                bce_loss = bce_loss * mask_reweighting[:, :, pos_idx_t]

            if cfg.mask_proto_normalize_emulate_roi_pooling:
                weight = mask_h * mask_w if cfg.mask_proto_crop else 1
                pos_gt_csize = center_size(pos_gt_box_t)
                gt_box_width = pos_gt_csize[:, 2] * mask_w
                gt_box_height = pos_gt_csize[:, 3] * mask_h
                bce_loss = (
                    bce_loss.sum(dim=(0, 1)) / gt_box_width / gt_box_height * weight
                )
            else:
                # 아직 3D면 spatial sum
                if bce_loss.dim() == 3:
                    bce_loss = bce_loss.sum(dim=(0, 1))  # [num_pos]

            # size-aware weight
            if size_weight is not None:
                bce_loss = bce_loss * size_weight  # [num_pos]

            # If the number of masks were limited scale the loss accordingly
            if old_num_pos > num_pos:
                bce_loss *= old_num_pos / num_pos

            bce_term = bce_loss.sum()

            # Dice loss (선택적)
            dice_term = 0.0
            if lambda_dice > 0:
                dice_loss_per_mask = self._dice_loss_from_probs(
                    pred_masks_c, mask_t_c
                )  # [num_pos]

                if size_weight is not None:
                    dice_loss_per_mask = dice_loss_per_mask * size_weight

                if old_num_pos > num_pos:
                    dice_loss_per_mask *= old_num_pos / num_pos

                dice_term = lambda_dice * dice_loss_per_mask.sum()

            loss_m += bce_term + dice_term

            # MaskIoU
            if cfg.use_maskiou:
                if cfg.discard_mask_area > 0:
                    gt_mask_area = torch.sum(mask_t, dim=(0, 1))
                    select = gt_mask_area > cfg.discard_mask_area

                    if torch.sum(select) < 1:
                        continue

                    pos_gt_box_t = pos_gt_box_t[select, :]
                    pred_masks = pred_masks[:, :, select]
                    mask_t = mask_t[:, :, select]
                    label_t = label_t[select]

                maskiou_net_input = (
                    pred_masks.permute(2, 0, 1).contiguous().unsqueeze(1)
                )
                pred_masks_bin = pred_masks.gt(0.5).float()
                maskiou_t = self._mask_iou(pred_masks_bin, mask_t)

                maskiou_net_input_list.append(maskiou_net_input)
                maskiou_t_list.append(maskiou_t)
                label_t_list.append(label_t)

        losses = {"M": loss_m * cfg.mask_alpha / mask_h / mask_w}

        if cfg.mask_proto_coeff_diversity_loss:
            losses["D"] = loss_d

        if cfg.use_maskiou:
            # discard_mask_area discarded every mask in the batch, so nothing to do here
            if len(maskiou_t_list) == 0:
                return losses, None

            maskiou_t = torch.cat(maskiou_t_list)
            label_t = torch.cat(label_t_list)
            maskiou_net_input = torch.cat(maskiou_net_input_list)

            num_samples = maskiou_t.size(0)
            if cfg.maskious_to_train > 0 and num_samples > cfg.maskious_to_train:
                perm = torch.randperm(num_samples)
                # BUG FIX: masks_to_train → maskious_to_train
                select = perm[: cfg.maskious_to_train]
                maskiou_t = maskiou_t[select]
                label_t = label_t[select]
                maskiou_net_input = maskiou_net_input[select]

            return losses, [maskiou_net_input, maskiou_t, label_t]

        return losses

    def _mask_iou(self, mask1, mask2):
        intersection = torch.sum(mask1 * mask2, dim=(0, 1))
        area1 = torch.sum(mask1, dim=(0, 1))
        area2 = torch.sum(mask2, dim=(0, 1))
        union = (area1 + area2) - intersection

        eps = 1e-6
        union_safe = union.clone()
        union_safe[union_safe <= eps] = eps

        iou = intersection / union_safe
        # union이 0(사실상 empty)인 경우 IoU=0 처리
        iou[union <= eps] = 0.0

        return iou

    def mask_iou_loss(self, net, maskiou_targets):
        maskiou_net_input, maskiou_t, label_t = maskiou_targets

        maskiou_p = net.maskiou_net(maskiou_net_input)

        label_t = label_t[:, None]
        maskiou_p = torch.gather(maskiou_p, dim=1, index=label_t).view(-1)

        loss_i = F.smooth_l1_loss(maskiou_p, maskiou_t, reduction="sum")

        return loss_i * cfg.maskiou_alpha

    def _dice_loss_from_probs(self, probs, targets, eps=1e-6):
        """
        Dice loss from probability predictions.

        Args:
            probs: [H, W, N] - predicted mask probabilities
            targets: [H, W, N] - ground truth masks (0 or 1)
        Returns:
            [N] - per-mask Dice loss
        """
        N = probs.shape[2]
        probs_flat = probs.permute(2, 0, 1).contiguous().view(N, -1)
        targets_flat = targets.permute(2, 0, 1).contiguous().view(N, -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + eps) / (union + eps)
        loss = 1.0 - dice

        # union이 거의 0(둘 다 empty)인 경우 loss=0
        zero_union = union <= eps
        loss = torch.where(zero_union, torch.zeros_like(loss), loss)

        return loss
