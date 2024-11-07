import numpy as np
import torch
import torch.nn as nn
import os
from backbone import build_backbone
from utils.modules import CONV, reorg_layer
import tools_for_yolov2 as tools


class yolo_v2_d19(nn.Module):
    def __init__(self, device='cuda', input_size=None, num_classes=20, train=False, confidence_thresh=0.001,
                 nms_thresh=0.5, anchors_size=None):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.train = train
        self.confidence_thresh = confidence_thresh
        self.nms_thresh = nms_thresh
        self.anchors_size = torch.tensor(anchors_size).to(self.device)
        self.stride = 32
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)
        self.num_anchor = len(anchors_size)

        self.backbone = build_backbone(pretrained=True)

        self.conv1 = nn.Sequential(
            CONV(1024, 1024, ksize=3, padding=1),
            CONV(1024, 1024, ksize=3, padding=1)
        )

        self.route_layer = CONV(512, 64, ksize=1)

        self.reorg = reorg_layer(s=2)

        self.conv2 = CONV(1280, 1024, ksize=3, padding=1)

        self.pred = nn.Conv2d(1024, self.num_anchor * (1 + 4 + self.num_classes), kernel_size=1)
        test = self.num_anchor * self.input_size * self.input_size

    def load_pretrained_weights(self, weight_path='path_to_pretrained_weights.pth'):
        # 加载预训练权重
        if os.path.isfile(weight_path):
            self.load_state_dict(torch.load(weight_path))
            print("预训练权重加载成功。")
        else:
            print(f"预训练权重文件未找到：{weight_path}")

    def create_grid(self, input_size):
        w, h = input_size, input_size
        w_out, h_out = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(h_out), torch.arange(w_out)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, h_out * w_out, 1, 2).to(self.device)
        anchor_wh = self.anchors_size.repeat(h_out * w_out, 1, 1).unsqueeze(0).to(self.device)
        return grid_xy, anchor_wh
        # print('x:',grid_x,'\ny:', grid_y)

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)

    def decode_xywh(self, pred):
        pred = pred.to(self.device)
        B, H_W, anchor_num, _ = pred.size()
        xy_pred = torch.sigmoid(pred[..., :2]) + self.grid_cell
        wh_pred = torch.exp(pred[..., 2:]) * self.all_anchor_wh
        all_pred = torch.cat([xy_pred, wh_pred], dim=-1).view(B, -1, 4) * self.stride
        return all_pred

    def decode_boxes(self, pred):
        pred = pred.to(self.device)
        x_y_w_h = self.decode_xywh(pred)
        x1_y1 = x_y_w_h[..., :2] - x_y_w_h[..., 2:] * 0.5
        x2_y2 = x_y_w_h[..., :2] + x_y_w_h[..., 2:] * 0.5
        boxes = torch.cat([x1_y1, x2_y2], dim=-1)
        return boxes

    def nms(self, dets, scores):
        # dets=torch.tensor(dets).to(self.device)
        # scores=torch.tensor(scores).to(self.device)
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order_scores = scores.argsort(descending=True)

        keep = []

        while order_scores.size(0) > 0:
            i = order_scores[0]
            keep.append(i)
            xx1 = torch.max(x1[i], x1[order_scores[1:]])
            yy1 = torch.max(y1[i], y1[order_scores[1:]])
            xx2 = torch.min(x2[i], x2[order_scores[1:]])
            yy2 = torch.min(y2[i], y2[order_scores[1:]])

            w = torch.max(torch.tensor(1e-10, device=self.device), xx2 - xx1)
            h = torch.max(torch.tensor(1e-10, device=self.device), yy2 - yy1)
            inter = w * h

            over = inter / (areas[i] + areas[order_scores[1:]] - inter)
            inds = torch.where(over <= self.nms_thresh)[0]
            order_scores = order_scores[inds + 1]
        return keep

    def post_process(self, boxes, scores):
        scores = scores.clone().detach().to(self.device)
        boxes = boxes.clone().detach().to(self.device)
        cls_inds = torch.argmax(scores, dim=1)
        scores = scores[torch.arange(scores.size(0)), cls_inds]

        keep = torch.zeros(len(boxes), dtype=torch.int)
        for i in range(self.num_classes):
            inds = torch.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_boxes = boxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_boxes, c_scores)

            # 将c_keep转换为张量
            c_keep = torch.tensor(c_keep, dtype=torch.int64, device=self.device)

            if c_keep.numel() > 0:  # 检查c_keep是否有效
                keep[inds[c_keep]] = 1

        keep = torch.where(keep == 1)
        boxes = boxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]
        return boxes, scores, cls_inds


    def inference(self, x):
        feats = self.backbone(x)

        p5 = self.conv1(feats['layer3'])
        p4 = self.reorg(self.route_layer(feats['layer2']))
        p5 = torch.cat([p4, p5], dim=1)

        p5 = self.conv2(p5)

        pred = self.pred(p5)
        B, anchor_C, H, W = pred.size()
        """
        pred: (B, num_anchor*C,H,W)->(B,H,W,num_anchor*C)->(B,H*W,num_anchor*C)
        """
        pred = pred.permute(0, 2, 3, 1).contiguous().view(B, H * W, anchor_C)

        """
        conf_pred: (B,H*W,num_anchor,1)
        cls_pred: (B,H*W,num_anchor,num_classes)
        reg_pred: (B,H*W,num_anchor,4)
        """
        conf_pred = pred[:, :, :1 * self.num_anchor].contiguous().view(B, H * W * self.num_anchor, 1)
        cls_pred = pred[:, :, 1 * self.num_anchor:(1 + self.num_classes) * self.num_anchor].contiguous().view(B,
                                                                                                              H * W * self.num_anchor,
                                                                                                              self.num_classes)
        reg_pred = pred[:, :, (1 + self.num_classes) * self.num_anchor:].contiguous().view(B, H * W, self.num_anchor, 4)
        box_pred = self.decode_boxes(reg_pred)

        """
        you only look one photo at a time
        """
        box_pred = box_pred[0]
        conf_pred = conf_pred[0]
        cls_pred = cls_pred[0]

        """
        you need scores
        """
        scores = torch.sigmoid(conf_pred) * torch.softmax(cls_pred, dim=-1)

        boxes = torch.clamp(box_pred / self.input_size, 0., 1.)

        boxes, scores, cls_inds = self.post_process(boxes, scores)
        return boxes, scores, cls_inds

    def forward(self, x, targets=None):
        if not self.train:
            return self.inference(x)
        else:
            if targets is None:
                return 'You need targets in training mode'
            x = x.to(self.device)

            feats = self.backbone(x)

            p5 = self.conv1(feats['layer3'])
            p4 = self.reorg(self.route_layer(feats['layer2']))
            p5 = torch.cat([p4, p5], dim=1)

            p5 = self.conv2(p5)

            pred = self.pred(p5)
            B, anchor_C, H, W = pred.size()
            """
            pred: (B, num_anchor*C,H,W)->(B,H,W,num_anchor*C)->(B,H*W,num_anchor*C)
            """
            pred = pred.permute(0, 2, 3, 1).contiguous().view(B, H * W, anchor_C)
            """
            conf_pred: (B,H*W,num_anchor,1)
            cls_pred: (B,H*W,num_anchor,num_classes)
            reg_pred: (B,H*W,num_anchor,4)
            """
            conf_pred = pred[:, :, :1 * self.num_anchor].contiguous().view(B, H * W * self.num_anchor, 1)
            cls_pred = pred[:, :, 1 * self.num_anchor:(1 + self.num_classes) * self.num_anchor].contiguous().view(B,
                                                                                                                  H * W * self.num_anchor,
                                                                                                                  self.num_classes)
            reg_pred = pred[:, :, (1 + self.num_classes) * self.num_anchor:].contiguous().view(B, H * W,
                                                                                               self.num_anchor, 4)
            x1_y1_x2_y2_pred = (self.decode_boxes(reg_pred) / self.input_size)[0].view(-1, 4)
            # 假设目标维度不一致，可以根据预测框的数量进行裁剪或扩展
            x1_y1_x2_y2_target = targets[:, :, 7:].view(x1_y1_x2_y2_pred.shape)

            reg_pred = reg_pred.view(B, -1, 4)

            iou_pred = tools.iou_score(x1_y1_x2_y2_pred, x1_y1_x2_y2_target).view(B, -1, 1)

            gt_conf = iou_pred.clone().detach()
            targets = torch.cat((gt_conf, targets[:, :, :7]), dim=2)

            conf_loss, cls_loss, box_loss, iou_loss = tools.loss(pred_conf=conf_pred, pred_cls=cls_pred,
                                                                 pred_tx_ty_tw_th=reg_pred, pred_iou=iou_pred,
                                                                 label=targets)
            return conf_loss, cls_loss, box_loss




# 创建模型打印输出以进行调试
