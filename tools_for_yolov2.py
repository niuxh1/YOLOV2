import torch
import torch.nn as nn
import numpy as np
ignore_thresh = 0.5


import torch
import torch.nn as nn

ignore_thresh = 0.5

class MSE_logits_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets, mask):
        outputs = torch.sigmoid(logits)  # 修正拼写错误

        pos_id = (mask == 1.0).float()
        neg_id = (mask == 0.0).float()
        pos_loss = pos_id * (outputs - targets) ** 2
        neg_loss = neg_id * (outputs ** 2)
        loss = 5.0 * pos_loss + 1.0 * neg_loss
        if self.reduction == 'mean':
            batch_size = logits.size(0)
            return torch.sum(loss) / batch_size
        else:
            return loss

# 省略其他函数


def compute_iou(anchor_boxes, target_boxes):
    """

    :param anchor_boxes: [[c_x,c_y,w,h],...]
    :param target_boxes: [[c_x,c_y,w,h]]
    :return: [iou1,iou2,...]
    """
    ab_x1_y1_x2_y2 = torch.zeros(len(anchor_boxes), 4)
    ab_x1_y1_x2_y2[:, 0] = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2
    ab_x1_y1_x2_y2[:, 1] = anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2
    ab_x1_y1_x2_y2[:, 2] = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2
    ab_x1_y1_x2_y2[:, 3] = anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2
    w_ab, h_ab = anchor_boxes[:, 2], anchor_boxes[:, 3]

    target_boxes_like_anchor = target_boxes.repeat(len(anchor_boxes), 1)

    tgt_x1_y1_x2_y2 = torch.zeros(len(anchor_boxes), 4)
    tgt_x1_y1_x2_y2[:, 0] = target_boxes_like_anchor[:, 0] - target_boxes_like_anchor[:, 2] / 2
    tgt_x1_y1_x2_y2[:, 1] = target_boxes_like_anchor[:, 1] - target_boxes_like_anchor[:, 3] / 2
    tgt_x1_y1_x2_y2[:, 2] = target_boxes_like_anchor[:, 0] + target_boxes_like_anchor[:, 2] / 2
    tgt_x1_y1_x2_y2[:, 3] = target_boxes_like_anchor[:, 1] + target_boxes_like_anchor[:, 3] / 2
    w_tgt, h_tgt = target_boxes_like_anchor[:, 2], target_boxes_like_anchor[:, 3]

    S_gt = w_tgt * h_tgt
    S_ab = w_ab * h_ab
    I_w = torch.min(ab_x1_y1_x2_y2[:, 2], tgt_x1_y1_x2_y2[:, 2]) - torch.max(ab_x1_y1_x2_y2[:, 0],
                                                                             tgt_x1_y1_x2_y2[:, 0])
    I_h = torch.min(ab_x1_y1_x2_y2[:, 3], tgt_x1_y1_x2_y2[:, 3]) - torch.max(ab_x1_y1_x2_y2[:, 1],
                                                                             tgt_x1_y1_x2_y2[:, 1])
    S_I = I_w * I_h
    Iou = S_I / (S_gt + S_ab - S_I + 1e-20)
    return Iou


def set_anchor(anchor_size):
    """

    :param anchor_size: [[h1,w1],[h2,w2],...]
    :return: [[0,0,h1,w1],[0,0,h2,w2],...]
    """
    anchor_num = len(anchor_size)
    anchors = torch.zeros(anchor_num, 4)

    for ind, size in enumerate(anchor_size):
        anchors[ind, 2:] = torch.tensor(size)

    return anchors


def generate_tx_ty_tw_th(target_label, w, h, s, anchor_size):
    x_min, y_min, x_max, y_max = target_label[:-1]
    c_x = (x_min + x_max) / 2 * w
    c_y = (y_min + y_max) / 2 * h
    box_w = x_max - x_min
    box_h = y_max - y_min

    if box_w < 1. or box_h < 1.:
        return False

    c_x_s = c_x / s
    c_y_s = c_y / s
    box_h_s = box_h / s
    box_w_s = box_w / s
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    anchor_boxes = set_anchor(anchor_size)
    target_boxes = torch.tensor([0, 0, box_w_s, box_h_s])

    iou = compute_iou(anchor_boxes, target_boxes)
    iou_mask = (iou > ignore_thresh)

    result = []

    if iou_mask.sum() == 0:
        idx = torch.argmax(iou)
        p_w, p_h = anchor_size[idx]
        tx = c_x_s - grid_x
        ty = c_y_s - grid_y
        tw = torch.log(box_w_s / p_w)
        th = torch.log(box_h_s / p_h)
        weight = 2.0 - box_w_s * box_h_s
        result.append([idx, grid_x, grid_y, tx, ty, tw, th, weight, x_min, y_min, x_max, y_max])
    else:
        best_idx = torch.argmax(iou)
        for idx, iou_m in enumerate(iou_mask):
            if iou_m:
                if idx == best_idx:
                    p_w, p_h = anchor_size[idx]
                    tx = c_x_s - grid_x
                    ty = c_y_s - grid_y
                    tw = torch.log(box_w_s / p_w)
                    th = torch.log(box_h_s / p_h)
                    weight = 2.0 - box_w_s * box_h_s
                    result.append([idx, grid_x, grid_y, tx, ty, tw, th, weight, x_min, y_min, x_max, y_max])
                else:
                    result.append([idx, grid_x, grid_y, 0., 0., 0., 0., -1.0, 0., 0., 0., 0.])

                return torch.tensor(result)


def target_creat(itorchut_size, stride, label_list, anchor_size):
    """

    :param itorchut_size:
    :param stride:
    :param label_list: [[[x_min,y_min,x_max,y_max,class],...],...]
                        len(label_list)=batch_size
    :param anchor_size:
    :return:
    """
    batch_size = label_list.size(0)
    h, w = itorchut_size
    s = stride
    ws = w // s
    hs = h // s

    all_anchor_size = torch.tensor(anchor_size)
    anchor_num = len(anchor_size)

    target_tenor = torch.zeros(batch_size, hs, ws, anchor_num, 11)

    for batch_index in range(batch_size):
        for target_label in label_list[batch_index]:
            target_class = target_label[-1]
            results = generate_tx_ty_tw_th(target_label, w, h, s, all_anchor_size)
            if results:
                for result in results:
                    idx, grid_x, grid_y, tx, ty, tw, th, weight, x_min, y_min, x_max, y_max = result
                    if weight > 0:
                        if grid_x < target_tenor.shape[2] and grid_y < target_tenor.shape[1]:
                            target_tenor[batch_index, grid_y, grid_x, idx, 0] = 1.0
                            target_tenor[batch_index, grid_y, grid_x, idx, 1] = target_class
                            target_tenor[batch_index, grid_y, grid_x, idx, 2:6] = torch.tensor([tx, ty, tw, th])
                            target_tenor[batch_index, grid_y, grid_x, idx, 6] = weight
                            target_tenor[batch_index, grid_y, grid_x, idx, 7:] = torch.tensor(
                                [x_min, y_min, x_max, y_max])
                    else:
                        target_tenor[batch_index, grid_y, grid_x, idx, 0] = -1.0
                        target_tenor[batch_index, grid_y, grid_x, idx, 6] = -1.0

    target_tenor = target_tenor.view(batch_size, -1, 11)

    return target_tenor


def iou_score(bboxes_a, bboxes_b):
    """
        bbox_1 : [B*N, 4] = [x1, y1, x2, y2]
        bbox_2 : [B*N, 4] = [x1, y1, x2, y2]
    """
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
    return area_i / (area_a + area_b - area_i + 1e-14)


def multi_gt_creator(itorchut_size, strides, label_lists, anchor_size):
    """creator multi scales gt"""
    # prepare the all empty gt datas
    batch_size = len(label_lists)
    h = w = itorchut_size
    num_scale = len(strides)
    gt_tensor = []
    all_anchor_size = anchor_size
    anchor_number = len(all_anchor_size) // num_scale

    for s in strides:
        gt_tensor.append(torch.zeros([batch_size, h // s, w // s, anchor_number, 1 + 1 + 4 + 1 + 4]))

    # generate gt datas
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            # get a bbox coords
            gt_class = int(gt_label[-1])
            xmin, ymin, xmax, ymax = gt_label[:-1]
            # compute the center, width and height
            c_x = (xmax + xmin) / 2 * w
            c_y = (ymax + ymin) / 2 * h
            box_w = (xmax - xmin) * w
            box_h = (ymax - ymin) * h

            if box_w < 1. or box_h < 1.:
                # print('A dirty data !!!')
                continue

                # compute the IoU
            anchor_boxes = set_anchor(all_anchor_size)
            gt_box = torch.tensor([[0, 0, box_w, box_h]])
            iou = compute_iou(anchor_boxes, gt_box)

            # We only consider those anchor boxes whose IoU is more than ignore thresh,
            iou_mask = (iou > ignore_thresh)

            if iou_mask.sum() == 0:
                # We assign the anchor box with highest IoU score.
                index = torch.argmax(iou)
                # s_indx, ab_ind = index // num_scale, index % num_scale
                s_indx = index // anchor_number
                ab_ind = index - s_indx * anchor_number
                # get the corresponding stride
                s = strides[s_indx]
                # get the corresponding anchor box
                p_w, p_h = anchor_boxes[index, 2], anchor_boxes[index, 3]
                # compute the gride cell location
                c_x_s = c_x / s
                c_y_s = c_y / s
                grid_x = int(c_x_s)
                grid_y = int(c_y_s)
                # compute gt labels
                tx = c_x_s - grid_x
                ty = c_y_s - grid_y
                tw = torch.log(box_w / p_w)
                th = torch.log(box_h / p_h)
                weight = 2.0 - (box_w / w) * (box_h / h)

                if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2]:
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = 1.0
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 1] = gt_class
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 2:6] = torch.tensor([tx, ty, tw, th])
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = weight
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 7:] = torch.tensor([xmin, ymin, xmax, ymax])

            else:
                # There are more than one anchor boxes whose IoU are higher than ignore thresh. But we only assign
                # only one anchor box whose IoU is the best(objectness target is 1) and ignore other anchor boxes
                # whose(we set their objectness as -1 which means we will ignore them during computing obj loss )
                # iou_ = iou * iou_mask

                # We get the index of the best IoU
                best_index = torch.argmax(iou)
                for index, iou_m in enumerate(iou_mask):
                    if iou_m:
                        if index == best_index:
                            # s_indx, ab_ind = index // num_scale, index % num_scale
                            s_indx = index // anchor_number
                            ab_ind = index - s_indx * anchor_number
                            # get the corresponding stride
                            s = strides[s_indx]
                            # get the corresponding anchor box
                            p_w, p_h = anchor_boxes[index, 2], anchor_boxes[index, 3]
                            # compute the gride cell location
                            c_x_s = c_x / s
                            c_y_s = c_y / s
                            grid_x = int(c_x_s)
                            grid_y = int(c_y_s)
                            # compute gt labels
                            tx = c_x_s - grid_x
                            ty = c_y_s - grid_y
                            tw = torch.log(box_w / p_w)
                            th = torch.log(box_h / p_h)
                            weight = 2.0 - (box_w / w) * (box_h / h)

                            if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2]:
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = 1.0
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 1] = gt_class
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 2:6] = torch.tensor(
                                    [tx, ty, tw, th])
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = weight
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 7:] = torch.tensor(
                                    [xmin, ymin, xmax, ymax])

                        else:
                            # we ignore other anchor boxes even if their iou scores are higher than ignore thresh
                            # s_indx, ab_ind = index // num_scale, index % num_scale
                            s_indx = index // anchor_number
                            ab_ind = index - s_indx * anchor_number
                            s = strides[s_indx]
                            c_x_s = c_x / s
                            c_y_s = c_y / s
                            grid_x = int(c_x_s)
                            grid_y = int(c_y_s)
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = -1.0
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = -1.0

    gt_tensor = [gt.reshape(batch_size, -1, 1 + 1 + 4 + 1 + 4) for gt in gt_tensor]
    gt_tensor = torch.concatenate(gt_tensor, 1)

    return gt_tensor


def loss(pred_conf, pred_cls, pred_tx_ty_tw_th, pred_iou, label):
    conf_loss_f = MSE_logits_loss(reduction='mean')
    cls_loss_f = nn.CrossEntropyLoss(reduction='mean')
    tx_ty_loss_f = nn.MSELoss(reduction='mean')
    tw_th_loss_f = nn.MSELoss(reduction='mean')
    iou_loss_f = nn.SmoothL1Loss(reduction='mean')

    pred_conf = pred_conf[:,:,0]
    pred_cls  = pred_cls.permute(0,2,1).contiguous()
    pred_tx_ty= pred_tx_ty_tw_th[:,:,:2]
    pred_tw_th = pred_tx_ty_tw_th[:,:,2:]
    pred_iou = pred_iou[:,:,0]

    target_conf = label[:,:,0].float()
    target_obj = label[:,:,1].float()
    target_cls = label[:,:,2].long()
    target_tx_ty = label[:,:,3:5].float()
    target_tw_th = label[:,:,5:7].float()
    target_box_weight = label[:,:,7].float()
    target_iou=(target_box_weight>0.).float()
    target_mask=(target_box_weight>0.).float()

    batch_size = pred_conf.size(0)

    conf_loss=conf_loss_f(pred_conf,target_conf,target_obj)
    cls_loss=torch.sum(cls_loss_f(pred_cls,target_cls)*target_mask)/batch_size
    tx_ty_loss=torch.sum(torch.sum(tx_ty_loss_f(pred_tx_ty,target_tx_ty),dim=-1)*target_box_weight*target_mask)/batch_size
    tw_th_loss=torch.sum(torch.sum(tw_th_loss_f(pred_tw_th,target_tw_th),dim=-1)*target_box_weight*target_mask)/batch_size
    box_loss=tx_ty_loss+tw_th_loss

    iou_loss=torch.sum(iou_loss_f(pred_iou,target_iou)*target_mask)/batch_size

    return conf_loss,cls_loss,box_loss,iou_loss

def collate_fn(batch):
    images, boxes, labels = zip(*batch)  # 解压每个批次的样本

    # 填充目标框和标签
    max_boxes = max(len(b) for b in boxes)  # 找到最大目标框数
    boxes_padded = []
    labels_padded = []

    for box, label in zip(boxes, labels):
        # 填充到 max_boxes 的长度，填充值可以是0
        box_padded = np.pad(box, ((0, max_boxes - len(box)), (0, 0)), 'constant')
        label_padded = label + [''] * (max_boxes - len(label))  # 标签填充
        boxes_padded.append(box_padded)
        labels_padded.append(label_padded)

    return torch.stack(images), boxes_padded, labels_padded

# Iou=compute_iou(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), torch.tensor([1, 2, 3, 4]))
# print(Iou)
# multi_gt_creator(416, [32, 16, 8], [[[0.1, 0.1, 0.3, 0.3, 1], [0.3, 0.3, 0.5, 0.5, 2]]], [[1, 2], [3, 4], [5, 6]])
