import numpy as np
import torch
import yaml

from model import YOLONet

def calculate_iou(bbox1, bbox2):
    """ 计算IOU """
    # bbox: x y w h
    bbox1, bbox2 = bbox1.cpu().detach().numpy().tolist(),    bbox2.cpu().detach().numpy().tolist()

    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]

    max_left = max(bbox1[0] - bbox1[2] / 2, bbox2[0] - bbox2[2] / 2)
    min_right = min(bbox1[0] + bbox1[2] / 2, bbox2[0] + bbox2[2] / 2)
    max_top = max(bbox1[1] - bbox1[3] / 2, bbox2[1] - bbox2[3] / 2)
    min_bottom = min(bbox1[1] + bbox1[3] / 2, bbox2[1] + bbox2[3] / 2)

    if max_left >= min_right or max_top >= min_bottom:
        return 0
    else:
        intersect = (min_right - max_left) * (min_bottom - max_top)
        return intersect / (area1 + area2 - intersect)

def xywhc2label(bboxs, S, B, num_classes):
    # 将一组边界框信息转换成用于目标检测任务的标签数据
    # bboxs is a xywhc list: [(x,y,w,h,c),(x,y,w,h,c),....]
    label = np.zeros((S, S, 5 * B + num_classes))
    for x, y, w, h, c in bboxs:
        x_grid = int(x//(1.0/S))
        y_grid = int(y//(1.0/S))

        xx, yy = x, y
        label[y_grid, x_grid, 0:5] = np.array([xx, yy, w, h, 1])
        label[y_grid, x_grid, 5:10] = np.array([xx, yy, w, h, 1])
        label[y_grid, x_grid, 10 + c] = 1

    return label    


def pred2xywhcc(pred, S, B, num_classes, conf_thresh, iou_thresh):
    """ 得到最终的预测bounding box """
    bboxs = torch.zeros((S * S, 5 + num_classes))
    for x in range(5):
        for y in range(5):
            conf1, conf2 = pred[x, y, 4], pred[x, y, 9]
            if conf1 > conf2:
                bboxs[(x * S + y), 0:4] = torch.Tensor([
                    pred[x, y, 0], pred[x, y, 1], pred[x, y, 2], pred[x, y, 3]])
                bboxs[((x * S + y)), 4] = pred[x, y, 4]
                bboxs[((x * S + y)), 5:] = pred[x, y, 10:]

            else:
                bboxs[(x * S + y), 0:4] = torch.Tensor([
                    pred[x, y, 5], pred[x, y, 6], pred[x, y, 7], pred[x, y, 8]])
                bboxs[((x * S + y)), 4] = pred[x, y, 9]
                bboxs[((x * S + y)), 5:] = pred[x, y, 10:]

    # 非极大值抑制
    xywhcc = nms(bboxs, num_classes, conf_thresh, iou_thresh)
    return xywhcc

def nms(bboxs, num_classes, conf_thresh = 0.1, iou_thresh = 0.3):
    """ 非极大值抑制，得到最终的预测框 """
    # bbox is a 98*15 tensor
    bbox_prob = bboxs[:, 5:].clone().detach() # 98*10
    bbox_conf = bboxs[:, 4].clone().detach().unsqueeze(1).expand_as(bbox_prob)       # 98*10
    bboxs_cls_spec_conf = bbox_conf * bbox_prob
    bboxs_cls_spec_conf[bboxs_cls_spec_conf <= conf_thresh] = 0
    
    # 对于每一类，将置信度排序
    for c in range(num_classes):
        rank = torch.sort(bboxs_cls_spec_conf[:, c], descending=True).indices
        # 对于每一个grid
        for i in range(bboxs.shape[0]):
            if bboxs_cls_spec_conf[rank[i], c] == 0:
                continue
            for j in range(i+1, bboxs.shape[0]):
                if bboxs_cls_spec_conf[rank[j], c] != 0:
                    iou = calculate_iou(bboxs[rank[i], 0:4], bboxs[rank[j], 0:4])
                    if iou > iou_thresh:
                        bboxs_cls_spec_conf[rank[j], c] = 0
    
    # 排除类别概率与目标置信度的乘积为0的边界框
    bboxs = bboxs[torch.max(bboxs_cls_spec_conf, dim=1).values > 0]

    bboxs_cls_spec_conf = bboxs_cls_spec_conf[torch.max(bboxs_cls_spec_conf, dim=1).values > 0]

    ret = torch.ones(bboxs.size()[0], 6)
    # ret: x y w h (conf) (class)

    if bboxs.size()[0] == 0:
        return torch.tensor([])
    
    ret[:, 0:4] = bboxs[:, 0:4]
    ret[:, 4] = torch.max(bboxs_cls_spec_conf, dim=1).values
    ret[:, 5] = torch.argmax(bboxs[:, 5:], dim=1).int()

    return ret

def parse_cfg(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print('Config:', cfg)
    return cfg

def build_model(weight_path, S, B, num_classes):
    model = YOLONet(S, B, num_classes)
    if weight_path and weight_path != '':
        model.load_state_dict(torch.load(weight_path))
    return model
