import torch
from torch import nn

from .util import calculate_iou

class YOLOLoss(nn.Module):

    def __init__(self, S, B):
        super().__init__()
        self.S = S
        self.B = B

    def forward(self, preds, labels):
        # labes:(batch_size, S, S, B * 5 + num_classes)
        batch_size = labels.size(0)

        loss_coord_xy = 0.
        loss_coord_wh = 0.
        loss_obj = 0.
        loss_no_obj = 0.
        loss_class = 0.

        for i in range(batch_size):
            for y in range(self.S):
                for x in range(self.S):
                    if labels[i, y, x, 4] == 1:
                        # 说明有物体
                        pred_bbox1 = torch.tensor(
                            [preds[i, y, x, 0], preds[i, y, x, 1], preds[i, y, x, 2], preds[i, y, x, 3]])
                        pred_bbox2 = torch.tensor(
                            [preds[i, y, x, 5], preds[i, y, x, 6], preds[i, y, x, 7], preds[i, y, x, 8]])
                        label_bbox = torch.tensor(
                            [labels[i, y, x, 0], labels[i, y, x, 1], labels[i, y, x, 2], labels[i, y, x, 3]])
                        
                        iou1 = calculate_iou(pred_bbox1, label_bbox)
                        iou2 = calculate_iou(pred_bbox2, label_bbox)

                        if iou1 > iou2:
                            loss_coord_xy += 5 * torch.sum(
                                (labels[i, y, x, 0:2] - preds[i, y, x, 0:2]) ** 2
                            )

                            loss_coord_wh += torch.sum(
                                (labels[i, y, x, 2:4].sqrt() - preds[i, y, x, 2:4].sqrt() ** 2)
                            )

                            loss_obj += (iou1 - preds[i, y,x, 4]) ** 2

                            loss_no_obj += 0.5*((0 - preds[i, y, x, 9]) ** 2)

                        else:
                             loss_coord_xy += 5 * torch.sum(
                                (labels[i, y, x, 5:7] - preds[i, y, x, 5:7]) ** 2
                            )

                             loss_coord_wh += torch.sum(
                                (labels[i, y, x, 7:9].sqrt() - preds[i, y, x, 7:9].sqrt() ** 2)
                            )

                             loss_obj += (iou1 - preds[i, y,x, 9]) ** 2

                             loss_no_obj += 0.5*((0 - preds[i, y, x, 4]) ** 2)
                        
                        loss_class += torch.sum(
                            (labels[i, y, x, 10:] - preds[i,y, x, 10:]) ** 2               
                        )

                    else:
                        loss_no_obj += 0.5 * torch.sum(
                            (0 - preds[i, y, x, [4, 9]] ** 2)
                        )
        loss = loss_coord_xy + loss_coord_wh + loss_obj  + loss_no_obj + loss_class

        return loss / batch_size
