import torch

def compute_iou(preds, masks, num_classes=10):
    preds = torch.argmax(preds, dim=1)

    ious = []

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (masks == cls)

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            continue

        iou = intersection / union
        ious.append(iou)

    if len(ious) == 0:
        return torch.tensor(0.0)

    return sum(ious) / len(ious)