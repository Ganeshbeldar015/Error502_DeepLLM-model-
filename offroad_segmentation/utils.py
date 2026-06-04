import torch

def compute_iou(outputs, masks, num_classes=10):
    """
    Fast IoU calculation by argmaxing predictions once and iterating over categorical classes.
    """
    preds = torch.argmax(outputs, dim=1)
    
    # Reshape to (Batch * H * W)
    preds = preds.view(-1)
    masks = masks.view(-1)
    
    iou_list = []
    for cls in range(num_classes):
        # Using vectorized logic on entire tensor
        pred_mask = (preds == cls)
        target_mask = (masks == cls)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union > 0:
            iou_list.append(intersection / union)
            
    if not iou_list:
        return torch.tensor(0.0, device=outputs.device)
        
    return torch.mean(torch.stack(iou_list))