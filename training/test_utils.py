import wandb
import torch
import numpy as np

def calculate_iou(preds, labels):
    # Flatten label and prediction tensors
    preds = preds.view(-1)
    labels = labels.view(-1)

    # Calculate intersection and union
    intersection = (preds & labels).float().sum()
    union = (preds | labels).float().sum()

    if union == 0:
        iou = 1  # If both prediction and label are empty, set IoU to 1
    else:
        iou = intersection / union  # Calculate IoU
    return iou

# Modify the test loop to include logging and metrics calculation
import numpy as np
import torch
import wandb

def test_model(model, test_loader):
    model.eval()
    all_ious = []
    test_images_list = []
    ground_truth_list = []
    prediction_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            images, masks = batch
            predictions = model(images)
            preds = torch.argmax(predictions, dim=1)
            
            # Separate lists for test images, ground truths, and predictions
            test_images_list.extend([wandb.Image(image.float()) for image in images])
            ground_truth_list.extend([wandb.Image(mask.float()) for mask in masks])
            prediction_list.extend([wandb.Image(pred.float()) for pred in preds])
            
            # Calculate IoU for the batch
            iou = calculate_iou(preds, masks)
            all_ious.append(iou)
    
    # Calculate mean IoU
    mean_iou = np.mean(all_ious)
    
    # Log images under separate keys for a three-column layout
    wandb.log({
        "mean_test_iou": mean_iou,
        "test": {"test_images": test_images_list,
        "ground_truths": ground_truth_list,
        "predictions": prediction_list}

    })

    return mean_iou
