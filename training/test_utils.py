import wandb
import torch
import torchmetrics
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

# Class labels dictionary
class_labels = {
    0: "No Car Element",
    1: "Hood",
    2: "Front Door",
    3: "Rear Door",
    4: "Frame",
    5: "Rear Quarter Panel",
    6: "Trunk Lid",
    7: "Fender",
    8: "Bumper",
    9: "Rest of Car"
}

def preprocess_image(image_tensor):
    # Convert tensor to PIL Image if necessary
    return image_tensor

def create_mask_dictionary(mask_tensor):
    """
    Create a dictionary for the mask with class descriptions.
    :param mask_tensor: Tensor of the mask.
    :return: Dictionary suitable for wandb.Image masks parameter.
    """
    mask_array = mask_tensor.numpy()
    mask_dict = {
        "mask_data": mask_array,
        "class_labels": class_labels
    }
    return mask_dict

def test_model(model, test_loader):
    model.eval()
    all_ious = []
    all_dice = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            predictions = model(images)
            preds = torch.argmax(predictions, dim=1)

            for idx, (image, mask, pred) in enumerate(zip(images, masks, preds)):
                vis_image = preprocess_image(image.float())
                gt_mask_dict = create_mask_dictionary(mask)
                pred_mask_dict = create_mask_dictionary(pred)

                caption = f"Batch {batch_idx}, Image {idx}"
                wandb.log({
                    "test_image": wandb.Image(vis_image, 
                                              caption=caption, 
                                              masks={"ground_truth": gt_mask_dict, "prediction": pred_mask_dict})
                })

            all_ious.append(calculate_iou(preds, masks))
            all_dice.append(torchmetrics.functional.dice(preds, masks))

    mean_iou = np.mean(all_ious)
    mean_dice = np.mean(all_dice)
    wandb.log({
        "mean_test_iou": mean_iou,
        "dice_score": mean_dice})


    return mean_iou
