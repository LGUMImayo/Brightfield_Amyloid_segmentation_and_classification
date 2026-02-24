import torch
import torchvision
import os

# Create the models directory if it doesn't exist
#os.makedirs('../models', exist_ok=True)

print("Downloading pretrained Faster R-CNN weights...")
# Load the standard model with default (COCO) weights
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

# Save the weights to the file your script expects
save_path = '/fslustre/qhs/ext_chen_yuheng_mayo_edu/Matt_codes/s311590_plaque_ai/s311590_plaque_ai/plaque_ai/models/torchvision_fasterrcnn_resnet50_fpn.pt'
torch.save(model.state_dict(), save_path)
print(f"Saved to {save_path}")