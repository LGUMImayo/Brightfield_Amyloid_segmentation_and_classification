import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import os

# --- 1. Define the Model Architecture ---
# This must be the exact same architecture as in your inference script.
print("Initializing the DeepLabV3-ResNet50 model...")
model = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')

# The number of output classes for your task (tissue, gray-matter, white-matter)
num_classes = 3 
model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

# Set the model to evaluation mode
model.eval()
print("Model initialized.")

# --- 2. Create a Dummy Input Tensor ---
# The model graph is traced by passing a sample input through it.
# The input shape should match what your model expects:
# (batch_size, channels, height, width)
image_size = 520 # From your inference_test.py config
dummy_input = torch.randn(1, 3, image_size, image_size)
print(f"Created a dummy input tensor of shape: {dummy_input.shape}")

# --- 3. Use SummaryWriter to Export the Graph ---
# Get the absolute path of the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, 'runs/deeplabv3_fluxogram')

print(f"Exporting graph to log directory: {log_dir}")
writer = SummaryWriter(log_dir)

# Add the model graph to the writer, allowing for non-strict tracing
writer.add_graph(model, dummy_input, use_strict_trace=False)

# Close the writer
writer.close()

print("\nModel graph has been exported successfully!")
print("To view the fluxogram, run the following command in your terminal:")
print(f"tensorboard --logdir={log_dir}")