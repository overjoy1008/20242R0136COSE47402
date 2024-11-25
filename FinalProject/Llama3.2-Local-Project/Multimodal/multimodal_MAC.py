import os
import torch

# Set the environment variable to disable the upper limit for memory allocations
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# GPU 사용 여부 확인
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using GPU: Apple Metal Performance Shaders (MPS)")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU.")

from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import requests

# Update the paths to use relative paths with Assets directory
model_path = "Assets/Models/llama_3_2_11B_instruct"
image_path = "Assets/Images/mathProblem2.png"

# Initialize model and processor
model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": device},  # Use the specific device
)
processor = AutoProcessor.from_pretrained(model_path)

# Open the image
image = Image.open(image_path)

# Create a message data structure
messageDataStructure = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": "Compute the side of c of the triangle in the image.",
            },
        ],
    }
]

# Create text input
textInput = processor.apply_chat_template(
    messageDataStructure, add_generation_prompt=True
)

# Call the processor
inputs = processor(image, textInput, return_tensors="pt").to(device)

# Here, change the number of tokens to get a more detailed answer
output = model.generate(**inputs, max_new_tokens=2000)

# Here, we decode and store the response so we can print it
generatedOutput = processor.decode(output[0])

print(generatedOutput)

with open("output.txt", "w", encoding="utf-8") as text_file:
    text_file.write(generatedOutput)
