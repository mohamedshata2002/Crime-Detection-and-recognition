import os
from transformers import AutoModel, AutoTokenizer
from diffusers import StableDiffusionUpscalePipeline

# Specify local directories
model_dir = "./model"
data_dir = "./data"

# Create directories if they do not exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Download and save the model locally
model = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler")
model.save_pretrained(model_dir)

# Download and save the image locally
url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
response = requests.get(url)
with open(os.path.join(data_dir, "low_res_cat.png"), "wb") as f:
    f.write(response.content)
