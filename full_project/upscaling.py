import requests
from PIL import Image
from io import BytesIO
from diffusers import LDMSuperResolutionPipeline
import torch
import cv2
import numpy as np
import os 

def diffusion():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path= "E:\\workstation\\project\\project\\crime_detection\\full_project\\upscaling_weights"
    model_path = os.path.normpath(model_path)
    # load model and scheduler
    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_path,local_files_only=True)
    pipeline = pipeline.to(device)
    return  pipeline

def upscale_image(input_image, pipeline):
    # Convert the input CV2 image to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    
    # Resize the image to 128x128
    pil_image = pil_image.resize((128, 128))
    
    # Run the pipeline on the low-resolution image
    upscaled_image = pipeline(pil_image, num_inference_steps=100, eta=1).images[0]
    
    # Convert the output PIL Image back to CV2 image
    cv2_upscaled_image = cv2.cvtColor(np.array(upscaled_image), cv2.COLOR_RGB2BGR)
    
    return cv2_upscaled_image

# Example usage
# pipeline = diffusion()
# input_image = cv2.imread("C:\\Users\\moham\\Downloads\\test4.jpg")
# upscaled_image = upscale_image(input_image,pipeline)
# cv2.imshow("show",upscaled_image)
# cv2.waitKey()
# cv2.destroyAllWindows()