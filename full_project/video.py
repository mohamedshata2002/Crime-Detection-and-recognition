from ast import arg
import numpy as np
import os
import argparse
import torch.nn as nn
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte
import cv2


from full_project.basicsr.models import create_model
from full_project.basicsr.utils.options import parse

def image_enhancement(image, weights='pretrained_weights/SDSD_indoor.pth', gpus='0'):
    # Set GPU
    gpu_list = ','.join(str(x) for x in gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    # Load model and weights
    opt = parse("E:\\workstation\\project\\project\\enhancement_Retinex\\Options\\RetinexFormer_SDSD_indoor.yml", is_train=False)
    opt['dist'] = False
    model_restoration = create_model(opt).net_g

    checkpoint = torch.load(weights)
    try:
        model_restoration.load_state_dict(checkpoint['params'])
    except:
        new_checkpoint = {}
        for k in checkpoint['params']:
            new_checkpoint['module.' + k] = checkpoint['params'][k]
        model_restoration.load_state_dict(new_checkpoint)

    print("===> Testing using weights: ", weights)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()



    # Process each frame in the video
    with torch.inference_mode():
            img = np.float32(image) / 255.
            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()

            # Padding
            factor = 4
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            input_ = F.pad(input_, (0, W - w, 0, H - h), 'reflect')

            # Enhancement
            restored = model_restoration(input_)

            # Unpad
            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            # Convert back to uint8
            restored_uint8 = img_as_ubyte(restored)
    return restored_uint8

# img  = cv2.imread("C:\\Users\\moham\\Downloads\\test.jpg")
# cv2.imshow("Image",image_enhancement(img,"E:\\workstation\\project\\project\\enhancement_Retinex\\weight\\best_psnr_21.66_95000.pth"))
# cv2.waitKey(0)
# cv2.destroyAllWindows()