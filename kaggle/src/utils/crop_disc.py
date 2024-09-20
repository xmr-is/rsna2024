
#from src.utils.visualize_helper import visualize_prediction
from os import path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

class CropDiscs(object):

    def crop_by_coordinate(img, coordinates, patch_size=30):
        cropped_patches = []
        img = (img.cpu().numpy() + 1) / 2
        img = np.squeeze(img, 0)  
        cs = coordinates[0, ...].cpu().numpy() * 384  
        
        for i, (x, y) in enumerate(zip(cs[0::2], cs[1::2])):
            
            x, y = int(x), int(y)

            patch = img[:, max(0,y-patch_size):y+patch_size, max(0,x-patch_size-50):x+patch_size+20]  # Shape: [B, C, h, w] for each patch
            patch = torch.from_numpy(patch).unsqueeze(0)
            patch_resized = F.interpolate(patch, size=(384, 384), mode='bilinear', align_corners=False)
            patch_resized = patch_resized.squeeze(0)
            # if patch.shape[0] > 1: patch = patch[4, :, :]  # 最初のチャンネルを選択

            # plt.figure(figsize=(5, 5))
            # # Plot the extracted disc
            # plt.imshow(patch, cmap='gray')
            # plt.axis('off')
            # plt.savefig(f"/Users/markun/git/rsna2024/kaggle/outputs/discs{i}.png")
            # visualize_prediction(patch)
            cropped_patches.append(patch_resized)
        cropped_patches_tensor = torch.concat(cropped_patches)
        return cropped_patches_tensor

    def visualize_prediction(in_image): 
        
            # # Select Data
            # img = (in_image.cpu().numpy() + 1) / 2
            # img = np.squeeze(img, 0).transpose(1, 2, 0)  # Transpose for (H, W, C) format

            if img.shape[2] > 1:
                img = img[:, :, 0]

            plt.figure(figsize=(5, 5))
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.savefig("/Users/markun/git/rsna2024/kaggle/outputs/discs.png")
            #plt.show()
    #         plt.close(fig)