
#from src.utils.visualize_helper import visualize_prediction
from os import path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate

class CropDiscs(object):

    def crop_by_coordinate(img, coordinates, patch_size=30):
        cropped_patches = []
        img = (img.cpu().numpy() + 1) / 2
        img = np.squeeze(img, 0)  
        cs = coordinates[0, ...].cpu().numpy() * 384
        
        for i, (x, y) in enumerate(zip(cs[0::2], cs[1::2])):
            
            x, y = int(x), int(y)
            angles=[15, 12, 5, -5, -30]
            angle = angles[i % len(angles)]  # 5箇所に対して角度を適用

            img_tensor = torch.from_numpy(img).unsqueeze(0) # (C, H, W) -> (1, C, H, W)
            patch_rotated = rotate(img_tensor, angle)
            patch_rotated = patch_rotated.squeeze(0).numpy() # (1, C, H, W) -> (C, H, W)

            if i == 4:
                patch = patch_rotated[:, max(0,y-patch_size-20):y+patch_size-10, max(0,x-patch_size-100):x+patch_size-50]
            elif i == 3:
                patch = patch_rotated[:, max(0,y-patch_size-10):y+patch_size, max(0,x-patch_size-50):x+patch_size-20]
            elif i == 2:
                patch = patch_rotated[:, max(0,y-patch_size):y+patch_size, max(0,x-patch_size-50):x+patch_size-8]
            elif i == 1:
                patch = patch_rotated[:, max(0,y-patch_size):y+patch_size+10, max(0,x-patch_size-50):x+patch_size]
            else:
                patch = patch_rotated[:, max(0,y-patch_size+8):y+patch_size+10, max(0,x-patch_size-50):x+patch_size]
            patch = torch.from_numpy(patch).unsqueeze(0) # (C, H, W) -> (1, C, H, W)
                                                      
            patch_resized = F.interpolate(patch, size=(384, 384), mode='bilinear', align_corners=False)
            patch_resized = patch_resized.squeeze(0)  # (1, C, H, W) -> (C, H, W)
            
            # if patch_resized.shape[1:] != (512, 512):
            #         print(f"Warning: patch shape is {patch_resized.shape}, resizing again.")
            #         patch_resized = F.interpolate(patch.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
            
            #if patch.shape[0] == 1: patch = patch.squeeze(0)[4, :, :]  # 最初のチャンネルを選択
            # if patch_resized.shape[0] == 10: patch_resized = patch_resized[4, :, :]  # 最初のチャンネルを選択
            # plt.figure(figsize=(5, 5))
            # plt.imshow(patch_resized, cmap='gray')
            # plt.axis('off')
            # plt.savefig(f"/Users/markun/git/rsna2024/kaggle/outputs/discs{i}.png",bbox_inches='tight', pad_inches=0)
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