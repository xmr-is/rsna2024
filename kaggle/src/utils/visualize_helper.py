import matplotlib.pyplot as plt
import numpy as np

def visualize_prediction(in_image, xy_coord, patch_size=30): 
    # Plot
    for idx in range(1):
    
        # Select Data
        img = (in_image.cpu().numpy() + 1) / 2
        img = np.squeeze(img, 0).transpose(1, 2, 0)  # Transpose for (H, W, C) format
        cs = xy_coord[idx, ...].cpu().numpy() * 384  # Coordinates scaled

        coords_list = [("PRED", "orange", cs)]
        text_labels = [str(x) for x in range(1, 6)]

        # Plot coords on the full image
        # fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        # for title, color, coords in coords_list:
        #     ax.imshow(img, cmap='gray')
        #     ax.scatter(coords[0::2], coords[1::2], c=color, s=50)
        #     ax.axis('off')
        #     ax.set_title(title)

        #     # Add text labels near the coordinates
        #     for i, (x, y) in enumerate(zip(coords[0::2], coords[1::2])):
        #         if i < len(text_labels):  # Ensure there are enough labels
        #             ax.text(x + 10, y, text_labels[i], color='white', fontsize=15, bbox=dict(facecolor='black', alpha=0.5))

        #plt.show()
        
        
        # Extract intervertebral discs and plot them as separate images
        num_discs = len(text_labels)
        fig, axs = plt.subplots(1, num_discs, figsize=(15, 5))

        for i, (x, y) in enumerate(zip(cs[0::2], cs[1::2])):
            # Extract patch around each point
            x, y = int(x), int(y)
            disc_patch = img[max(0, y - patch_size):y + patch_size, max(0, x - patch_size -50):x + patch_size+20]

            # Plot the extracted disc
            axs[i].imshow(disc_patch, cmap='gray')
            axs[i].axis('off')
            axs[i].set_title(f"Disc {i+1}")

        plt.show()
        plt.savefig("discs.png")
#         plt.close(fig)

    return
