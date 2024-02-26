import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

def compare(model,test_dataset,device):
    model.eval()
    model.to(device)

    mse_loss = nn.MSELoss()

    for original_image, _ in test_dataset:
        original_image = original_image.to(device)
        reconstructed_image = model(original_image)

        target_size = original_image.size()[2:]
        reconstructed_image = F.interpolate(reconstructed_image, size=target_size, mode='bilinear', align_corners=False)

        test_loss = mse_loss(reconstructed_image, original_image)
        print("Total loss is:" + str(test_loss.item()))

        # Convert to numpy arrays
        original_image=original_image.to("cpu")
        original_image_np = original_image.squeeze().numpy()
        reconstructed_image = reconstructed_image.to("cpu")
        reconstructed_image_np = reconstructed_image.squeeze().detach().numpy()
        
        
        # Display the original and reconstructed images
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(original_image_np[0], cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title('Reconstructed Image')
        plt.imshow(reconstructed_image_np[0], cmap='gray')

        plt.show()

