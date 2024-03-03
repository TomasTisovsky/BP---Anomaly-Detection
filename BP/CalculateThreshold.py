import torch.nn.functional as F
import torch.nn as nn
import torch
from skimage.metrics import structural_similarity as ssim
import numpy as np


def MSEThreshold(model,device,test_loader):
    # Set the model to evaluation mode
    model.to(device)
    model.eval()
    
    mse_loss = nn.MSELoss()
    total_test_loss = 0.0
    total_images = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data
            for input in inputs:
                input = input.unsqueeze(0)
                input = input.to(device)
                output = model(input)
                target_size = input.size()[2:]
                output = F.interpolate(output, size=target_size, mode='bilinear', align_corners=False)

                test_loss = mse_loss(output, input)  # Mean Squared Error (MSE)
                print(test_loss.item())
                # total test loss
                total_test_loss += test_loss.item() 
                total_images += 1

    # Calculate the average test loss
    average_test_loss = total_test_loss / total_images

    #average test loss
    print(f'Average Test Loss: {average_test_loss}')

def SSIMThreshold(model,device,test_loader):
    # Set the model to evaluation mode
    model.to(device)
    model.eval()
    total_ssim = 0
    total_images = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data
            for input in inputs:
                input = input.unsqueeze(0)
                input = input.to(device)
                output = model(input)
                target_size = input.size()[2:]
                output = F.interpolate(output, size=target_size, mode='bilinear', align_corners=False)

                
                # Convert tensors to NumPy arrays
                input_np = input.squeeze(0).cpu().numpy()
                input_np = np.squeeze(input_np)
                output_np = output.squeeze(0).cpu().numpy()
                output_np = np.squeeze(output_np)
                print(input_np.shape)

                # SSIM (Structural simmilarity index) 
                # vypocet SSIM
                ssim_value= ssim(input_np, output_np,data_range=1.0, win_size=7)
                print("SSIM hodnota:", ssim_value)

                total_ssim += ssim_value
                total_images += 1

    # Calculate the average test loss
    average_ssim = total_ssim / total_images

    #average ssim loss
    print(f'Average SSIM: {average_ssim}')
