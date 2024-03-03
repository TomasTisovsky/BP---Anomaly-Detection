import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim


def test2(model,device,test_dataset,type,threshold):
    # Set the model to evaluation mode
    model.to(device)
    model.eval()
    
    mse_loss = nn.MSELoss()

    image_count = 0
    NOK_count = 0
    good_count = 0

    for inputs, _ in test_dataset:
        for input in inputs:
            input = input.unsqueeze(0)
            image_count +=1
            input = input.to(device)
            reconstructed_image = model(input)

            target_size = input.size()[2:]
            reconstructed_image = F.interpolate(reconstructed_image, size=target_size, mode='bilinear', align_corners=False)

            test_loss = mse_loss(reconstructed_image, input)
            #print("Total loss is:" + str(test_loss.item()))
            if test_loss.item()>threshold :
                #print("NOK")
                NOK_count +=1
            else:
                #print("good")
                good_count +=1

    if type == "NOK":
        print("accuracy:" + str(NOK_count/image_count*100) + "%")

    if type == "good":
        print("accuracy:" + str(good_count/image_count*100) + "%")


def acc(model,device,test_dataset,type,threshold):
    # Set the model to evaluation mode
    model.to(device)
    model.eval()
    total_ssim = 0
    

    image_count = 0
    NOK_count = 0
    good_count = 0

    for inputs, _ in test_dataset:
        for input in inputs:
            input = input.unsqueeze(0)
            image_count +=1
            input = input.to(device)
            reconstructed_image = model(input)

            target_size = input.size()[2:]
            reconstructed_image = F.interpolate(reconstructed_image, size=target_size, mode='bilinear', align_corners=False)

            # Convert tensors to NumPy arrays
            input_np = input.squeeze(0).cpu().numpy()
            input_np = np.squeeze(input_np)
            output_np = reconstructed_image.detach().cpu().numpy()
            output_np = np.squeeze(output_np)
            

            # SSIM (Structural simmilarity index) 
            # vypocet SSIM
            ssim_value= ssim(input_np, output_np,data_range=1.0, win_size=7)
            print("SSIM hodnota:", ssim_value)
            if ssim_value<threshold :
                #print("NOK")
                NOK_count +=1
            else:
                #print("good")
                good_count +=1

    if type == "NOK":
        print("accuracy:" + str(NOK_count/image_count*100) + "%")

    if type == "good":
        print("accuracy:" + str(good_count/image_count*100) + "%")

