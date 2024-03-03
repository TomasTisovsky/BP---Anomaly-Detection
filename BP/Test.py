import torch.nn.functional as F
import torch.nn as nn
import torch


def test(model,device,test_loader):

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
                input = input.to(device)
                output = model(input.unsqueeze(0))
                target_size = input.size()[2:]
                output = F.interpolate(output.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)

                test_loss = mse_loss(output, input)  # Mean Squared Error (MSE)
                print(test_loss.item())
                # total test loss
                total_test_loss += test_loss.item() 
                total_images += 1

    # Calculate the average test loss
    average_test_loss = total_test_loss / total_images

    #average test loss
    print(f'Average Test Loss: {average_test_loss}')

    

   