import torch.nn.functional as F
import torch.nn as nn
import torch


def test(model,device,test_loader):
    # Set the model to evaluation mode
    model.to(device)
    model.eval()
    
    mse_loss = nn.MSELoss()
    total_test_loss = 0.0

    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data
            inputs = inputs.to(device)

            outputs = model(inputs)
            target_size = outputs.size()[2:]
            inputs = F.interpolate(inputs, size=target_size, mode='bilinear', align_corners=False)

            test_loss = mse_loss(outputs, inputs)  # Mean Squared Error (MSE)

            # total test loss
            total_test_loss += test_loss.item()

    # Calculate the average test loss
    average_test_loss = total_test_loss / len(test_loader)

    #average test loss
    print(f'Average Test Loss: {average_test_loss}')

    

   