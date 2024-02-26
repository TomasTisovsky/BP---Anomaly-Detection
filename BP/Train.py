import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim


def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), int(seconds)


def train_model(epochs, model, device, train_loader,loss_fn=None, logging_interval=5):
    
    torch.manual_seed(42)

    if loss_fn is None:
        loss_fn = nn.MSELoss()

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    for epoch in range(epochs):
        start_epoch_time = time.time()
        model.train()

        for batch_index, (inputs,_)in enumerate(train_loader):   
            inputs = inputs.to(device)
            outputs = model(inputs)
            target_size = outputs.size()[2:]
            inputs = F.interpolate(inputs, size=target_size, mode='bilinear', align_corners=False)

            optimizer.zero_grad() 
            loss =  loss_fn(outputs, inputs)  # Mean Squared Error (MSE) loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
       

        #if batch_index % logging_interval == 0:
        # Print the loss every epoch
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        end_epoch_time = time.time()
        print("Epoch time: " + str(end_epoch_time - start_epoch_time) + " s")

    end_time = time.time()
    total_time = end_time - start_time
    hours, minutes, seconds = seconds_to_hms(total_time)
    print(f"{hours} hours, {minutes} minutes, {seconds} seconds") 


