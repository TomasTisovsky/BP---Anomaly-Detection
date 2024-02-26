import torch.nn.functional as F
import torch.nn as nn



def test2(model,device,test_dataset,type,treshold):
    # Set the model to evaluation mode
    model.to(device)
    model.eval()
    
    mse_loss = nn.MSELoss()

    image_count = 0
    NOK_count = 0
    good_count = 0

    for original_image, _ in test_dataset:
        image_count +=1
        original_image = original_image.to(device)
        reconstructed_image = model(original_image)

        target_size = original_image.size()[2:]
        reconstructed_image = F.interpolate(reconstructed_image, size=target_size, mode='bilinear', align_corners=False)

        test_loss = mse_loss(reconstructed_image, original_image)
        print("Total loss is:" + str(test_loss.item()))
        if test_loss>treshold :
            print("NOK")
            NOK_count +=1
        else:
            print("good")
            good_count +=1

    if type == "NOK":
        print("accuracy:" + str(NOK_count/image_count*100) + "%")

    if type == "good":
        print("accuracy:" + str(good_count/image_count*100) + "%")


