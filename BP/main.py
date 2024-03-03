import torch
from AutoencoderSaverLoader import AutoencoderSaverLoader
from CalculateThreshold import MSEThreshold, SSIMThreshold
from Train import train_model
from AutoEncoder import AutoEncoder
from Test import test
from DataLoader import train_data,test_good_data,test_NOK_data,example
from compare import compare
from test2 import acc, test2


def train():
    # Setup device-agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    torch.cuda.empty_cache()
    autoencoder = AutoEncoder()

    train_dataloader = train_data()
    #example(train_dataloader,"Training dataset")
    print(type(train_dataloader))
    train_model(epochs = 100, model = autoencoder, device = device, 
                               train_loader = train_dataloader, logging_interval=5)

    # save trained model
    autoencoder.to("cpu")
    saver_loader = AutoencoderSaverLoader(model = autoencoder)
    saver_loader.save_model()

def test_model():
    # Setup device-agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    torch.cuda.empty_cache()

    autoencoder = AutoEncoder()
    # Load the model
    saver_loader = AutoencoderSaverLoader(model=autoencoder)
    saver_loader.load_model()
    autoencoder = saver_loader.model
    
    test(model=autoencoder,device = device, test_loader = train_data())


def compare_images():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder = AutoEncoder()
    # Load the model
    saver_loader = AutoencoderSaverLoader(model=autoencoder)
    saver_loader.load_model()
    autoencoder = saver_loader.model
    compare(autoencoder,test_good_data(),device)

def test_accuracy():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder = AutoEncoder()
    # Load the model
    saver_loader = AutoencoderSaverLoader(model=autoencoder)
    saver_loader.load_model()
    autoencoder = saver_loader.model
    
    acc(autoencoder,device,test_good_data(),"good",0.85)
    acc(autoencoder,device,test_NOK_data(),"NOK",0.85)
    
def calculateMSEThreshold():
    # Setup device-agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    torch.cuda.empty_cache()

    autoencoder = AutoEncoder()
    # Load the model
    saver_loader = AutoencoderSaverLoader(model=autoencoder)
    saver_loader.load_model()
    autoencoder = saver_loader.model
    
    MSEThreshold(model=autoencoder,device = device, test_loader = train_data())



def calculateSSIMThreshold():
    # Setup device-agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    torch.cuda.empty_cache()

    autoencoder = AutoEncoder()
    # Load the model
    saver_loader = AutoencoderSaverLoader(model=autoencoder)
    saver_loader.load_model()
    autoencoder = saver_loader.model
    
    SSIMThreshold(model=autoencoder,device = device, test_loader = train_data())




#train()
#test_model()
compare_images()
#test_accuracy()
#calculateMSEThreshold()
#calculateSSIMThreshold()
