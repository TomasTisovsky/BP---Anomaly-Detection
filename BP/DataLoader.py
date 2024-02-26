import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.file_list = [file for file in os.listdir(root) if file.endswith('.bmp')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, self.file_list[idx])
        image = Image.open(img_name).convert('L')  # Convert to grayscale
        label = 0

        if self.transform:
            image = self.transform(image)

        return image, label


# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(), 
    transforms.Resize((512, 512)),  # size as needed
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

def train_data():
    # Path to dataset
    TRAIN_GOOD_DATASET_PATH = 'AnomalyDetectionDataset\\train\\good\\'
    TRAIN_GOOD_NEW_DATASET_PATH = 'AnomalyDetectionDataset\\train\\good_new'
    # Load the dataset
    train_good_dataset = CustomDataset(root=TRAIN_GOOD_DATASET_PATH, transform=transform)
    train_good_new_dataset = CustomDataset(root=TRAIN_GOOD_NEW_DATASET_PATH, transform=transform)
    # Concat datasets
    train_dataset = torch.utils.data.ConcatDataset([train_good_dataset,train_good_new_dataset])
    # Create a DataLoader for batching and shuffling
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    return train_dataloader

def test_good_data():
    # Path to dataset
    TEST_GOOD_DATASET_PATH = 'AnomalyDetectionDataset\\test\\good'
    # Load the dataset
    test_good_dataset = CustomDataset(root=TEST_GOOD_DATASET_PATH, transform=transform)
    # Create a DataLoader for batching and shuffling
    test_dataloader = torch.utils.data.DataLoader(test_good_dataset, batch_size=2, shuffle=False)
    return test_dataloader

def test_NOK_data():
    # Path to dataset
    TEST_NOK_DATASET_PATH = 'AnomalyDetectionDataset\\test\\NOK'
    TEST_TESTNOK_DATASET_PATH = 'AnomalyDetectionDataset\\test\\testNOK'
    # Load the dataset
    test_NOK_dataset = CustomDataset(root=TEST_NOK_DATASET_PATH, transform=transform)
    test_testNOK_dataset = CustomDataset(root=TEST_TESTNOK_DATASET_PATH, transform=transform)
    # Concat datasets
    test_NOK_dataset = torch.utils.data.ConcatDataset([test_NOK_dataset,test_testNOK_dataset])
    # Create a DataLoader for batching and shuffling
    test_NOK_dataloader = torch.utils.data.DataLoader(test_NOK_dataset, batch_size=2, shuffle=False)
    return test_NOK_dataloader



def example(dataloader,name = ""):
    print(name)
    print("Number of batches:", len(dataloader))
    print("Total samples:", len(dataloader.dataset))
    print("Batch size:", dataloader.batch_size)

    sample_batch = next(iter(dataloader))
    inputs, labels = sample_batch

    print("\nExample from the Dataloader:")
    print("Data shape:", inputs.shape)
    print("Data type:", inputs.dtype)
    print("Label shape:", labels.shape)
    print("Label type:", labels.dtype)


    import matplotlib.pyplot as plt

    example_image = inputs[0].permute(1, 2, 0).numpy()  # Convert from tensor to NumPy array
    plt.imshow(example_image, cmap='gray')
    plt.title('Example Image from Dataloader')
    plt.show()







