import torch

class AutoencoderSaverLoader:
    def __init__(self, model, file_path='SavedModels\\autoencoder_model_6.pth'):
        model.to("cpu")

        self.model = model
        self.file_path = file_path

    def save_model(self):
        torch.save(self.model.state_dict(), self.file_path)
        print(f'Model saved to {self.file_path}')

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load(self.file_path))
            print(f'Model loaded from {self.file_path}')
        except FileNotFoundError:
            print(f'Error: File {self.file_path} not found. Make sure to save the model first.')
