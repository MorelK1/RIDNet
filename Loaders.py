import glob
import cv2
import numpy as np
import copy
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

class NoisingDataset(Dataset):
    def __init__(self, img_dir, train_val, transform):
        super(NoisingDataset, self).__init__()

        self.img_dir = [f for f in glob.glob(img_dir + '/**/*.jpg', recursive=True)]
        self.train_val = train_val
        
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_dir = self.img_dir[idx]

        clean = cv2.imread(img_dir, cv2.IMREAD_COLOR)
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        noisy = np.copy(clean)
        origin_img = copy.deepcopy(clean)

        noisy = self.apply_random_noise(clean)

        data = {'noisy': noisy, 'clean': clean}
        if self.transform:
            data = self.transform(data)

        return data

    def apply_random_noise(self, clean_image):
        noise_type = np.random.choice(['gaussian', 'salt_pepper', 'poisson'])

        if noise_type == 'gaussian':
            noisy_image = self.gaussian_noise(clean_image)
        elif noise_type == 'salt_pepper':
            noisy_image = self.salt_pepper_noise(clean_image)
        else:
            noisy_image = self.poisson_noise(clean_image)

        return noisy_image

    def gaussian_noise(self, img, noise_level=[15, 25, 50]):
        sigma = np.random.choice(noise_level, size=(1, 1, 3))
        sigma *= 3 
        gaussian_noise = np.random.normal(0, sigma, img.shape)

        noisy_img = img + gaussian_noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

        return noisy_img

    def salt_pepper_noise(self, image, amount=0.05):
        noisy_image = np.copy(image)
        num_salt = np.ceil(amount * image.size * 0.5)
        num_pepper = np.ceil(amount * image.size * 0.5)

        # Générer des coordonnées pour le sel
        salt_coords = [np.random.randint(0, d - 1, int(num_salt)) for d in image.shape]
        noisy_image[salt_coords[0], salt_coords[1]] = 1

        # Générer des coordonnées pour le poivre
        pepper_coords = [np.random.randint(0, d - 1, int(num_pepper)) for d in image.shape]
        noisy_image[pepper_coords[0], pepper_coords[1]] = 0

        return noisy_image

    def poisson_noise(self, image, scale=0.005):
        noisy_image = np.random.poisson(image * scale) / scale
        return noisy_image


# Transformers
class ToTensor(object):
    def __call__(self, data):        
        noisy, clean = data['noisy'], data['clean']
        
        noisy = torch.from_numpy(noisy.copy()).type(torch.float32)
        clean = torch.from_numpy(clean.copy()).type(torch.float32)
                
        # (H, W, C) -> (C, H, W)
        noisy = noisy.permute(2, 0, 1)
        clean = clean.permute(2, 0, 1)        

        data = {'noisy': noisy, 'clean': clean}

        return data

class Normalize(object):
    def __call__(self, data):
        noisy, clean = data['noisy'], data['clean']
        
        noisy = noisy / 255.
        clean = clean / 255.        

        data = {'noisy': noisy, 'clean': clean}

        return data


#   Training DataLoader       
train_transform = transforms.Compose([          
                                    Normalize(),
                                    ToTensor()
                                    ])


def get_train_loader(dataset_path, batch_size):
    train_dataset = NoisingDataset(img_dir=dataset_path, train_val='train', transform=train_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    return train_loader


# Validation DataLoader 

val_transform = transforms.Compose([          
                                    Normalize(),
                                    ToTensor()
                                    ])
def get_val_loader(dataset_path, batch_size):
    val_dataset = NoisingDataset(img_dir=dataset_path, train_val='val', transform=val_transform)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    
    return val_loader

