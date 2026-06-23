import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
import os

class HandwritingDataset(Dataset):
    def __init__(self, csv_file, image_dir=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        
        # Standard charset
        self.charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.char_to_idx = {char: idx+1 for idx, char in enumerate(self.charset)}  # +1 for blank=0
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['FILENAME']
        label = str(row['IDENTITY'])
        
        # Handle image path
        if self.image_dir:
            image_path = os.path.join(self.image_dir, image_path)
        
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Cannot load image: {image_path}")
            # Return a dummy image to skip this sample
            image = np.ones((32, 128), dtype=np.uint8) * 255
            
        # Preprocess
        image = cv2.resize(image, (128, 32))
        image = image.astype(np.float32) / 255.0
        image = torch.FloatTensor(image).unsqueeze(0)
        
        # Convert label to indices
        target = [self.char_to_idx.get(char, 0) for char in label]
        target = torch.LongTensor(target)
        
        return image, target, len(target)
    
    def collate_fn(self, batch):
        images, targets, target_lengths = zip(*batch)
        
        images = torch.stack(images)
        targets = torch.cat(targets)
        target_lengths = torch.LongTensor(target_lengths)
        
        return images, targets, target_lengths