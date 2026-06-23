import torch
import cv2
import numpy as np
import argparse
import sys
import os

from src.model_crnn import CRNN, ctc_decode

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import model_crnn

def preprocess_image(image_path):
    """Preprocess image for inference"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    img = cv2.resize(img, (128, 32))
    img = img.astype(np.float32) / 255.0
    img = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
    return img

def load_model(model_path, device):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model info from checkpoint
    vocab_size = checkpoint['vocab_size']
    charset = checkpoint['charset']
    
    # Load model
    model = CRNN(vocab_size=vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, charset

def predict(model, image, charset, device):
    """Predict text from image"""
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        decoded_indices = ctc_decode(output)
    
    # Convert indices to text
    text = ''.join([charset[idx-1] for idx in decoded_indices if 0 < idx <= len(charset)])
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', default='../checkpoints/model_final.pth', help='Path to trained model')
    parser.add_argument('--device', default='cuda', help='Device to use')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load model and charset
        model, charset = load_model(args.model, device)
        print(f"Model loaded. Vocab size: {len(charset)+1}")
        
        # Preprocess image
        image = preprocess_image(args.image)
        
        # Predict
        result = predict(model, image, charset, device)
        print(f"Predicted text: '{result}'")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()