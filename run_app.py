#!/usr/bin/env python3
"""
Script để chạy Handwriting Recognition App
"""

import os
import sys
import subprocess

def check_requirements():
    """Kiểm tra và cài đặt requirements"""
    try:
        import torch
        import cv2
        import numpy
        from PIL import Image
        print("✓ All requirements are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing requirement: {e}")
        print("Installing requirements...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✓ Requirements installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to install requirements")
            return False

def check_model():
    """Kiểm tra model file"""
    model_path = "checkpoints/model_epoch_25.pth"
    if os.path.exists(model_path):
        print(f"✓ Model found: {model_path}")
        return True
    else:
        print(f"✗ Model not found: {model_path}")
        print("Please train the model first using: python train.py")
        return False

def main():
    print("=== Handwriting Recognition App ===")
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check model
    if not check_model():
        print("Warning: App will run but recognition won't work without model")
    
    # Run app
    print("Starting app...")
    try:
        from handwriting_app import HandwritingRecognitionApp
        import tkinter as tk
        
        root = tk.Tk()
        app = HandwritingRecognitionApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error running app: {e}")

if __name__ == "__main__":
    main()