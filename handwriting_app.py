import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import sys
import torch
import cv2
import numpy as np

from src.model_crnn import ctc_decode, CRNN

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# cSpell:ignore crnn
import model_crnn

class HandwritingRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwriting Recognition App")
        self.root.geometry("1200x600")
        self.root.configure(bg='#f0f0f0')
        
        # Model setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.charset = None
        self.model_epoch = 'Unknown'
        self.load_model()
        
        # Main title
        title_label = tk.Label(root, text="Handwriting Recognition App", 
                              font=("Arial", 24, "bold"), bg='#f0f0f0', fg='#333')
        title_label.pack(pady=20)
        
        # Main container
        main_frame = tk.Frame(root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=40, pady=20)
        
        # Left panel - Upload Image
        left_frame = tk.Frame(main_frame, bg='white', relief='solid', bd=1, width=500)
        left_frame.pack(side='left', fill='y', padx=(0, 20))
        left_frame.pack_propagate(False)
        
        # Upload Image title
        upload_title = tk.Label(left_frame, text="Upload Image", 
                               font=("Arial", 16, "bold"), bg='white', fg='#333')
        upload_title.pack(pady=20)
        
        # Upload area
        self.upload_frame = tk.Frame(left_frame, bg='#f8f9fa', relief='ridge', bd=2, height=450)
        self.upload_frame.pack(fill='both', expand=True, padx=30, pady=20)
        self.upload_frame.pack_propagate(False)
        
        # Folder icon (using text as placeholder)
        folder_icon = tk.Label(self.upload_frame, text="📁", font=("Arial", 48), 
                              bg='#f8f9fa', fg='#ffa500')
        folder_icon.pack(pady=30)
        
        # Upload instruction
        instruction_label = tk.Label(self.upload_frame, 
                                   text="Click to browse or drag & drop image here",
                                   font=("Arial", 12), bg='#f8f9fa', fg='#666')
        instruction_label.pack()
        
        # Supported formats
        format_label = tk.Label(self.upload_frame, 
                               text="Supports .jpg, .jpeg, .png formats",
                               font=("Arial", 10), bg='#f8f9fa', fg='#999')
        format_label.pack(pady=5)
        
        # Browse button
        browse_btn = tk.Button(left_frame, text="Browse File", 
                              font=("Arial", 12, "bold"), bg='#00BFFF', fg='white',
                              padx=30, pady=12, relief='flat', bd=0,
                              activebackground='#0099CC', cursor='hand2',
                              command=self.browse_file)
        browse_btn.pack(pady=20)
        
        # Right panel - Recognition Result
        right_frame = tk.Frame(main_frame, bg='white', relief='solid', bd=1)
        right_frame.pack(side='right', fill='both', expand=True)
        
        # Recognition Result title
        result_title = tk.Label(right_frame, text="Recognition Result", 
                               font=("Arial", 16, "bold"), bg='white', fg='#333')
        result_title.pack(pady=20)
        
        # Result area
        self.result_frame = tk.Frame(right_frame, bg='#f8f9fa', height=300)
        self.result_frame.pack(fill='both', expand=True, padx=30, pady=20)
        
        # Placeholder text
        placeholder_label = tk.Label(self.result_frame, 
                                   text="Upload an image to see the recognition result",
                                   font=("Arial", 12), bg='#f8f9fa', fg='#999')
        placeholder_label.pack(expand=True)
        
        # Bind click event to upload frame
        self.upload_frame.bind("<Button-1>", lambda e: self.browse_file())
        folder_icon.bind("<Button-1>", lambda e: self.browse_file())
        instruction_label.bind("<Button-1>", lambda e: self.browse_file())
        format_label.bind("<Button-1>", lambda e: self.browse_file())
    
    def load_model(self):
        """Load trained model"""
        model_path = "checkpoints\model_epoch_25.pth" 
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                vocab_size = checkpoint['vocab_size']
                self.charset = checkpoint['charset']
                
                self.model = CRNN(vocab_size=vocab_size)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                # Get epoch info from checkpoint
                epoch = checkpoint.get('epoch', 'final')
                self.model_epoch = epoch
            else:
                pass
        except Exception as e:
            pass
    
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Invert if background is dark
        if np.mean(img) < 127:
            img = 255 - img
            
        # Apply threshold to make text clearer
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        img = cv2.resize(img, (128, 32))
        img = img.astype(np.float32) / 255.0
        img = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
        return img
    
    def predict_text(self, image_path):
        """Predict text from image"""
        if self.model is None or self.charset is None:
            return "Model not loaded"
        
        try:
            image = self.preprocess_image(image_path)
            image = image.to(self.device)
            
            with torch.no_grad():
                output = self.model(image)
                decoded_indices = ctc_decode(output)
            
            # Convert indices to text
            text = ''.join([self.charset[idx-1] for idx in decoded_indices if 0 < idx <= len(self.charset)])
            return text if text else "No text detected"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.display_image(file_path)
            self.process_recognition(file_path)
    
    def display_image(self, file_path):
        try:
            # Clear upload frame
            for widget in self.upload_frame.winfo_children():
                widget.destroy()
            
            # Load and resize image
            image = Image.open(file_path)
            image.thumbnail((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            # Display image
            img_label = tk.Label(self.upload_frame, image=photo, bg='#f8f9fa')
            img_label.image = photo  # Keep a reference
            img_label.pack(pady=10)
            
            # Add file name
            file_name = os.path.basename(file_path)
            name_label = tk.Label(self.upload_frame, text=file_name, 
                                 font=("Arial", 10), bg='#f8f9fa', fg='#333')
            name_label.pack(pady=5)
            
            # Add upload new image button
            new_upload_btn = tk.Button(self.upload_frame, text="Upload New Image", 
                                      font=("Arial", 10, "bold"), bg='#28a745', fg='white',
                                      padx=20, pady=8, relief='flat', bd=0,
                                      activebackground='#218838', cursor='hand2',
                                      command=self.browse_file)
            new_upload_btn.pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")
    
    def process_recognition(self, file_path):
        # Clear result frame
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        # Show loading message
        loading_label = tk.Label(self.result_frame, text="Processing...", 
                                font=("Arial", 12), bg='#f8f9fa', fg='#666')
        loading_label.pack(expand=True)
        self.root.update()
        
        # Get prediction
        predicted_text = self.predict_text(file_path)
        
        # Clear loading message
        loading_label.destroy()
        
        # Show result
        result_text = tk.Text(self.result_frame, font=("Arial", 14), 
                             bg='white', fg='#333', wrap='word')
        result_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        result_content = f"Recognized Text:\n\n{predicted_text}"
        
        result_text.insert('1.0', result_content)
        result_text.config(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingRecognitionApp(root)
    root.mainloop()