import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    """CRNN model hiện đại hơn - CNN + RNN + CTC"""
    def __init__(self, vocab_size, hidden_size=256):
        super().__init__()
        
        # CNN Backbone (ResNet-like)
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x128 -> 16x64
            
            # Block 2  
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x64 -> 8x32
            
            # Block 3
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),  # 8x32 -> 4x32
            
            # Block 4
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),  # 4x32 -> 2x32
            
            # Block 5
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(),  # 2x32 -> 1x31
        )
        
        # RNN
        self.rnn = nn.LSTM(512, hidden_size, 2, bidirectional=True, dropout=0.1, batch_first=False)
        
        # Classifier
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)
        
    def forward(self, x):
        # CNN features
        conv = self.cnn(x)  # [B, 512, 1, W]
        b, c, h, w = conv.size()
        
        # Reshape for RNN: [W, B, C]
        conv = conv.squeeze(2).permute(2, 0, 1)  # [W, B, 512]
        
        # RNN
        output, _ = self.rnn(conv)  # [W, B, hidden*2]
        
        # Classifier
        output = self.classifier(output)  # [W, B, vocab_size]
        return F.log_softmax(output, dim=2)

class CTCLoss(nn.Module):
    """CTC Loss wrapper"""
    def __init__(self, blank=0):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='mean', zero_infinity=True)
        
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

def ctc_decode(log_probs, blank=0):
    """CTC greedy decoding với batch support"""
    if log_probs.dim() == 3:  # [T, B, C]
        # Single sequence
        _, preds = log_probs.max(2)
        preds = preds[:, 0]  # Take first batch
    else:
        _, preds = log_probs.max(1)
    
    # Remove blanks and duplicates
    decoded = []
    prev = blank
    for p in preds:
        p_val = p.item() if hasattr(p, 'item') else p
        if p_val != blank and p_val != prev:
            decoded.append(p_val)
        prev = p_val
    return decoded