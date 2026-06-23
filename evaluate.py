import torch
import argparse
from torch.utils.data import DataLoader
from src.model_crnn import CRNN, ctc_decode
from src.dataset import HandwritingDataset

def calculate_accuracy(predicted, target):
    """Calculate character-level and word-level accuracy"""
    if predicted == target:
        return 1.0, 1.0  # Perfect match
    
    # Character-level accuracy
    char_correct = sum(1 for p, t in zip(predicted, target) if p == t)
    char_total = max(len(predicted), len(target))
    char_acc = char_correct / char_total if char_total > 0 else 0.0
    
    # Word-level accuracy (exact match)
    word_acc = 1.0 if predicted == target else 0.0
    
    return char_acc, word_acc

def evaluate_model(model, dataloader, charset, device):
    """Evaluate model on dataset"""
    model.eval()
    total_char_acc = 0.0
    total_word_acc = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, targets, target_lengths in dataloader:
            images = images.to(device)
            
            # Get predictions
            outputs = model(images)
            
            # Process each sample in batch
            start_idx = 0
            for i, length in enumerate(target_lengths):
                # Get target text
                target_indices = targets[start_idx:start_idx + length]
                target_text = ''.join([charset[idx-1] for idx in target_indices if 0 < idx <= len(charset)])
                
                # Get prediction for this sample
                sample_output = outputs[:, i:i+1, :]  # [T, 1, C]
                decoded_indices = ctc_decode(sample_output)
                pred_text = ''.join([charset[idx-1] for idx in decoded_indices if 0 < idx <= len(charset)])
                
                # Calculate accuracy
                char_acc, word_acc = calculate_accuracy(pred_text, target_text)
                total_char_acc += char_acc
                total_word_acc += word_acc
                total_samples += 1
                
                # Print some examples
                if total_samples <= 10:
                    print(f"Target: '{target_text}' | Predicted: '{pred_text}' | Char Acc: {char_acc:.2f}")
                
                start_idx += length
    
    avg_char_acc = total_char_acc / total_samples * 100
    avg_word_acc = total_word_acc / total_samples * 100
    
    return avg_char_acc, avg_word_acc, total_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', required=True, help='Test CSV file')
    parser.add_argument('--image_dir', help='Test images directory')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    vocab_size = checkpoint['vocab_size']
    charset = checkpoint['charset']
    
    model = CRNN(vocab_size=vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Model loaded. Vocab size: {vocab_size}")
    print(f"Charset: {charset}")
    
    # Load test dataset
    test_dataset = HandwritingDataset(args.test_csv, args.image_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                collate_fn=test_dataset.collate_fn, num_workers=2)
    
    print(f"\nEvaluating on {len(test_dataset)} samples...")
    
    # Evaluate
    char_acc, word_acc, total_samples = evaluate_model(model, test_dataloader, charset, device)
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Total samples: {total_samples}")
    print(f"Character-level accuracy: {char_acc:.2f}%")
    print(f"Word-level accuracy: {word_acc:.2f}%")

if __name__ == '__main__':
    main()