import torch
import argparse
from torch.utils.data import DataLoader
from src.model_crnn import CRNN, ctc_decode
from src.dataset import HandwritingDataset

def test_model_simple(model, dataloader, charset, device, max_samples=100):
    """Test model and count correct predictions"""
    model.eval()
    correct_predictions = 0
    total_tested = 0
    
    print("Testing samples:")
    print("-" * 60)
    
    with torch.no_grad():
        for images, targets, target_lengths in dataloader:
            if total_tested >= max_samples:
                break
                
            images = images.to(device)
            outputs = model(images)
            
            # Process each sample in batch
            start_idx = 0
            for i, length in enumerate(target_lengths):
                if total_tested >= max_samples:
                    break
                
                # Get target text
                target_indices = targets[start_idx:start_idx + length]
                target_text = ''.join([charset[idx-1] for idx in target_indices if 0 < idx <= len(charset)])
                
                # Get prediction
                sample_output = outputs[:, i:i+1, :]
                decoded_indices = ctc_decode(sample_output)
                pred_text = ''.join([charset[idx-1] for idx in decoded_indices if 0 < idx <= len(charset)])
                
                # Check if correct
                is_correct = (pred_text == target_text)
                if is_correct:
                    correct_predictions += 1
                
                total_tested += 1
                
                # Print result
                status = "✓ CORRECT" if is_correct else "✗ WRONG"
                print(f"{total_tested:3d}. {status} | Target: '{target_text}' | Predicted: '{pred_text}'")
                
                start_idx += length
    
    accuracy = (correct_predictions / total_tested) * 100
    return correct_predictions, total_tested, accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', required=True, help='Test CSV file')
    parser.add_argument('--image_dir', help='Test images directory')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to test')
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
    print(f"Testing {args.num_samples} samples...\n")
    
    # Load test dataset
    test_dataset = HandwritingDataset(args.test_csv, args.image_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                collate_fn=test_dataset.collate_fn, num_workers=2)
    
    # Test model
    correct, total, accuracy = test_model_simple(model, test_dataloader, charset, device, args.num_samples)
    
    print("\n" + "=" * 60)
    print(f"FINAL RESULTS:")
    print(f"Tested: {total} images")
    print(f"Correct: {correct} images")
    print(f"Wrong: {total - correct} images")
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print("=" * 60)

if __name__ == '__main__':
    main()