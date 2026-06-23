import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from .model_crnn import CRNN, CTCLoss
from .dataset import HandwritingDataset

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for images, targets, target_lengths in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)
        loss = criterion(outputs, targets, input_lengths, target_lengths)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Gradient clipping
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', '--train_labels', required=True, help='Path to training CSV file')
    parser.add_argument('--image_dir', '--train_images', help='Directory containing training images')
    parser.add_argument('--val_labels', help='Path to validation CSV file (optional)')
    parser.add_argument('--val_images', help='Directory containing validation images (optional)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save_dir', default='checkpoints')
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Dataset
    train_csv = args.train_csv
    image_dir = args.image_dir
    dataset = HandwritingDataset(train_csv, image_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                          collate_fn=dataset.collate_fn, num_workers=0)
    
    # Model
    vocab_size = len(dataset.charset) + 1  # +1 for blank
    model = CRNN(vocab_size=vocab_size).to(args.device)
    
    # Loss and optimizer
    criterion = CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    print(f"Training with vocab_size: {vocab_size}")
    print(f"Charset: {dataset.charset}")
    
    # Training loop
    for epoch in range(args.epochs):
        loss = train_epoch(model, dataloader, criterion, optimizer, args.device)
        scheduler.step()
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        if (epoch + 1) % 5 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'charset': dataset.charset,
                'vocab_size': vocab_size
            }, f'{args.save_dir}/model_epoch_{epoch+1}.pth')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': args.epochs-1,
        'loss': loss,
        'charset': dataset.charset,
        'vocab_size': vocab_size
    }, f'{args.save_dir}/model_final.pth')

if __name__ == '__main__':
    main()