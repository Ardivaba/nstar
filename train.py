import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from model import PathPredictionModel  # Updated model

class MazeTrainer:
    def __init__(self, grid_size=128):  # Updated grid size
        self.device = torch.device("cuda")
        self.grid_size = grid_size
        
        # Model setup with 3 input channels
        self.model = PathPredictionModel(grid_size).to(self.device)
        self.scaler = GradScaler()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2e-4)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Checkpoint handling
        self.checkpoint_path = "maze_128_checkpoint.pth"  # Updated checkpoint path
        if os.path.exists(self.checkpoint_path):
            self._load_checkpoint()

    def _load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded checkpoint")

    def _save_checkpoint(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.checkpoint_path)

    def load_data(self, path="maze_dataset_128.pt"):  # Updated dataset path
        full_dataset = torch.load(path)
        return random_split(full_dataset, [int(0.95*len(full_dataset)), 
                                         len(full_dataset)-int(0.95*len(full_dataset))])

    def train(self, epochs=50, batch_size=32):  # Reduced batch size for 512x512
        train_set, val_set = self.load_data()
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, pin_memory=True)

        scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-3,
                                                steps_per_epoch=len(train_loader),
                                                epochs=epochs)

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                train_loss += loss.item()
                scheduler.step()

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                    val_loss += self.criterion(outputs, targets).item()

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val Loss: {val_loss/len(val_loader):.4f}")
            
            self._save_checkpoint()

        torch.save(self.model.state_dict(), "maze_model_128.pth")  # Updated model name

if __name__ == "__main__":
    trainer = MazeTrainer()
    trainer.train(epochs=100, batch_size=16)