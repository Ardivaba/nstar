import numpy as np
import torch
from maze_generator import MazeGenerator

class MazeDataset:
    def __init__(self, grid_size=128, num_samples=1000):  # Updated grid size
        self.grid_size = grid_size
        self.num_samples = num_samples
        self.generator = MazeGenerator(grid_size)
        self.maze_types = ['dfs', 'noise', 'walls']
        
    def generate(self, save_path="maze_dataset_128.pt"):  # Updated save path
        # Generate directly on CPU and move to GPU in batches
        dataset = []
        
        for i in range(self.num_samples):
            # Randomly select maze type
            maze_type = np.random.choice(self.maze_types)
            
            if maze_type == 'noise':
                noise_strength = np.random.uniform(0.1, 0.9)
                maze = self.generator._generate_noise_maze(noise_strength)
            else:
                maze = self.generator.generate_maze(maze_type)
            
            # Get random valid positions
            start, end = self.generator.find_random_valid_positions(maze)
            
            # Create input tensor with 3 channels: maze, start, end
            input_tensor = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
            input_tensor[0] = maze  # Maze channel
            input_tensor[1, start[1], start[0]] = 1.0  # Start position
            input_tensor[2, end[1], end[0]] = 1.0  # End position
            
            # Find path
            path_tensor = self.generator.find_path_tensor(maze, start, end)
            
            # Convert to tensors
            dataset.append((
                torch.tensor(input_tensor, dtype=torch.float16),
                path_tensor.to(torch.float16)
            ))
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{self.num_samples} samples")
        
        # Save dataset
        torch.save(dataset, save_path)
        print(f"Dataset saved to {save_path}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    dataset = MazeDataset(grid_size=512)  # Updated grid size
    dataset.generate()