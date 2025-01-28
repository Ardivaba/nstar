import numpy as np
import torch
import os
import time
import gc
from multiprocessing import Pool
from maze_generator import MazeGenerator
from datetime import datetime

class MazeDataset:
    def __init__(self, grid_size=128, num_samples=1000000):
        self.grid_size = grid_size
        self.num_samples = num_samples
        self.maze_types = ['dfs', 'noise', 'walls']
        self.start_time = None
        
    def generate(self, save_path="maze_dataset_128.pt"):
        self.start_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting dataset generation of {self.num_samples} samples")
        
        num_workers = 12
        samples_per_worker = self.num_samples // num_workers
        remaining = self.num_samples % num_workers

        input_file = 'inputs.dat'
        target_file = 'targets.dat'

        inputs_shape = (self.num_samples, 3, self.grid_size, self.grid_size)
        targets_shape = (self.num_samples, self.grid_size, self.grid_size)
        
        inputs_mmap = np.memmap(input_file, dtype=np.float16, mode='w+', shape=inputs_shape)
        targets_mmap = np.memmap(target_file, dtype=np.float16, mode='w+', shape=targets_shape)

        tasks = []
        current_idx = 0
        for i in range(num_workers):
            samples = samples_per_worker + (1 if i < remaining else 0)
            if samples == 0:
                continue
            tasks.append((self.grid_size, self.maze_types, samples, current_idx))
            current_idx += samples

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Using {len(tasks)} workers for {self.num_samples} total samples")

        with Pool(num_workers) as pool:
            results = pool.imap_unordered(generate_batch, tasks)
            completed_samples = 0
            
            for batch_idx, (batch_data, start_idx) in enumerate(results):
                batch_size = len(batch_data)
                completed_samples += batch_size
                
                for i, (input_tensor, target_tensor) in enumerate(batch_data):
                    idx = start_idx + i
                    inputs_mmap[idx] = input_tensor.numpy().astype(np.float16)
                    targets_mmap[idx] = target_tensor.numpy().astype(np.float16)
                
                inputs_mmap.flush()
                targets_mmap.flush()
                
                elapsed = time.time() - self.start_time
                progress = completed_samples / self.num_samples
                remaining_time = elapsed / progress - elapsed if progress > 0 else 0
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch {batch_idx+1}/{len(tasks)} "
                      f"({completed_samples}/{self.num_samples} samples) | "
                      f"Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))} | "
                      f"ETA: {time.strftime('%H:%M:%S', time.gmtime(remaining_time))}")

        final_inputs = torch.from_numpy(np.array(inputs_mmap))
        final_targets = torch.from_numpy(np.array(targets_mmap))

        del inputs_mmap, targets_mmap
        gc.collect()

        torch.save({'inputs': final_inputs, 'targets': final_targets}, save_path)

        try:
            os.remove(input_file)
            os.remove(target_file)
        except:
            pass

        total_time = time.time() - self.start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset saved to {save_path} | "
              f"Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

def generate_batch(args):
    grid_size, maze_types, num_samples, start_idx = args
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    generator = MazeGenerator(grid_size)
    batch = []
    
    worker_start = time.time()
    last_log = worker_start
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Worker {os.getpid()} started: "
          f"{num_samples} samples starting at {start_idx}")
    
    for i in range(num_samples):
        maze_type = np.random.choice(maze_types)
        if maze_type == 'noise':
            noise_strength = np.random.uniform(0.1, 0.9)
            maze = generator._generate_noise_maze(noise_strength)
        else:
            maze = generator.generate_maze(maze_type)
        
        start, end = generator.find_random_valid_positions(maze)
        
        input_tensor = np.zeros((3, grid_size, grid_size), dtype=np.float32)
        input_tensor[0] = maze
        input_tensor[1, start[1], start[0]] = 1.0
        input_tensor[2, end[1], end[0]] = 1.0
        
        path_tensor = generator.find_path_tensor(maze, start, end)
        
        batch.append((torch.tensor(input_tensor, dtype=torch.float16), path_tensor.to(torch.float16)))

        current_time = time.time()
        if current_time - last_log > 30 or i == num_samples-1:
            elapsed = current_time - worker_start
            progress = (i+1)/num_samples
            remaining = (elapsed / progress - elapsed) if progress > 0 else 0
            last_log = current_time
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Worker {os.getpid()} "
                  f"progress: {i+1}/{num_samples} ({progress:.1%}) | "
                  f"Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))} | "
                  f"ETA: {time.strftime('%H:%M:%S', time.gmtime(remaining))}")

    total_time = time.time() - worker_start
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Worker {os.getpid()} completed "
          f"{num_samples} samples in {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    return batch, start_idx

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    dataset = MazeDataset(grid_size=128, num_samples=250000)
    dataset.generate()