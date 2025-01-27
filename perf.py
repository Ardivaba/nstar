import torch
import numpy as np
import time
import multiprocessing as mp
from maze_generator import MazeGenerator
from model import PathPredictionModel
from concurrent.futures import ProcessPoolExecutor

def process_maze_astar(args):
    """Run A* on a single maze in a separate process"""
    generator, grid_size = args
    maze = generator.generate_maze()
    start_time = time.perf_counter()
    _ = generator.a_star(maze, (0, 0), (grid_size-1, grid_size-1))
    end_time = time.perf_counter()
    return end_time - start_time, maze

def generate_maze_only(generator):
    """Just generate a maze for AI testing"""
    return generator.generate_maze()

def run_performance_test(grid_size=41, num_mazes=10000):
    print("Initializing performance test...", flush=True)
    
    # Calculate optimal batch size for RTX 4090 (24GB VRAM)
    batch_size = 10240
    
    # Initialize components
    print("Creating maze generator...", flush=True)
    generator = MazeGenerator(grid_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_cores = mp.cpu_count()
    print(f"Using device: {device}", flush=True)
    print(f"Number of CPU cores available: {num_cores}", flush=True)
    print(f"Batch size for GPU: {batch_size}", flush=True)
    
    # Load model first
    print("\nLoading AI model...", flush=True)
    try:
        model = PathPredictionModel(grid_size).half().to(device)
        model.load_state_dict(torch.load("maze_model_large.pth", map_location=device))
        model.eval()
        print("Model loaded successfully!", flush=True)
    except Exception as e:
        print(f"Error loading model: {e}", flush=True)
        return
    
    # Warm up CUDA
    print("\nWarming up CUDA with batch processing...", flush=True)
    dummy_batch = torch.zeros((batch_size, grid_size, grid_size), device=device, dtype=torch.float16)
    for i in range(5):
        print(f"Warmup batch {i+1}/5", flush=True)
        with torch.no_grad(), torch.cuda.amp.autocast():
            model(dummy_batch)
    
    print("\nStarting parallel A* pathfinding test...", flush=True)
    a_star_times = []
    
    # Create arguments list for parallel processing
    args_list = [(generator, grid_size) for _ in range(num_mazes)]
    
    # Run A* tests in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        completed = 0
        for time_taken, maze in executor.map(process_maze_astar, args_list):
            a_star_times.append(time_taken)
            completed += 1
            if completed % 10 == 0:
                print(f"Completed {completed}/{num_mazes} A* mazes", flush=True)
    
    print("\nGenerating mazes for AI testing...", flush=True)
    # Generate all mazes first using parallel processing
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        mazes = list(executor.map(generate_maze_only, [generator] * num_mazes))
    
    print("\nStarting batched AI prediction test...", flush=True)
    ai_times = []
    
    # Process AI predictions in batches
    for i in range(0, num_mazes, batch_size):
        batch_end = min(i + batch_size, num_mazes)
        batch_mazes = mazes[i:batch_end]
        
        # Convert batch to tensor
        maze_tensor = torch.tensor(batch_mazes, device=device, dtype=torch.float16)
        
        # Time the batch prediction
        start_time = time.perf_counter()
        with torch.no_grad(), torch.cuda.amp.autocast():
            _ = model(maze_tensor)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # Record time per maze in batch
        batch_time = (end_time - start_time) / len(batch_mazes)
        ai_times.extend([batch_time] * len(batch_mazes))
        
        print(f"Completed AI predictions {batch_end}/{num_mazes}", flush=True)
    
    if not a_star_times or not ai_times:
        print("No timing data collected!", flush=True)
        return
        
    # Calculate and print statistics
    a_star_avg = np.mean(a_star_times) * 1000  # Convert to milliseconds
    a_star_std = np.std(a_star_times) * 1000
    ai_avg = np.mean(ai_times) * 1000
    ai_std = np.std(ai_times) * 1000
    
    print("\nResults:", flush=True)
    print(f"A* Average Time: {a_star_avg:.2f}ms (±{a_star_std:.2f}ms)", flush=True)
    print(f"AI Average Time: {ai_avg:.2f}ms (±{ai_std:.2f}ms)", flush=True)
    print(f"Speed Difference: AI is {a_star_avg/ai_avg:.1f}x faster than A*", flush=True)
    
    # Save detailed results to file
    print("\nSaving detailed results to performance_results.txt...", flush=True)
    try:
        with open("performance_results.txt", "w") as f:
            f.write("Performance Test Results\n")
            f.write("======================\n\n")
            f.write(f"Grid Size: {grid_size}x{grid_size}\n")
            f.write(f"Number of mazes tested: {num_mazes}\n")
            f.write(f"CPU Cores Used: {num_cores}\n")
            f.write(f"GPU Batch Size: {batch_size}\n\n")
            f.write("A* Pathfinding:\n")
            f.write(f"- Average: {a_star_avg:.2f}ms\n")
            f.write(f"- Standard Deviation: {a_star_std:.2f}ms\n")
            f.write(f"- Min: {min(a_star_times)*1000:.2f}ms\n")
            f.write(f"- Max: {max(a_star_times)*1000:.2f}ms\n\n")
            f.write("AI Prediction:\n")
            f.write(f"- Average: {ai_avg:.2f}ms\n")
            f.write(f"- Standard Deviation: {ai_std:.2f}ms\n")
            f.write(f"- Min: {min(ai_times)*1000:.2f}ms\n")
            f.write(f"- Max: {max(ai_times)*1000:.2f}ms\n\n")
            f.write(f"Speed Comparison: AI is {a_star_avg/ai_avg:.1f}x faster than A*\n")
            f.write(f"Total throughput: {num_mazes/max(sum(a_star_times), sum(ai_times)):.1f} mazes/second\n")
        print("Results saved successfully!", flush=True)
    except Exception as e:
        print(f"Error saving results: {e}", flush=True)

if __name__ == "__main__":
    mp.freeze_support()
    print("Starting performance test script...", flush=True)
    run_performance_test()
    print("Performance test completed!", flush=True)