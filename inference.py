import pygame
import torch
import numpy as np
from maze_generator import MazeGenerator
from model import PathPredictionModel

class MazeVisualizer:
    def __init__(self, grid_size=128, cell_size=20):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.device = torch.device("cuda")
        self.types = ['dfs', 'noise', 'walls', 'rooms', 'cave']
        self.current_type = 'dfs'
        
        self.model = PathPredictionModel(grid_size).half().to(self.device)
        self.model.load_state_dict(torch.load("maze_model_large.pth", map_location='cuda'))
        self.model.eval()
        
        self.generator = MazeGenerator(grid_size)
        self.maze = self.generator.generate_maze(self.current_type)
        self.start, self.end = self.generator.find_random_valid_positions(self.maze)
        
        pygame.init()
        self.screen = pygame.display.set_mode((grid_size*cell_size, grid_size*cell_size))
        self.font = pygame.font.Font(None, 36)
        pygame.display.set_caption("Maze Pathfinder - AI vs A*")

    def create_input_tensor(self):
        maze_layer = torch.from_numpy(self.maze).float()
        start_layer = torch.zeros_like(maze_layer)
        start_layer[self.start[1], self.start[0]] = 1
        end_layer = torch.zeros_like(maze_layer)
        end_layer[self.end[1], self.end[0]] = 1
        return torch.stack([maze_layer, start_layer, end_layer], dim=0).unsqueeze(0).half().to(self.device)

    def draw_maze(self):
        arr = pygame.surfarray.pixels3d(self.screen)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                color = (255,255,255) if self.maze[y,x]==0 else (0,0,0)
                arr[x*self.cell_size:(x+1)*self.cell_size, y*self.cell_size:(y+1)*self.cell_size] = color
        
        start_color = (0,255,0)
        end_color = (255,0,0)
        sx, sy = self.start
        ex, ey = self.end
        arr[sx*self.cell_size:(sx+1)*self.cell_size, sy*self.cell_size:(sy+1)*self.cell_size] = start_color
        arr[ex*self.cell_size:(ex+1)*self.cell_size, ey*self.cell_size:(ey+1)*self.cell_size] = end_color
        del arr  # Release surface lock
        
        text = self.font.render(f"Type: {self.current_type}", True, (255,255,255))
        self.screen.blit(text, (10, 10))

    def draw_paths(self, pred_path, true_path):
        surface = pygame.Surface((self.grid_size*self.cell_size, self.grid_size*self.cell_size), pygame.SRCALPHA)
        
        # for x,y in true_path:
            # pygame.draw.rect(surface, (0,255,0,200), (x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size))
        
        pred_mask = pred_path > 0.5
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if pred_mask[y,x]:
                    pygame.draw.rect(surface, (255,0,0,150), (x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size))
        
        self.screen.blit(surface, (0,0))

    def handle_click(self, pos, is_start):
        x,y = pos[0]//self.cell_size, pos[1]//self.cell_size
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size and self.maze[y,x] == 0:
            if is_start: self.start = (x,y)
            else: self.end = (x,y)

    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            input_tensor = self.create_input_tensor()
            pred_path = self.model(input_tensor).squeeze().cpu().numpy()
            true_path = self.generator.a_star(self.maze, self.start, self.end)

        while running:
            self.screen.fill((0,0,0))
            self.draw_maze()
            self.draw_paths(pred_path, true_path)
            pygame.display.flip()
            clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: self.handle_click(event.pos, True)
                    elif event.button == 3: self.handle_click(event.pos, False)
                    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                        input_tensor = self.create_input_tensor()
                        pred_path = self.model(input_tensor).squeeze().cpu().numpy()
                        true_path = self.generator.a_star(self.maze, self.start, self.end)
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.maze = self.generator.generate_maze(self.current_type)
                        self.start, self.end = self.generator.find_random_valid_positions(self.maze)
                    elif event.key == pygame.K_RIGHT:
                        idx = self.types.index(self.current_type)
                        self.current_type = self.types[(idx+1)%len(self.types)]
                        self.maze = self.generator.generate_maze(self.current_type)
                        self.start, self.end = self.generator.find_random_valid_positions(self.maze)
                    elif event.key == pygame.K_LEFT:
                        idx = self.types.index(self.current_type)
                        self.current_type = self.types[(idx-1)%len(self.types)]
                        self.maze = self.generator.generate_maze(self.current_type)
                        self.start, self.end = self.generator.find_random_valid_positions(self.maze)
                    
                    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                        input_tensor = self.create_input_tensor()
                        pred_path = self.model(input_tensor).squeeze().cpu().numpy()
                        true_path = self.generator.a_star(self.maze, self.start, self.end)

        pygame.quit()

if __name__ == "__main__":
    visualizer = MazeVisualizer()
    visualizer.run()