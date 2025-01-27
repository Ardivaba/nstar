import pygame
import numpy as np
from maze_generator import MazeGenerator
import random

class ManualMazeVisualizer:
    def __init__(self, grid_size=128, cell_size=8):  # Use odd grid size!
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.types = ['dfs', 'noise', 'walls']
        self.current_type = 'dfs'  # Start with DFS
        
        self.generator = MazeGenerator(grid_size)
        
        pygame.init()
        self.screen = pygame.display.set_mode(
            (grid_size * cell_size, grid_size * cell_size)
        )
        self.font = pygame.font.Font(None, 36)
        pygame.display.set_caption("Manual Maze Pathfinder (A*)")
        
    def draw_maze(self, maze, start, end):
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                color = (255, 255, 255) if maze[y, x] == 0 else (0, 0, 0)
                pygame.draw.rect(self.screen, color, (
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size - 1,
                    self.cell_size - 1
                ))

        # Draw start and end markers
        self._draw_position(start, (25, 255, 0))  # Start in green
        self._draw_position(end, (255, 25, 0))    # End in red
        
        # Draw status text
        status = f"Current Type: {self.current_type.capitalize()}"
        text = self.font.render(status, True, (0, 0, 0))
        self.screen.blit(text, (10, 10))
        
    def _draw_position(self, pos, color):
        x, y = pos
        pygame.draw.rect(self.screen, color, (
            x * self.cell_size + self.cell_size//4,
            y * self.cell_size + self.cell_size//4,
            self.cell_size//2,
            self.cell_size//2
        ))

    def draw_path(self, path, color):
        for (x, y) in path:
            pygame.draw.rect(self.screen, color, (
                x * self.cell_size + self.cell_size//4,
                y * self.cell_size + self.cell_size//4,
                self.cell_size//2,
                self.cell_size//2
            ))
    
    def run(self):
        # Initialize with current type and generate positions
        maze = self.generator.generate_maze(self.current_type)
        start, end = self._get_random_positions(maze)
        path = self.generator.a_star(maze, start, end)
        
        running = True
        while running:
            self.screen.fill((128, 128, 128))
            
            # Draw maze and path
            self.draw_maze(maze, start, end)
            if path:
                self.draw_path(path, (0, 255, 0))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_UP:
                        # Generate new maze with current type and new positions
                        maze = self.generator.generate_maze(self.current_type)
                        start, end = self._get_random_positions(maze)
                        path = self.generator.a_star(maze, start, end)
                        
                    elif event.key == pygame.K_RIGHT:
                        # Cycle to next maze type
                        idx = self.types.index(self.current_type)
                        self.current_type = self.types[(idx + 1) % len(self.types)]
                        maze = self.generator.generate_maze(self.current_type)
                        start, end = self._get_random_positions(maze)
                        path = self.generator.a_star(maze, start, end)
                        
                    elif event.key == pygame.K_LEFT:
                        # Cycle to previous maze type
                        idx = self.types.index(self.current_type)
                        self.current_type = self.types[(idx - 1) % len(self.types)]
                        maze = self.generator.generate_maze(self.current_type)
                        start, end = self._get_random_positions(maze)
                        path = self.generator.a_star(maze, start, end)
        
        pygame.quit()
    
    def _get_random_positions(self, maze):
        return self.generator.find_random_valid_positions(maze)

if __name__ == "__main__":
    visualizer = ManualMazeVisualizer()
    visualizer.run()