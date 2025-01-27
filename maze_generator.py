import numpy as np
import torch
from heapq import heappush, heappop
import random

class MazeGenerator:
    def __init__(self, size=41):
        self.size = size
        
    def generate_maze(self, maze_type='dfs'):
        """Generate different types of mazes:
        - 'dfs': Traditional DFS maze
        - 'noise': Random noise pattern
        - 'walls': Random walls/rectangles
        """
        if maze_type == 'dfs':
            return self._generate_dfs_maze()
        elif maze_type == 'noise':
            return self._generate_noise_maze()
        elif maze_type == 'walls':
            return self._generate_wall_maze()
        else:
            return self._generate_dfs_maze()

    def _generate_dfs_maze(self):
        maze = np.ones((self.size, self.size), dtype=np.float32)
        stack = [(0, 0)]
        maze[0, 0] = 0

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = neighbors[np.random.randint(len(neighbors))]
                maze[(y + (ny - y)//2), (x + (nx - x)//2)] = 0
                maze[ny, nx] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _generate_noise_maze(self, noise_strength=0.3):
        maze = np.random.choice([0, 1], size=(self.size, self.size), 
                              p=[1-noise_strength, noise_strength])
        return maze.astype(np.float32)

    def _generate_wall_maze(self):
        maze = np.zeros((self.size, self.size), dtype=np.float32)
        # Add random rectangles
        for _ in range(random.randint(3, 8)):
            x1 = random.randint(0, self.size-1)
            y1 = random.randint(0, self.size-1)
            x2 = min(self.size-1, x1 + random.randint(3, self.size//2))
            y2 = min(self.size-1, y1 + random.randint(3, self.size//2))
            maze[y1:y2, x1:x2] = 1
        return maze

    def _generate_room_maze(self):
        maze = np.ones((self.size, self.size), dtype=np.float32)
        # Create rooms
        rooms = []
        for _ in range(5):
            w = random.randint(5, 8)
            h = random.randint(5, 8)
            x = random.randint(1, self.size - w - 1)
            y = random.randint(1, self.size - h - 1)
            maze[y:y+h, x:x+w] = 0
            rooms.append((x, y, w, h))
        
        # Connect rooms with corridors
        for i in range(len(rooms)-1):
            x1 = rooms[i][0] + random.randint(1, rooms[i][2]-1)
            y1 = rooms[i][1] + random.randint(1, rooms[i][3]-1)
            x2 = rooms[i+1][0] + random.randint(1, rooms[i+1][2]-1)
            y2 = rooms[i+1][1] + random.randint(1, rooms[i+1][3]-1)
            
            while x1 != x2 or y1 != y2:
                if x1 < x2: x1 += 1
                elif x1 > x2: x1 -= 1
                if y1 < y2: y1 += 1
                elif y1 > y2: y1 -= 1
                maze[y1, x1] = 0
        return maze

    def _generate_cave_maze(self):
        # Cellular automaton cave generation
        maze = np.random.choice([0, 1], size=(self.size, self.size), p=[0.4, 0.6])
        
        for _ in range(5):
            new_maze = maze.copy()
            for y in range(1, self.size-1):
                for x in range(1, self.size-1):
                    neighbors = np.sum(maze[y-1:y+2, x-1:x+2])
                    if neighbors >= 5:
                        new_maze[y, x] = 1
                    else:
                        new_maze[y, x] = 0
            maze = new_maze
        return maze.astype(np.float32)

    def find_random_valid_positions(self, maze):
        """Find random start and end positions that are not walls"""
        valid_positions = np.argwhere(maze == 0)
        if len(valid_positions) < 2:
            return (0, 0), (0, 0)
        
        start_idx, end_idx = np.random.choice(len(valid_positions), 2, replace=False)
        
        start_pos = (int(valid_positions[start_idx][1]), int(valid_positions[start_idx][0]))
        end_pos = (int(valid_positions[end_idx][1]), int(valid_positions[end_idx][0]))
        
        return start_pos, end_pos

    def a_star(self, maze, start, end):
        if maze[start[1], start[0]] == 1 or maze[end[1], end[0]] == 1:
            return None  # Invalid positions

        open_heap = []
        heappush(open_heap, (0, start))
        came_from = {}
        g_score = {start: 0}

        while open_heap:
            current = heappop(open_heap)[1]
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor = (current[0]+dx, current[1]+dy)
                if 0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size:
                    if maze[neighbor[1], neighbor[0]] == 1:
                        continue
                    tentative_g = g_score[current] + 1
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + abs(neighbor[0]-end[0]) + abs(neighbor[1]-end[1])
                        heappush(open_heap, (f_score, neighbor))
        return None

    def find_path_tensor(self, maze, start, end):
        path = self.a_star(maze, start, end)
        path_tensor = np.zeros((self.size, self.size), dtype=np.float32)
        if path:
            for (x, y) in path:
                path_tensor[y, x] = 1.0
        return torch.tensor(path_tensor)