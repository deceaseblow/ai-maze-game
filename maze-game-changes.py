import pygame
import time
import heapq
from collections import deque
import random
import math
from PIL import Image, ImageSequence 
import json
import os
import csv
from datetime import datetime

# Updated Color Palette (more vibrant and visually appealing)
WHITE = (255, 255, 255)
BLACK = (30, 30, 30)
RED = (220, 50, 50)
GREEN = (50, 205, 50)
BLUE = (30, 144, 255)
YELLOW = (255, 215, 0)
PURPLE = (147, 112, 219)
ORANGE = (255, 140, 0)
CYAN = (0, 206, 209)
PINK = (255, 105, 180)
GRAY = (128, 128, 128)
LIGHT_GRAY = (211, 211, 211)
BACKGROUND = (245, 245, 245)
GRID_LINES = (220, 220, 220)

# New colors for path gradient
PATH_GRADIENT_START = PURPLE 
PATH_GRADIENT_END =RED    

class MazePathfinder:
    def __init__(self, width=800, height=600, cell_size=20):
        # Initialize pygame
        self.maze_offset_x = 0
        self.maze_offset_y = 120 
        pygame.init()
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Maze Pathfinding Visualizer: BFS, DFS, A*")
        self.clock = pygame.time.Clock()
        
        pygame.font.init()
        self.font_small = pygame.font.SysFont("Arial", 16)
        self.font_medium = pygame.font.SysFont("Arial", 24)
        self.font_large = pygame.font.SysFont("Arial", 36)
        self.font_title = pygame.font.SysFont("Arial", 42, bold=True)
        
        # MAZE VISIBILITY
        available_height = height - self.maze_offset_y - 70 
        self.grid_width = (width - 2 * self.maze_offset_x) // cell_size
        self.grid_height = available_height // cell_size
        
        self.maze = [[0 for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        
        # Make sure the start and goal positions are within the adjusted grid dimensions
        
        self.start = (1, 1)
        self.goal = (min(self.grid_height - 2, self.grid_height - 2), 
                    min(self.grid_width - 2, self.grid_width - 2))
        
        # Visual effect settings
        self.cell_padding = 1  # Padding inside cells for a cleaner look
        self.animation_speed = 0.05  # Default animation speed
        
        # Generate random maze
        self.generate_random_maze(obstacle_density=0.3)
        
        # Ensure start and goal positions are open
        self.maze[self.start[0]][self.start[1]] = 0
        self.maze[self.goal[0]][self.goal[1]] = 0
        
        # Tracking variables
        self.visited = set()
        self.path = []
        self.steps = 0
        self.execution_time = 0
        
        # BFS A* DFS COLORS SETTINGS
        self.path_colors = {
            'BFS': CYAN,
            'DFS': PURPLE,
            'A*': ORANGE
        }
        self.visited_colors = {
            'BFS': (100, 200, 230),
            'DFS': (200, 150, 220),  
            'A*': (255, 200, 150) 
        }
        
        self.current_algorithm = 'BFS'  # Default algorithm
        
        # Direction vectors (up, right, down, left)
        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # RESULTS DATA
        self.results = []
        self.show_results = False
        
        # Path animation variables
        self.animate_path = True
        self.path_animation_progress = 0
        self.path_animation_speed = 0.02
        self.animate_visited = True
        
        self.running_algorithm = False  # Flag to track when an algorithm is running
        self.stop_requested = False
        
        # BUTTONS
        self.buttons = []
        self.create_buttons()
        self.start_gif_frames = []
        self.goal_gif_frames = []
        self.current_frame_index = 0
        self.last_frame_update = pygame.time.get_ticks()
        self.frame_delay = 100  # milliseconds between frames
        
        # Load GIF animations
        self.load_gifs()
    
    def create_buttons(self):
        """Create buttons for the main interface"""
        button_width = 80
        button_height = 40
        start_x = 20
        start_y = self.height - 60
        spacing = button_width + 20
        
        self.buttons = [
            {'rect': pygame.Rect(start_x, start_y, button_width, button_height),
            'text': 'BFS', 'action': 'bfs'},
            {'rect': pygame.Rect(start_x + spacing, start_y, button_width, button_height),
            'text': 'DFS', 'action': 'dfs'},
            {'rect': pygame.Rect(start_x + spacing * 2, start_y, button_width, button_height),
            'text': 'A*', 'action': 'astar'},
            {'rect': pygame.Rect(start_x + spacing * 3, start_y, button_width, button_height),
            'text': 'Compare All', 'action': 'compare'},
            {'rect': pygame.Rect(start_x + spacing * 4, start_y, button_width, button_height),
            'text': 'New Maze', 'action': 'reset'},
            {'rect': pygame.Rect(start_x + spacing * 5, start_y, button_width, button_height),
            'text': 'Stop', 'action': 'stop'},
            {'rect': pygame.Rect(start_x + spacing * 6, start_y, button_width, button_height),
            'text': 'Save Data', 'action': 'save'},
        ]
    
    def update_animation(self):
        """Update the animation frame index based on elapsed time"""
        current_time = pygame.time.get_ticks()
        if current_time - self.last_frame_update > self.frame_delay:
            self.current_frame_index += 1
            if self.current_frame_index >= len(self.start_gif_frames):
                self.current_frame_index = 0
            if self.current_frame_index >= len(self.goal_gif_frames):
                self.current_frame_index = 0
            self.last_frame_update = current_time
    
    
    def load_gifs(self):
        """Load and prepare the GIF images for start and goal"""
        try:
            # Load start GIF
            start_gif = Image.open("cat.gif")
            for frame in ImageSequence.Iterator(start_gif):
                # Convert PIL image to Pygame surface
                frame = frame.convert("RGBA")
                frame_data = frame.tobytes()
                size = frame.size
                mode = frame.mode
                
                # Create pygame surface from PIL image data
                py_image = pygame.image.fromstring(frame_data, size, mode)
                
                # Scale the image to fit the cell size
                py_image = pygame.transform.scale(py_image, (self.cell_size - 2 * self.cell_padding, 
                                                         self.cell_size - 2 * self.cell_padding))
                
                self.start_gif_frames.append(py_image)
            
            # Load goal GIF with the same process
            goal_gif = Image.open("cheese.gif")
            for frame in ImageSequence.Iterator(goal_gif):
                frame = frame.convert("RGBA")
                frame_data = frame.tobytes()
                size = frame.size
                mode = frame.mode
                
                py_image = pygame.image.fromstring(frame_data, size, mode)
                py_image = pygame.transform.scale(py_image, (self.cell_size - 2 * self.cell_padding, 
                                                         self.cell_size - 2 * self.cell_padding))
                
                self.goal_gif_frames.append(py_image)
                
            print(f"GIFs loaded successfully. Start: {len(self.start_gif_frames)} frames, Goal: {len(self.goal_gif_frames)} frames")
            
        except Exception as e:
            print(f"Error loading GIFs: {e}")
            # Create fallback colored squares if GIFs can't be loaded
            surface = pygame.Surface((self.cell_size - 2 * self.cell_padding, self.cell_size - 2 * self.cell_padding))
            surface.fill(GREEN)
            self.start_gif_frames = [surface]
            
            surface = pygame.Surface((self.cell_size - 2 * self.cell_padding, self.cell_size - 2 * self.cell_padding))
            surface.fill(RED)
            self.goal_gif_frames = [surface]
    
    
    def generate_random_maze(self, obstacle_density=0.3):
        """Generate a random maze with guaranteed path from start to goal"""
        # Define directions here if not available yet
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
        
        # First, create a clear maze with only border walls
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if i == 0 or i == self.grid_height - 1 or j == 0 or j == self.grid_width - 1:
                    self.maze[i][j] = 1  # Border walls
                else:
                    self.maze[i][j] = 0  # Clear all cells initially
        
        # Ensure start and goal are clear
        self.maze[self.start[0]][self.start[1]] = 0
        self.maze[self.goal[0]][self.goal[1]] = 0
        
        # Use a modified DFS to create a guaranteed path from start to goal
        def create_path_dfs():
            path = []
            visited = set()
            stack = [self.start]
            came_from = {self.start: None}
            visited.add(self.start)
            
            while stack:
                current = stack.pop()
                
                if current == self.goal:
                    # Reconstruct path
                    while current != self.start:
                        path.append(current)
                        current = came_from[current]
                    path.append(self.start)
                    return path
                
                # Randomize direction exploration for more interesting paths
                local_directions = list(directions)  # Use local variable instead of self.directions
                random.shuffle(local_directions)
                
                for dx, dy in local_directions:
                    next_x, next_y = current[0] + dx, current[1] + dy
                    next_pos = (next_x, next_y)
                    
                    if (0 < next_x < self.grid_height - 1 and 
                        0 < next_y < self.grid_width - 1 and 
                        next_pos not in visited):
                        stack.append(next_pos)
                        visited.add(next_pos)
                        came_from[next_pos] = current
            
            return []  # No path found (shouldn't happen with our grid)
        
        # Get guaranteed path
        solution_path = create_path_dfs()
        solution_path_set = set(solution_path)
        
        # Add random obstacles, but avoid the solution path
        obstacle_candidates = []
        for i in range(1, self.grid_height - 1):
            for j in range(1, self.grid_width - 1):
                if (i, j) not in solution_path_set and (i, j) != self.start and (i, j) != self.goal:
                    obstacle_candidates.append((i, j))
        
        # Randomly select cells to be obstacles based on density
        num_obstacles = int(len(obstacle_candidates) * obstacle_density)
        if obstacle_candidates and num_obstacles > 0:
            obstacle_positions = random.sample(obstacle_candidates, min(num_obstacles, len(obstacle_candidates)))
            
            for pos in obstacle_positions:
                self.maze[pos[0]][pos[1]] = 1
        
        # Add some random obstacles near (but not on) the path for more challenge
        # Get cells adjacent to path but not on path
        path_adjacent = set()
        for pos in solution_path:
            for dx, dy in directions:  # Use local directions variable
                adj_pos = (pos[0] + dx, pos[1] + dy)
                if (0 < adj_pos[0] < self.grid_height - 1 and 
                    0 < adj_pos[1] < self.grid_width - 1 and
                    adj_pos not in solution_path_set and
                    adj_pos != self.start and adj_pos != self.goal):
                    path_adjacent.add(adj_pos)
        
        # Add some obstacles near the path (but leave it solvable)
        path_adj_list = list(path_adjacent)
        if path_adj_list:
            random.shuffle(path_adj_list)
            for pos in path_adj_list[:max(1, len(path_adj_list) // 3)]:  # Use about 1/3 of adjacent cells
                self.maze[pos[0]][pos[1]] = 1
            
        # Double-check maze is still solvable with BFS
        def is_solvable():
            queue = deque([self.start])
            visited = {self.start}
            
            while queue:
                current = queue.popleft()
                
                if current == self.goal:
                    return True
                    
                for dx, dy in directions:  # Use local directions variable
                    next_x, next_y = current[0] + dx, current[1] + dy
                    next_pos = (next_x, next_y)
                    
                    if (0 <= next_x < self.grid_height and 
                        0 <= next_y < self.grid_width and 
                        self.maze[next_x][next_y] == 0 and 
                        next_pos not in visited):
                        queue.append(next_pos)
                        visited.add(next_pos)
            
            return False
        
        # If somehow we made it unsolvable, clear some obstacles
        if not is_solvable():
            # Clear obstacles along the original solution path
            for pos in solution_path:
                self.maze[pos[0]][pos[1]] = 0
                
        # Ensure start and goal positions are definitely open
        self.maze[self.start[0]][self.start[1]] = 0
        self.maze[self.goal[0]][self.goal[1]] = 0
        
    
    def is_valid(self, x, y):
        """Check if a position is valid (within bounds and not an obstacle)"""
        return (0 <= x < self.grid_height and 
                0 <= y < self.grid_width and 
                self.maze[x][y] == 0)
    
    def interpolate_color(self, color1, color2, t):
        """Interpolate between two colors"""
        return (
            int(color1[0] + (color2[0] - color1[0]) * t),
            int(color1[1] + (color2[1] - color1[1]) * t),
            int(color1[2] + (color2[2] - color1[2]) * t)
        )
    
    def draw_rounded_rect(self, surface, rect, color, radius=0.4):
        """Draw a rounded rectangle"""
        rect = pygame.Rect(rect)
        
        # If the radius is too large for the rect, reduce it
        radius = min(radius, rect.width / 2, rect.height / 2)
        
        if radius <= 0:
            pygame.draw.rect(surface, color, rect)
            return
            
        # Draw the main rectangle minus the corners
        pygame.draw.rect(surface, color, rect.inflate(-radius * 2, 0))
        pygame.draw.rect(surface, color, rect.inflate(0, -radius * 2))
        
        # Draw the four rounded corners
        circle_diameter = radius * 2
        circle_rect = pygame.Rect(0, 0, circle_diameter, circle_diameter)
        
        # Top left corner
        circle_rect.topleft = rect.topleft
        pygame.draw.ellipse(surface, color, circle_rect)
        
        # Top right corner
        circle_rect.topright = rect.topright
        pygame.draw.ellipse(surface, color, circle_rect)
        
        # Bottom left corner
        circle_rect.bottomleft = rect.bottomleft
        pygame.draw.ellipse(surface, color, circle_rect)
        
        # Bottom right corner
        circle_rect.bottomright = rect.bottomright
        pygame.draw.ellipse(surface, color, circle_rect)
    
    # Update in __init__ method to include offsets for the maze positioning

    # Update the draw_cell method to include these offsets
    def draw_cell(self, i, j, color):
        """Draw a single cell with padding for a cleaner look"""
        outer_rect = pygame.Rect(
            self.maze_offset_x + j * self.cell_size, 
            self.maze_offset_y + i * self.cell_size, 
            self.cell_size, 
            self.cell_size
        )
        
        inner_rect = pygame.Rect(
            self.maze_offset_x + j * self.cell_size + self.cell_padding, 
            self.maze_offset_y + i * self.cell_size + self.cell_padding, 
            self.cell_size - 2 * self.cell_padding, 
            self.cell_size - 2 * self.cell_padding
        )
        
        # Draw outer cell border
        pygame.draw.rect(self.screen, GRID_LINES, outer_rect)
        
        # Draw inner cell with rounded corners
        self.draw_rounded_rect(self.screen, inner_rect, color, radius=3)
        
    def draw_path_with_gradient(self):
        """Draw the path with a gradient effect to show direction"""
        if not self.path:
            return
            
        # Draw each segment with a gradient color
        for i, pos in enumerate(self.path):
            if i > self.path_animation_progress and self.animate_path:
                break
                
            # Calculate gradient position
            t = i / max(1, len(self.path) - 1)
            color = self.interpolate_color(PATH_GRADIENT_START, PATH_GRADIENT_END, t)
            
            self.draw_cell(pos[0], pos[1], color)      
            
    
    def draw_visited_cells_with_animation(self):
        """Draw visited cells with animation effect"""
        visited_list = list(self.visited)
        visited_color = self.visited_colors[self.current_algorithm]
        
        # When algorithm is running, always show ALL visited cells
        if self.running_algorithm:
            cells_to_show = len(visited_list)  # Show all cells during algorithm execution
        # During animation playback (not running), respect animation settings
        elif self.animate_visited:
            cells_to_show = int(len(visited_list) * min(1.0, self.path_animation_progress * 2))
        else:
            cells_to_show = len(visited_list)  # If not animating, show all cells
        
        # Draw cells up to the calculated limit
        for i, pos in enumerate(visited_list[:cells_to_show]):
            if pos not in self.path and pos != self.start and pos != self.goal:
                # Slight color variation based on distance from start
                dist = abs(pos[0] - self.start[0]) + abs(pos[1] - self.start[1])
                factor = min(1.0, dist / 20)  # Normalize distance
                color = self.interpolate_color(
                    visited_color,
                    (max(0, visited_color[0] - 30), 
                    max(0, visited_color[1] - 30),
                    max(0, visited_color[2] - 30)),
                    factor
                )
                
                self.draw_cell(pos[0], pos[1], color)
    
    def draw_path_with_gradient(self):
        """Draw the path with a gradient effect to show direction - FIXED"""
        if not self.path:
            return
            
        # Calculate how many path segments to show based on animation progress
        cells_to_show = len(self.path)
        if self.animate_path:
            cells_to_show = int(min(len(self.path), self.path_animation_progress * len(self.path)))
        
        # Draw each segment with a gradient color
        for i, pos in enumerate(self.path[:cells_to_show]):
            # Calculate gradient position
            t = i / max(1, len(self.path) - 1)
            color = self.interpolate_color(PATH_GRADIENT_START, PATH_GRADIENT_END, t)
            
            self.draw_cell(pos[0], pos[1], color)
            
    def draw_maze(self):
        """Draw the maze with proper layering - Modified to use GIFs for start/goal"""
        if self.show_results:
            self.draw_results_page()
            return
        
        self.screen.fill(BACKGROUND)
        
        # Draw grid background and walls first
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if self.maze[i][j] == 1:  # Wall
                    self.draw_cell(i, j, BLACK)
                else:  # Empty cell
                    self.draw_cell(i, j, WHITE)
        
        # Update animation frame
        self.update_animation()
        
        # Then draw visited cells
        self.draw_visited_cells_with_animation()
        
        # Then draw path over visited cells
        self.draw_path_with_gradient()
        
        # Finally draw start and goal positions with GIFs
        # Draw background cells for start and goal
        self.draw_cell(self.start[0], self.start[1], WHITE)
        self.draw_cell(self.goal[0], self.goal[1], ORANGE)
        
        # Draw GIF frames for start
        start_frame_index = min(self.current_frame_index, len(self.start_gif_frames) - 1)
        start_frame = self.start_gif_frames[start_frame_index]
        start_rect = pygame.Rect(
            self.maze_offset_x + self.start[1] * self.cell_size + self.cell_padding,
            self.maze_offset_y + self.start[0] * self.cell_size + self.cell_padding,
            self.cell_size - 2 * self.cell_padding,
            self.cell_size - 2 * self.cell_padding
        )
        self.screen.blit(start_frame, start_rect)
        
        # Draw GIF frames for goal
        goal_frame_index = min(self.current_frame_index, len(self.goal_gif_frames) - 1)
        goal_frame = self.goal_gif_frames[goal_frame_index]
        goal_rect = pygame.Rect(
            self.maze_offset_x + self.goal[1] * self.cell_size + self.cell_padding,
            self.maze_offset_y + self.goal[0] * self.cell_size + self.cell_padding,
            self.cell_size - 2 * self.cell_padding,
            self.cell_size - 2 * self.cell_padding
        )
        self.screen.blit(goal_frame, goal_rect)
        
        # Draw panel background
        info_panel = pygame.Rect(0, 0, self.width, 110)
        pygame.draw.rect(self.screen, (240, 240, 240), info_panel)
        pygame.draw.line(self.screen, GRAY, (0, 110), (self.width, 110), 1)
        
        # Draw title
        title = self.font_large.render("Maze Pathfinding Visualizer", True, BLACK)
        self.screen.blit(title, (10, 10))
        
        # Display information
        info_text = f"Algorithm: {self.current_algorithm}   Steps: {self.steps}   Time: {self.execution_time:.4f} s"
        text_surface = self.font_medium.render(info_text, True, BLACK)
        self.screen.blit(text_surface, (15, 43))
        
        # Display path length if available
        if self.path:
            path_text = f"Path Length: {len(self.path) - 1}"
            path_surface = self.font_medium.render(path_text, True, BLACK)
            self.screen.blit(path_surface, (20, 62 ))
        
        # Display legend
        legend_x = 20  # Starting x position
        legend_y = 90  # Fixed y position for all items
        legend_spacing = 80  # Horizontal spacing between legend items

        legend_items = [
            ("Start", WHITE),
            ("Goal", ORANGE),
            ("Wall", BLACK),
            ("BFS", BLUE),
            ("DFS", PURPLE),
            ("A*", ORANGE),
            ("Path", PATH_GRADIENT_START),
            ("Path Gradient", PATH_GRADIENT_END)
        ]

        for text, color in legend_items:
            # Draw color box
            color_rect = pygame.Rect(legend_x, legend_y, 15, 15)
            pygame.draw.rect(self.screen, color, color_rect)
            pygame.draw.rect(self.screen, BLACK, color_rect, 1)

            # Draw text next to the color box
            label = self.font_small.render(text, True, BLACK)
            self.screen.blit(label, (legend_x + 20, legend_y))

            # Move to the next horizontal position
            legend_x += legend_spacing
            
            
        # Draw bottom panel with buttons
        button_panel = pygame.Rect(0, self.height - 70, self.width, 70)
        pygame.draw.rect(self.screen, (240, 240, 240), button_panel)
        pygame.draw.line(self.screen, GRAY, (0, self.height - 70), (self.width, self.height - 70), 1)
        
        # Draw buttons
        for button in self.buttons:
            hover = button['rect'].collidepoint(pygame.mouse.get_pos())
            
            # Button background
            if hover:
                pygame.draw.rect(self.screen, LIGHT_GRAY, button['rect'])
            else:
                pygame.draw.rect(self.screen, WHITE, button['rect'])
                
            # Button border
            pygame.draw.rect(self.screen, GRAY, button['rect'], 1)
            
            # Button text
            button_text = self.font_medium.render(button['text'], True, BLACK)
            button_text_rect = button_text.get_rect(center=button['rect'].center)
            self.screen.blit(button_text, button_text_rect)
        
        pygame.display.flip()
        
        # Update path animation progress
        if (self.animate_path or self.animate_visited) and self.path:
            self.path_animation_progress += self.path_animation_speed
            if self.path_animation_progress > 1.0:
                self.path_animation_progress = 1.0       
        
    def compare_all_algorithms(self):
        """Run all algorithms with consistent delay and collect results"""
        # Clear previous results
        self.results = []
        self.show_results = False
        
        # Set a consistent delay for all algorithms
        consistent_delay = 0.001  # Very small delay for quick execution
        
        # Run each algorithm with the same delay
        self.bfs(visualize=True, delay=consistent_delay)
        
        # Reset the maze state but keep the same maze layout
        self.visited = set()
        self.path = []
        self.steps = 0
        self.path_animation_progress = 0
        
        self.dfs(visualize=True, delay=consistent_delay)
        
        # Reset again
        self.visited = set()
        self.path = []
        self.steps = 0
        self.path_animation_progress = 0
        
        self.a_star(visualize=True, delay=consistent_delay)
        
        # Save results to the single CSV file
        self.save_results_to_csv()
        
        # Show results after all algorithms have run
        self.show_results = True  
    
    def draw_results_page(self):
        """Draw the results comparison page with improved visuals and scrolling support"""
        self.screen.fill(WHITE)
        
        # Create a large surface to hold all content
        content_height = 150 + (len(self.results) * 50) + 400  # Base height + results rows + charts
        content_surface = pygame.Surface((self.width, content_height))
        content_surface.fill(WHITE)
        
        # Title with shadow effect
        title_shadow = self.font_title.render("Algorithm Comparison Results", True, GRAY)
        title = self.font_title.render("Algorithm Comparison Results", True, BLACK)
        title_rect = title.get_rect(center=(self.width // 2, 50))
        content_surface.blit(title_shadow, (title_rect.x + 2, title_rect.y + 2))
        content_surface.blit(title, title_rect)
        
        # Header background
        header_bg = pygame.Rect(40, 110, self.width - 80, 40)
        pygame.draw.rect(content_surface, (240, 240, 240), header_bg)
        pygame.draw.rect(content_surface, GRAY, header_bg, 1)
        
        # Headers
        y_pos = 120
        headers = ["Algorithm", "Found Path", "Path Length", "Steps", "Time (sec)"]
        x_positions = [80, 220, 370, 520, 650]
        
        for header, x in zip(headers, x_positions):
            header_text = self.font_medium.render(header, True, BLACK)
            header_rect = header_text.get_rect(center=(x, y_pos))
            content_surface.blit(header_text, header_rect)
        
        # Results data
        y_pos += 40
        
        for i, result in enumerate(self.results):
            name, found, path_length, steps, execution_time = result
            color = self.path_colors[name]
            
            # Row background (alternating colors)
            row_bg = pygame.Rect(40, y_pos - 15, self.width - 80, 40)
            bg_color = (250, 250, 250) if i % 2 == 0 else (240, 240, 240)
            pygame.draw.rect(content_surface, bg_color, row_bg)
            pygame.draw.rect(content_surface, LIGHT_GRAY, row_bg, 1)
            
            # Draw algorithm name with its color
            algo_text = self.font_medium.render(name, True, BLACK)
            algo_rect = algo_text.get_rect(center=(x_positions[0], y_pos))
            content_surface.blit(algo_text, algo_rect)
            
            # Draw color sample
            color_rect = pygame.Rect(x_positions[0] + 50, y_pos - 10, 20, 20)
            pygame.draw.rect(content_surface, color, color_rect)
            pygame.draw.rect(content_surface, BLACK, color_rect, 1)
            
            # Draw other values
            found_text = self.font_medium.render("Yes" if found else "No", True, GREEN if found else RED)
            found_rect = found_text.get_rect(center=(x_positions[1], y_pos))
            content_surface.blit(found_text, found_rect)
            
            path_text = self.font_medium.render(str(path_length), True, BLACK)
            path_rect = path_text.get_rect(center=(x_positions[2], y_pos))
            content_surface.blit(path_text, path_rect)
            
            steps_text = self.font_medium.render(str(steps), True, BLACK)
            steps_rect = steps_text.get_rect(center=(x_positions[3], y_pos))
            content_surface.blit(steps_text, steps_rect)
            
            time_text = self.font_medium.render(f"{execution_time:.4f}", True, BLACK)
            time_rect = time_text.get_rect(center=(x_positions[4], y_pos))
            content_surface.blit(time_text, time_rect)
            
            y_pos += 40
        
        # Performance comparison visualization
        if len(self.results) > 1:
            y_pos += 20
            
            # Draw performance comparison header
            comparison_title = self.font_large.render("Performance Comparison", True, BLACK)
            comparison_rect = comparison_title.get_rect(center=(self.width // 2, y_pos))
            content_surface.blit(comparison_title, comparison_rect)
            y_pos += 50
            
            # Find fastest and shortest path
            times = [result[4] for result in self.results]
            path_lengths = [result[2] for result in self.results]
            steps_counts = [result[3] for result in self.results]
            
            # Normalize values for bar charts
            max_time = max(times) if times else 1
            max_path = max(path_lengths) if path_lengths else 1
            max_steps = max(steps_counts) if steps_counts else 1
            
            # Draw bar charts
            chart_width = 200
            chart_height = 20
            chart_x = self.width // 2 - chart_width // 2
            labels = ["Execution Time", "Path Length", "Steps Explored"]
            
            # Draw each metric comparison
            for i, (metric_values, max_val) in enumerate([(times, max_time), 
                                                        (path_lengths, max_path), 
                                                        (steps_counts, max_steps)]):
                
                # Draw metric label
                label_text = self.font_medium.render(labels[i], True, BLACK)
                content_surface.blit(label_text, (80, y_pos))
                
                # Draw bars for each algorithm
                for j, (result, val) in enumerate(zip(self.results, metric_values)):
                    name = result[0]
                    color = self.path_colors[name]
                    
                    # Calculate bar length
                    bar_length = int((val / max_val) * chart_width) if max_val > 0 else 0
                    
                    # Draw bar background
                    bg_rect = pygame.Rect(chart_x, y_pos + j * 30, chart_width, chart_height)
                    pygame.draw.rect(content_surface, LIGHT_GRAY, bg_rect)
                    
                    # Draw actual bar
                    bar_rect = pygame.Rect(chart_x, y_pos + j * 30, bar_length, chart_height)
                    pygame.draw.rect(content_surface, color, bar_rect)
                    
                    # Draw algorithm name
                    name_text = self.font_small.render(name, True, BLACK)
                    content_surface.blit(name_text, (chart_x - 40, y_pos + j * 30 + 2))
                    
                    # Draw value
                    if i == 0:  # Time
                        val_text = self.font_small.render(f"{val:.4f} s", True, BLACK)
                    else:
                        val_text = self.font_small.render(str(val), True, BLACK)
                    content_surface.blit(val_text, (chart_x + chart_width + 10, y_pos + j * 30 + 2))
                
                y_pos += len(self.results) * 30 + 20
        
        # Add scroll functionality variables
        self.results_scroll_y = getattr(self, 'results_scroll_y', 0)
        max_scroll = max(0, content_height - self.height + 100)  # +100 for bottom padding
        
        # Test Again Button - fixed position at the bottom of visible screen
        button_rect = pygame.Rect(self.width // 2 - 100, self.height - 70, 200, 50)
        hover = button_rect.collidepoint(pygame.mouse.get_pos())
        
        # Draw scrollable content to main screen with current scroll position
        self.screen.blit(content_surface, (0, -self.results_scroll_y))
        
        # Draw fixed position button (always visible at bottom of screen)
        fixed_button_bg = pygame.Rect(0, self.height - 80, self.width, 80)
        pygame.draw.rect(self.screen, (240, 240, 240), fixed_button_bg)
        pygame.draw.line(self.screen, GRAY, (0, self.height - 80), (self.width, self.height - 80), 1)
        
        if hover:
            pygame.draw.rect(self.screen, (220, 220, 220), button_rect)
        else:
            pygame.draw.rect(self.screen, LIGHT_GRAY, button_rect)
        
        pygame.draw.rect(self.screen, GRAY, button_rect, 2)
        
        button_text = self.font_medium.render("Test Again", True, BLACK)
        button_text_rect = button_text.get_rect(center=button_rect.center)
        self.screen.blit(button_text, button_text_rect)
        
        # Draw scroll indicators if content is scrollable
        if max_scroll > 0:
            # Draw scroll bar
            scroll_track_height = self.height - 100  # Account for padding
            scroll_bar_height = max(30, scroll_track_height * self.height / content_height)
            scroll_bar_pos = (self.results_scroll_y / max_scroll) * (scroll_track_height - scroll_bar_height)
            
            # Scroll track
            pygame.draw.rect(self.screen, LIGHT_GRAY, 
                            (self.width - 15, 10, 10, scroll_track_height))
            
            # Scroll bar
            pygame.draw.rect(self.screen, GRAY, 
                            (self.width - 15, 10 + scroll_bar_pos, 10, scroll_bar_height))
            
            # Up/down indicators
            if self.results_scroll_y > 0:
                pygame.draw.polygon(self.screen, GRAY, 
                                [(self.width - 10, 10), (self.width - 15, 20), (self.width - 5, 20)])
            
            if self.results_scroll_y < max_scroll:
                pygame.draw.polygon(self.screen, GRAY, 
                                [(self.width - 10, scroll_track_height + 10), 
                                (self.width - 15, scroll_track_height), 
                                (self.width - 5, scroll_track_height)])
        
        pygame.display.flip()
        return button_rect, max_scroll  # Return both button rect and max_scroll for use in main
    
    def reconstruct_path(self, came_from):
        """Reconstruct the path from start to goal using the came_from dict"""
        current = self.goal
        path = [current]
        
        while current != self.start:
            current = came_from[current]
            path.append(current)
        
        path.reverse()
        return path
    
    def bfs(self, visualize=True, delay=None):
        """Breadth-First Search algorithm"""
        print("Running BFS...")
        self.current_algorithm = 'BFS'
        self.visited = set()
        self.path = []
        self.steps = 0
        self.path_animation_progress = 0
        self.stop_requested = False
        self.running_algorithm = True  # Set flag when algorithm starts
        
        # Use the specified delay or fall back to the default animation speed
        delay = delay if delay is not None else self.animation_speed
        
        # Initialize BFS
        queue = deque([self.start])
        came_from = {self.start: None}
        self.visited.add(self.start)
        
        start_time = time.time()
        found = False
        
        while queue and not found and not self.stop_requested:
            current = queue.popleft()
            self.steps += 1
            
            if current == self.goal:
                found = True
                break
            
            for dx, dy in self.directions:
                next_x, next_y = current[0] + dx, current[1] + dy
                next_pos = (next_x, next_y)
                
                if self.is_valid(next_x, next_y) and next_pos not in self.visited:
                    queue.append(next_pos)
                    self.visited.add(next_pos)
                    came_from[next_pos] = current
            
            if visualize:
                self.draw_maze()
                pygame.time.delay(int(delay * 1000))  # Convert to milliseconds
                
                # Process events to allow stopping the algorithm
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return False, [], 0, 0
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        for button in self.buttons:
                            if button['rect'].collidepoint(event.pos) and button['action'] == 'stop':
                                self.stop_requested = True
        
        self.running_algorithm = False  # Reset flag when algorithm ends
        end_time = time.time()
        self.execution_time = end_time - start_time
        
        if found:
            self.path = self.reconstruct_path(came_from)
            
        if visualize:
            self.animate_path = True
            self.path_animation_progress = 0
            self.draw_maze()
        
        path_length = len(self.path) - 1 if self.path else 0
        result = ('BFS', found, path_length, self.steps, self.execution_time)
        
        # Add result if not already in results list
        if not any(r[0] == 'BFS' for r in self.results):
            self.results.append(result)
        else:
            # Update existing result
            for i, r in enumerate(self.results):
                if r[0] == 'BFS':
                    self.results[i] = result
                    break
        
        return found, self.path, self.steps, self.execution_time
    
    def dfs(self, visualize=True, delay=None):
        """Depth-First Search algorithm"""
        print("Running DFS...")
        self.current_algorithm = 'DFS'
        self.visited = set()
        self.path = []
        self.steps = 0
        self.path_animation_progress = 0
        self.stop_requested = False
        self.running_algorithm = True # Set flag when algorithm starts
        
        # Use the specified delay or fall back to the default animation speed
        delay = delay if delay is not None else self.animation_speed
        
        # Initialize DFS
        stack = [self.start]
        came_from = {self.start: None}
        self.visited.add(self.start)
        
        start_time = time.time()
        found = False
        
        while stack and not found and not self.stop_requested:
            current = stack.pop()
            self.steps += 1
            
            if current == self.goal:
                found = True
                break
            
            for dx, dy in self.directions:
                next_x, next_y = current[0] + dx, current[1] + dy
                next_pos = (next_x, next_y)
                
                if self.is_valid(next_x, next_y) and next_pos not in self.visited:
                    stack.append(next_pos)
                    self.visited.add(next_pos)
                    came_from[next_pos] = current
            
            if visualize:
                self.draw_maze()
                pygame.time.delay(int(delay * 1000))
                
                # Process events to allow stopping the algorithm
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return False, [], 0, 0
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        for button in self.buttons:
                            if button['rect'].collidepoint(event.pos) and button['action'] == 'stop':
                                self.stop_requested = True
        self.running_algorithm = False 
        end_time = time.time()
        self.execution_time = end_time - start_time
        
        if found:
            self.path = self.reconstruct_path(came_from)
            
        if visualize:
            self.animate_path = True
            self.path_animation_progress = 0
            self.draw_maze()
        
        path_length = len(self.path) - 1 if self.path else 0
        result = ('DFS', found, path_length, self.steps, self.execution_time)
        
        # Add result if not already in results list
        if not any(r[0] == 'DFS' for r in self.results):
            self.results.append(result)
        else:
            # Update existing result
            for i, r in enumerate(self.results):
                if r[0] == 'DFS':
                    self.results[i] = result
                    break
        
        return found, self.path, self.steps, self.execution_time
    
    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    
    def a_star(self, visualize=True, delay=None):
        """A* Search algorithm"""
        print("Running A*...")
        self.current_algorithm = 'A*'
        self.visited = set()
        self.path = []
        self.steps = 0
        self.path_animation_progress = 0
        self.stop_requested = False
        self.running_algorithm = True 
        
        # Use the specified delay or fall back to the default animation speed
        delay = delay if delay is not None else self.animation_speed
        
        # Initialize A*
        open_set = []
        heapq.heappush(open_set, (0, self.steps, self.start))  # (f_score, steps, position)
        came_from = {self.start: None}
        
        # g_score[pos] is the cost from start to pos
        g_score = {self.start: 0}
        
        # f_score[pos] = g_score[pos] + h(pos)
        f_score = {self.start: self.manhattan_distance(self.start, self.goal)}
        
        start_time = time.time()
        found = False
        
        while open_set and not found and not self.stop_requested:
            # Get the node with lowest f_score
            _, _, current = heapq.heappop(open_set)
            self.steps += 1
            self.visited.add(current)
            
            if current == self.goal:
                found = True
                break
            
            for dx, dy in self.directions:
                next_x, next_y = current[0] + dx, current[1] + dy
                next_pos = (next_x, next_y)
                
                if not self.is_valid(next_x, next_y):
                    continue
                    
                # Tentative g_score
                tentative_g_score = g_score[current] + 1
                
                if next_pos not in g_score or tentative_g_score < g_score[next_pos]:
                    # This path is better than any previous one
                    came_from[next_pos] = current
                    g_score[next_pos] = tentative_g_score
                    f_score[next_pos] = tentative_g_score + self.manhattan_distance(next_pos, self.goal)
                    
                    if next_pos not in [item[2] for item in open_set]:
                        heapq.heappush(open_set, (f_score[next_pos], self.steps, next_pos))
            
            if visualize:
                self.draw_maze()
                pygame.time.delay(int(delay * 1000))
                
                # Process events to allow stopping the algorithm
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return False, [], 0, 0
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        for button in self.buttons:
                            if button['rect'].collidepoint(event.pos) and button['action'] == 'stop':
                                self.stop_requested = True
        self.running_algorithm = False 
        end_time = time.time()
        self.execution_time = end_time - start_time
        
        if found:
            self.path = self.reconstruct_path(came_from)
            
        if visualize:
            self.animate_path = True
            self.path_animation_progress = 0
            self.draw_maze()
        
        path_length = len(self.path) - 1 if self.path else 0
        result = ('A*', found, path_length, self.steps, self.execution_time)
        
        # Add result if not already in results list
        if not any(r[0] == 'A*' for r in self.results):
            self.results.append(result)
        else:
            # Update existing result
            for i, r in enumerate(self.results):
                if r[0] == 'A*':
                    self.results[i] = result
                    break
        
        return found, self.path, self.steps, self.execution_time
    
    def save_results_to_csv(self):
        """Save algorithm results to a single CSV file that accumulates all test results"""
        filename = "maze_pathfinding_results.csv"
        file_exists = os.path.isfile(filename)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        maze_size = f"{self.grid_height}x{self.grid_width}"
        
        # Calculate maze complexity (percentage of cells that are walls)
        wall_count = sum(row.count(1) for row in self.maze)
        total_cells = self.grid_height * self.grid_width
        maze_complexity = wall_count / total_cells
        
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header only if file is new
            if not file_exists:
                writer.writerow(['Timestamp', 'Maze Size', 'Complexity', 'Algorithm', 
                                'Found Path', 'Path Length', 'Steps', 'Time (sec)'])
            
            # Write data rows for each algorithm
            for result in self.results:
                algo, found, path_length, steps, execution_time = result
                writer.writerow([timestamp, maze_size, f"{maze_complexity:.2f}", 
                                algo, found, path_length, steps, execution_time])
        
        print(f"Results appended to {filename}")
        return filename  
    
    def save_maze_configuration(self):
        """Save just the maze configuration for later reuse"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"maze_config_{timestamp}.json"
        
        config = {
            "maze": self.maze,
            "start": self.start,
            "goal": self.goal,
            "dimensions": {
                "width": self.grid_width,
                "height": self.grid_height
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f)
        
        print(f"Maze configuration saved to {filename}")
        return filename

    def load_maze_configuration(self, filename):
        """Load a previously saved maze configuration"""
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
                
            self.maze = config["maze"]
            self.start = tuple(config["start"])  # Convert from list to tuple if needed
            self.goal = tuple(config["goal"])    # Convert from list to tuple if needed
            
            # Reset state
            self.visited = set()
            self.path = []
            self.steps = 0
            self.path_animation_progress = 0
            self.results = []
            
            print(f"Maze loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading maze: {e}")
            return False

def main():
    app = MazePathfinder()
    running = True
    
    while running:
        # Get ALL events first before any processing
        events = pygame.event.get()
        
        # Check if we're showing results page
        if app.show_results:
            # Draw the results page and get the button rectangle and max scroll
            test_again_button, max_scroll = app.draw_results_page()
            
            # Process all events
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        # Check if "Test Again" button was clicked
                        if test_again_button.collidepoint(event.pos):
                            print("Test Again button clicked!")  # Debugging output
                            app.show_results = False
                            app.visited = set()
                            app.path = []
                            app.steps = 0
                            app.execution_time = 0
                            app.path_animation_progress = 0
                    # Handle scrolling in results view
                    elif event.button == 4:  # Scroll up
                        app.results_scroll_y = max(0, app.results_scroll_y - 30)
                    elif event.button == 5:  # Scroll down
                        app.results_scroll_y = min(max_scroll, app.results_scroll_y + 30)
        else:
            # Normal maze visualization
            app.draw_maze()
            
            # Process regular events
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
                    # Handle algorithm and control buttons
                    for button in app.buttons:
                        if button['rect'].collidepoint(event.pos):
                            if button['action'] == 'bfs':
                                app.bfs()
                            elif button['action'] == 'dfs':
                                app.dfs()
                            elif button['action'] == 'astar':
                                app.a_star()
                            elif button['action'] == 'compare':
                                app.compare_all_algorithms()
                            elif button['action'] == 'reset':
                                app = MazePathfinder()
                            elif button['action'] == 'stop':
                                app.stop_requested = True
                                print("Algorithm execution stopped by user")
                            elif button['action'] == 'save':
                                app.save_results_to_csv()
                                #app.save_results_to_json()
                                app.save_maze_configuration()
        
        app.clock.tick(60)  # 60 FPS
    
    pygame.quit()


if __name__ == "__main__":
    main()