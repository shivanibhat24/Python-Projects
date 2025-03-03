import pygame
import random
import math
import numpy as np
from collections import deque
import os

class QNetwork:
    def __init__(self, state_size=16, action_size=4):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Simple Q-network model
        model = {
            'weights1': np.random.randn(self.state_size, 24),
            'bias1': np.zeros(24),
            'weights2': np.random.randn(24, 16),
            'bias2': np.zeros(16),
            'weights3': np.random.randn(16, self.action_size),
            'bias3': np.zeros(self.action_size)
        }
        return model

    def _relu(self, x):
        return np.maximum(0, x)

    def predict(self, state):
        # Forward pass through the network
        layer1 = np.dot(state, self.model['weights1']) + self.model['bias1']
        layer1_activation = self._relu(layer1)
        layer2 = np.dot(layer1_activation, self.model['weights2']) + self.model['bias2']
        layer2_activation = self._relu(layer2)
        output = np.dot(layer2_activation, self.model['weights3']) + self.model['bias3']
        return output

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.predict(state)
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_q_values = self.predict(next_state)
                target = reward + self.gamma * np.amax(next_q_values)
            
            # Simple update rule (very simplified for this example)
            target_f = self.predict(state)
            target_f[action] = target
            
            # Update weights (extremely simplified)
            error = target_f - self.predict(state)
            # In a real implementation, we would do proper backpropagation
            # This is a very simple approximation
            self.model['weights3'] += self.learning_rate * error.reshape(-1, 1)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class Game2048:
    # Color schemes and values
    COLORS = {
        0: (205, 192, 180),
        2: (237, 229, 218),
        4: (237, 227, 202),
        8: (242, 177, 121),
        16: (246, 149, 99),
        32: (247, 124, 95),
        64: (247, 95, 59),
        128: (237, 208, 115),
        256: (237, 204, 99),
        512: (236, 201, 80),
        1024: (236, 198, 65),
        2048: (237, 194, 46),
        4096: (239, 195, 45),
        8192: (255, 85, 85),
    }

    TEXT_COLORS = {
        0: (205, 192, 180),
        2: (119, 110, 101),
        4: (119, 110, 101),
        8: (249, 246, 242),
        16: (249, 246, 242),
        32: (249, 246, 242),
        64: (249, 246, 242),
        128: (249, 246, 242),
        256: (249, 246, 242),
        512: (249, 246, 242),
        1024: (249, 246, 242),
        2048: (249, 246, 242),
        4096: (249, 246, 242),
        8192: (249, 246, 242),
    }

    def __init__(self, width=800, height=800, rows=4, cols=4):
        pygame.init()
        self.width = width
        self.height = height
        self.rows = rows
        self.cols = cols
        self.tile_height = height // rows
        self.tile_width = width // cols

        # UI Constants
        self.outline_color = (187, 173, 160)
        self.background_color = (250, 248, 239)
        self.panel_color = (187, 173, 160)
        self.font_color = (119, 110, 101)
        self.outline_thickness = 10
        self.tile_margin = 10
        self.tile_border_radius = 10
        self.high_score = 0
        self.current_score = 0
        self.game_over_font = pygame.font.SysFont("Arial", 72, bold=True)
        self.score_font = pygame.font.SysFont("Arial", 36, bold=True)
        self.title_font = pygame.font.SysFont("Arial", 80, bold=True)
        
        # Dynamic font sizing
        self.font_sizes = {
            2: 60, 4: 60, 8: 60, 16: 60,
            32: 56, 64: 56, 128: 48,
            256: 48, 512: 48, 1024: 36,
            2048: 36, 4096: 36, 8192: 36
        }
        
        self.fonts = {
            num: pygame.font.SysFont("Arial", size, bold=True) 
            for num, size in self.font_sizes.items()
        }

        # Game constants
        self.fps = 60
        self.move_vel = 25  # Increased for smoother animation
        
        # ML components
        self.agent = QNetwork()
        self.is_ai_playing = False
        self.ai_delay = 500  # milliseconds between AI moves
        self.last_ai_move_time = 0
        
        # Init display
        self.window = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Enhanced 2048")
        
        # Game state
        self.tiles = self.generate_tiles()
        self.board = self.get_state_matrix()
        self.running = True
        self.game_over = False
        self.clock = pygame.time.Clock()

    def generate_tiles(self):
        tiles = {}
        for _ in range(2):
            row, col = self.get_random_pos(tiles)
            tiles[f"{row}{col}"] = Tile(2, row, col, self.tile_width, self.tile_height)
        return tiles

    def get_random_pos(self, tiles):
        available_pos = []
        for row in range(self.rows):
            for col in range(self.cols):
                if f"{row}{col}" not in tiles:
                    available_pos.append((row, col))
        
        if not available_pos:
            return None, None
        
        return random.choice(available_pos)

    def get_state_matrix(self):
        board = np.zeros((self.rows, self.cols), dtype=int)
        for key, tile in self.tiles.items():
            row, col = tile.row, tile.col
            board[row][col] = tile.value
        return board
    
    def get_flat_state(self):
        return self.get_state_matrix().flatten()

    def add_random_tile(self):
        if len(self.tiles) >= self.rows * self.cols:
            return False
        
        row, col = self.get_random_pos(self.tiles)
        if row is None:
            return False
            
        value = 2 if random.random() < 0.9 else 4
        self.tiles[f"{row}{col}"] = Tile(value, row, col, self.tile_width, self.tile_height)
        return True

    def draw_game_over(self):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((255, 255, 255, 180))
        self.window.blit(overlay, (0, 0))
        
        game_over_text = self.game_over_font.render("Game Over!", True, (119, 110, 101))
        restart_text = self.score_font.render("Press R to Restart", True, (119, 110, 101))
        
        self.window.blit(
            game_over_text,
            (self.width // 2 - game_over_text.get_width() // 2, 
             self.height // 2 - game_over_text.get_height() // 2 - 50)
        )
        
        self.window.blit(
            restart_text,
            (self.width // 2 - restart_text.get_width() // 2, 
             self.height // 2 - restart_text.get_height() // 2 + 50)
        )

    def draw_header(self):
        # Draw title
        title_text = self.title_font.render("2048", True, self.font_color)
        self.window.blit(title_text, (50, 30))
        
        # Draw score panel
        pygame.draw.rect(self.window, self.panel_color, 
                        (self.width - 320, 30, 140, 80), 
                        border_radius=5)
        pygame.draw.rect(self.window, self.panel_color, 
                        (self.width - 160, 30, 140, 80), 
                        border_radius=5)
        
        score_label = self.score_font.render("SCORE", True, (249, 246, 242))
        score_value = self.score_font.render(str(self.current_score), True, (255, 255, 255))
        
        high_score_label = self.score_font.render("BEST", True, (249, 246, 242))
        high_score_value = self.score_font.render(str(self.high_score), True, (255, 255, 255))
        
        # Score panel
        self.window.blit(
            score_label,
            (self.width - 320 + 70 - score_label.get_width() // 2, 40)
        )
        self.window.blit(
            score_value,
            (self.width - 320 + 70 - score_value.get_width() // 2, 70)
        )
        
        # Best score panel
        self.window.blit(
            high_score_label,
            (self.width - 160 + 70 - high_score_label.get_width() // 2, 40)
        )
        self.window.blit(
            high_score_value,
            (self.width - 160 + 70 - high_score_value.get_width() // 2, 70)
        )
        
        # AI toggle button
        ai_button_color = (142, 122, 101) if self.is_ai_playing else (187, 173, 160)
        pygame.draw.rect(self.window, ai_button_color, 
                        (50, 120, 200, 50), 
                        border_radius=5)
        ai_text = self.score_font.render("AI Mode: " + ("ON" if self.is_ai_playing else "OFF"), True, (249, 246, 242))
        self.window.blit(
            ai_text,
            (50 + 100 - ai_text.get_width() // 2, 120 + 25 - ai_text.get_height() // 2)
        )

    def draw_grid(self):
        # Draw the grid background
        pygame.draw.rect(
            self.window, 
            self.panel_color, 
            (0, 200, self.width, self.height - 200), 
            border_radius=10
        )
        
        # Draw empty cell backgrounds
        for row in range(self.rows):
            for col in range(self.cols):
                pygame.draw.rect(
                    self.window,
                    self.COLORS[0],  # Empty cell color
                    (
                        col * self.tile_width + self.tile_margin,
                        row * self.tile_height + self.tile_margin + 200,  # Offset for header
                        self.tile_width - 2 * self.tile_margin,
                        self.tile_height - 2 * self.tile_margin
                    ),
                    border_radius=self.tile_border_radius
                )

    def draw(self):
        self.window.fill(self.background_color)
        
        # Draw header elements
        self.draw_header()
        
        # Draw grid background
        self.draw_grid()

        # Draw tiles
        for tile in self.tiles.values():
            tile.draw(self.window, self.fonts, self.TEXT_COLORS)

        # Draw game over overlay if needed
        if self.game_over:
            self.draw_game_over()

        pygame.display.update()

    def check_game_over(self):
        # Check if board is full
        if len(self.tiles) < self.rows * self.cols:
            return False
        
        # Check if any moves are possible
        directions = ["left", "right", "up", "down"]
        board = self.get_state_matrix()
        
        for direction in directions:
            # Try each direction
            board_copy = np.copy(board)
            if direction == "left":
                for row in range(self.rows):
                    merged = [False] * self.cols
                    for col in range(1, self.cols):
                        if board_copy[row][col] == 0:
                            continue
                        
                        curr_col = col
                        while curr_col > 0 and board_copy[row][curr_col-1] == 0:
                            board_copy[row][curr_col-1] = board_copy[row][curr_col]
                            board_copy[row][curr_col] = 0
                            curr_col -= 1
                        
                        if curr_col > 0 and board_copy[row][curr_col-1] == board_copy[row][curr_col] and not merged[curr_col-1]:
                            board_copy[row][curr_col-1] *= 2
                            board_copy[row][curr_col] = 0
                            merged[curr_col-1] = True
            
            elif direction == "right":
                for row in range(self.rows):
                    merged = [False] * self.cols
                    for col in range(self.cols-2, -1, -1):
                        if board_copy[row][col] == 0:
                            continue
                        
                        curr_col = col
                        while curr_col < self.cols-1 and board_copy[row][curr_col+1] == 0:
                            board_copy[row][curr_col+1] = board_copy[row][curr_col]
                            board_copy[row][curr_col] = 0
                            curr_col += 1
                        
                        if curr_col < self.cols-1 and board_copy[row][curr_col+1] == board_copy[row][curr_col] and not merged[curr_col+1]:
                            board_copy[row][curr_col+1] *= 2
                            board_copy[row][curr_col] = 0
                            merged[curr_col+1] = True
            
            elif direction == "up":
                for col in range(self.cols):
                    merged = [False] * self.rows
                    for row in range(1, self.rows):
                        if board_copy[row][col] == 0:
                            continue
                        
                        curr_row = row
                        while curr_row > 0 and board_copy[curr_row-1][col] == 0:
                            board_copy[curr_row-1][col] = board_copy[curr_row][col]
                            board_copy[curr_row][col] = 0
                            curr_row -= 1
                        
                        if curr_row > 0 and board_copy[curr_row-1][col] == board_copy[curr_row][col] and not merged[curr_row-1]:
                            board_copy[curr_row-1][col] *= 2
                            board_copy[curr_row][col] = 0
                            merged[curr_row-1] = True
            
            elif direction == "down":
                for col in range(self.cols):
                    merged = [False] * self.rows
                    for row in range(self.rows-2, -1, -1):
                        if board_copy[row][col] == 0:
                            continue
                        
                        curr_row = row
                        while curr_row < self.rows-1 and board_copy[curr_row+1][col] == 0:
                            board_copy[curr_row+1][col] = board_copy[curr_row][col]
                            board_copy[curr_row][col] = 0
                            curr_row += 1
                        
                        if curr_row < self.rows-1 and board_copy[curr_row+1][col] == board_copy[curr_row][col] and not merged[curr_row+1]:
                            board_copy[curr_row+1][col] *= 2
                            board_copy[curr_row][col] = 0
                            merged[curr_row+1] = True
            
            # Check if board changed after move
            if not np.array_equal(board, board_copy):
                return False
                
        return True

    def move_tiles(self, direction):
        updated = True
        blocks = set()
        score_addition = 0
        original_board = self.get_state_matrix().copy()

        if direction == "left":
            sort_func = lambda x: x.col
            reverse = False
            delta = (-self.move_vel, 0)
            boundary_check = lambda tile: tile.col == 0
            get_next_tile = lambda tile: self.tiles.get(f"{tile.row}{tile.col - 1}")
            merge_check = lambda tile, next_tile: tile.x > next_tile.x + self.move_vel
            move_check = (
                lambda tile, next_tile: tile.x > next_tile.x + self.tile_width + self.move_vel
            )
            ceil = True
        elif direction == "right":
            sort_func = lambda x: x.col
            reverse = True
            delta = (self.move_vel, 0)
            boundary_check = lambda tile: tile.col == self.cols - 1
            get_next_tile = lambda tile: self.tiles.get(f"{tile.row}{tile.col + 1}")
            merge_check = lambda tile, next_tile: tile.x < next_tile.x - self.move_vel
            move_check = (
                lambda tile, next_tile: tile.x + self.tile_width + self.move_vel < next_tile.x
            )
            ceil = False
        elif direction == "up":
            sort_func = lambda x: x.row
            reverse = False
            delta = (0, -self.move_vel)
            boundary_check = lambda tile: tile.row == 0
            get_next_tile = lambda tile: self.tiles.get(f"{tile.row - 1}{tile.col}")
            merge_check = lambda tile, next_tile: tile.y > next_tile.y + self.move_vel
            move_check = (
                lambda tile, next_tile: tile.y > next_tile.y + self.tile_height + self.move_vel
            )
            ceil = True
        elif direction == "down":
            sort_func = lambda x: x.row
            reverse = True
            delta = (0, self.move_vel)
            boundary_check = lambda tile: tile.row == self.rows - 1
            get_next_tile = lambda tile: self.tiles.get(f"{tile.row + 1}{tile.col}")
            merge_check = lambda tile, next_tile: tile.y < next_tile.y - self.move_vel
            move_check = (
                lambda tile, next_tile: tile.y + self.tile_height + self.move_vel < next_tile.y
            )
            ceil = False

        while updated:
            self.clock.tick(self.fps)
            updated = False
            sorted_tiles = sorted(self.tiles.values(), key=sort_func, reverse=reverse)

            for i, tile in enumerate(sorted_tiles):
                if boundary_check(tile):
                    continue

                next_tile = get_next_tile(tile)
                if not next_tile:
                    tile.move(delta)
                elif (
                    tile.value == next_tile.value
                    and tile not in blocks
                    and next_tile not in blocks
                ):
                    if merge_check(tile, next_tile):
                        tile.move(delta)
                    else:
                        next_tile.value *= 2
                        score_addition += next_tile.value
                        sorted_tiles.pop(i)
                        blocks.add(next_tile)
                elif move_check(tile, next_tile):
                    tile.move(delta)
                else:
                    continue

                tile.set_pos(ceil)
                updated = True

            self.update_tiles(sorted_tiles)
            self.draw()

        # Update score
        self.current_score += score_addition
        if self.current_score > self.high_score:
            self.high_score = self.current_score

        # Check if the board changed
        new_board = self.get_state_matrix()
        if not np.array_equal(original_board, new_board):
            self.add_random_tile()
            self.game_over = self.check_game_over()
            return True
        return False

    def update_tiles(self, sorted_tiles):
        self.tiles.clear()
        for tile in sorted_tiles:
            self.tiles[f"{tile.row}{tile.col}"] = tile

    def reset_game(self):
        self.tiles = self.generate_tiles()
        self.current_score = 0
        self.game_over = False
        self.board = self.get_state_matrix()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN:
                if self.game_over:
                    if event.key == pygame.K_r:
                        self.reset_game()
                else:
                    direction = None
                    if event.key == pygame.K_LEFT:
                        direction = "left"
                    elif event.key == pygame.K_RIGHT:
                        direction = "right"
                    elif event.key == pygame.K_UP:
                        direction = "up"
                    elif event.key == pygame.K_DOWN:
                        direction = "down"
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_a:
                        self.is_ai_playing = not self.is_ai_playing
                        
                    if direction:
                        # If manual move performed, update the board
                        moved = self.move_tiles(direction)
                        
                        if moved:
                            # Store the experience
                            state = self.get_flat_state()
                            action_map = {"left": 0, "right": 1, "up": 2, "down": 3}
                            action = action_map[direction]
                            next_state = self.get_flat_state()
                            reward = self.current_score  # Use score as reward
                            self.agent.remember(state, action, reward, next_state, self.game_over)
            
            # Check for mouse clicks (for UI buttons)
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                # Check if AI button clicked
                if 50 <= mouse_pos[0] <= 250 and 120 <= mouse_pos[1] <= 170:
                    self.is_ai_playing = not self.is_ai_playing

    def ai_move(self):
        if not self.is_ai_playing or self.game_over:
            return
            
        current_time = pygame.time.get_ticks()
        if current_time - self.last_ai_move_time < self.ai_delay:
            return
            
        self.last_ai_move_time = current_time
        
        # Get the current state
        state = self.get_flat_state()
        
        # Get action from the AI
        action = self.agent.act(state)
        
        # Map action to direction
        action_map = {0: "left", 1: "right", 2: "up", 3: "down"}
        direction = action_map[action]
        
        # Perform the move
        moved = self.move_tiles(direction)
        
        if moved:
            # Get new state
            next_state = self.get_flat_state()
            
            # Calculate reward (use score difference)
            reward = self.current_score
            
            # Remember the experience
            self.agent.remember(state, action, reward, next_state, self.game_over)
            
            # Train the agent with a small batch
            if len(self.agent.memory) > 32:
                self.agent.replay(32)

    def run(self):
        while self.running:
            self.clock.tick(self.fps)
            
            self.handle_events()
            
            if self.is_ai_playing:
                self.ai_move()
                
            self.draw()
            
        pygame.quit()


class Tile:
    def __init__(self, value, row, col, tile_width, tile_height):
        self.value = value
        self.row = row
        self.col = col
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.x = col * tile_width
        self.y = row * tile_height + 200  # Offset for header
        self.margin = 10
        self.border_radius = 10
        self.tile_anim_scale = 1.0
        self.pulse_count = 0

    def draw(self, window, fonts, text_colors):
        # Get color from Game2048.COLORS
        color = Game2048.COLORS.get(self.value, (0, 0, 0))
        
        # Calculate drawing rectangle with margin
        rect = (
            self.x + self.margin,
            self.y + self.margin,
            self.tile_width - 2 * self.margin,
            self.tile_height - 2 * self.margin
        )
        
        # Draw the tile background
        pygame.draw.rect(window, color, rect, border_radius=self.border_radius)
        
        # Draw the value
        if self.value > 0:
            font = fonts.get(self.value, pygame.font.SysFont("Arial", 36, bold=True))
            text_color = text_colors.get(self.value, (0, 0, 0))
            text = font.render(str(self.value), True, text_color)
            
            # Position text in center of tile
            window.blit(
                text,
                (
                    self.x + (self.tile_width / 2 - text.get_width() / 2),
                    self.y + (self.tile_height / 2 - text.get_height() / 2),
                ),
            )

    def move(self, delta):
        self.x += delta[0]
        self.y += delta[1]

    def set_pos(self, ceil=False):
        if ceil:
            # Adjust for header offset when calculating row
            adjusted_y = self.y - 200
            self.row = math.ceil(adjusted_y / self.tile_height)
            self.col = math.ceil(self.x / self.tile_width)
        else:
            adjusted_y = self.y - 200
            self.row = math.floor(adjusted_y / self.tile_height)
            self.col = math.floor(self.x / self.tile_width)


if __name__ == "__main__":
    game = Game2048()
    game.run()
