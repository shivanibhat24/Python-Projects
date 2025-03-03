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
            
            # Get activations for backprop
            layer1 = np.dot(state, self.model['weights1']) + self.model['bias1']
            layer1_activation = self._relu(layer1)
            layer2 = np.dot(layer1_activation, self.model['weights2']) + self.model['bias2']
            layer2_activation = self._relu(layer2)
            
            # Calculate current prediction and target
            q_values = self.predict(state)
            target_f = np.copy(q_values)
            target_f[action] = target
            
            # Calculate error
            error = target_f - q_values
            
            # Simple gradient update for final layer weights
            d_weights3 = np.outer(layer2_activation, error)
            self.model['weights3'] += self.learning_rate * d_weights3
            self.model['bias3'] += self.learning_rate * error
            
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

    def __init__(self, width=480, height=640, rows=4, cols=4):
        pygame.init()
        self.width = width
        self.height = height
        self.header_height = 160  # Reduced header height
        self.game_height = height - self.header_height
        self.rows = rows
        self.cols = cols
        self.tile_height = self.game_height // rows
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
        self.game_over_font = pygame.font.SysFont("Arial", 48, bold=True)
        self.score_font = pygame.font.SysFont("Arial", 28, bold=True)
        self.title_font = pygame.font.SysFont("Arial", 60, bold=True)
        
        # Dynamic font sizing - adjusted for smaller tiles
        self.font_sizes = {
            2: 40, 4: 40, 8: 40, 16: 40,
            32: 36, 64: 36, 128: 32,
            256: 32, 512: 32, 1024: 24,
            2048: 24, 4096: 24, 8192: 24
        }
        
        self.fonts = {
            num: pygame.font.SysFont("Arial", size, bold=True) 
            for num, size in self.font_sizes.items()
        }

        # Game constants
        self.fps = 60
        self.move_vel = 20  # Adjusted for smaller tiles
        
        # ML components
        self.agent = QNetwork()
        self.is_ai_playing = False
        self.ai_delay = 500  # milliseconds between AI moves
        self.last_ai_move_time = 0
        
        # Init display
        self.window = pygame.display.set_mode((width, height))
        pygame.display.set_caption("2048 Game")
        
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
            tiles[f"{row}{col}"] = Tile(2, row, col, self.tile_width, self.tile_height, self.header_height)
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
        new_tile = Tile(value, row, col, self.tile_width, self.tile_height, self.header_height)
        new_tile.pulse_count = 3  # Set pulse animation for new tiles
        self.tiles[f"{row}{col}"] = new_tile
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
        # Draw title - adjusted positions for smaller window
        title_text = self.title_font.render("2048", True, self.font_color)
        self.window.blit(title_text, (30, 20))
        
        # Draw score panel - adjusted sizes and positions
        pygame.draw.rect(self.window, self.panel_color, 
                        (self.width - 240, 20, 100, 60), 
                        border_radius=5)
        pygame.draw.rect(self.window, self.panel_color, 
                        (self.width - 130, 20, 100, 60), 
                        border_radius=5)
        
        score_label = self.score_font.render("SCORE", True, (249, 246, 242))
        score_value = self.score_font.render(str(self.current_score), True, (255, 255, 255))
        
        high_score_label = self.score_font.render("BEST", True, (249, 246, 242))
        high_score_value = self.score_font.render(str(self.high_score), True, (255, 255, 255))
        
        # Score panel
        self.window.blit(
            score_label,
            (self.width - 240 + 50 - score_label.get_width() // 2, 25)
        )
        self.window.blit(
            score_value,
            (self.width - 240 + 50 - score_value.get_width() // 2, 50)
        )
        
        # Best score panel
        self.window.blit(
            high_score_label,
            (self.width - 130 + 50 - high_score_label.get_width() // 2, 25)
        )
        self.window.blit(
            high_score_value,
            (self.width - 130 + 50 - high_score_value.get_width() // 2, 50)
        )
        
        # AI toggle button - adjusted position and size
        ai_button_color = (142, 122, 101) if self.is_ai_playing else (187, 173, 160)
        pygame.draw.rect(self.window, ai_button_color, 
                        (30, 90, 160, 40), 
                        border_radius=5)
        ai_text = self.score_font.render("AI: " + ("ON" if self.is_ai_playing else "OFF"), True, (249, 246, 242))
        self.window.blit(
            ai_text,
            (30 + 80 - ai_text.get_width() // 2, 90 + 20 - ai_text.get_height() // 2)
        )

    def draw_grid(self):
        # Draw the grid background
        pygame.draw.rect(
            self.window, 
            self.panel_color, 
            (0, self.header_height, self.width, self.game_height), 
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
                        row * self.tile_height + self.tile_margin + self.header_height,
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
            # Create a copy of the board for testing
            board_copy = np.copy(board)
            moved = self.try_move(direction, board_copy)
            if moved:
                return False
                
        return True
    
    def try_move(self, direction, board):
        """Test if a move is valid without actually making the move"""
        board_changed = False
        
        if direction == "left":
            for row in range(self.rows):
                # Process each row left to right
                merged = [False] * self.cols
                for col in range(1, self.cols):
                    if board[row][col] == 0:
                        continue
                    
                    # Try to move left
                    curr_col = col
                    while curr_col > 0 and board[row][curr_col-1] == 0:
                        board[row][curr_col-1] = board[row][curr_col]
                        board[row][curr_col] = 0
                        curr_col -= 1
                        board_changed = True
                    
                    # Try to merge with the tile to the left
                    if curr_col > 0 and board[row][curr_col-1] == board[row][curr_col] and not merged[curr_col-1]:
                        board[row][curr_col-1] *= 2
                        board[row][curr_col] = 0
                        merged[curr_col-1] = True
                        board_changed = True
        
        elif direction == "right":
            for row in range(self.rows):
                # Process each row right to left
                merged = [False] * self.cols
                for col in range(self.cols-2, -1, -1):
                    if board[row][col] == 0:
                        continue
                    
                    # Try to move right
                    curr_col = col
                    while curr_col < self.cols-1 and board[row][curr_col+1] == 0:
                        board[row][curr_col+1] = board[row][curr_col]
                        board[row][curr_col] = 0
                        curr_col += 1
                        board_changed = True
                    
                    # Try to merge with the tile to the right
                    if curr_col < self.cols-1 and board[row][curr_col+1] == board[row][curr_col] and not merged[curr_col+1]:
                        board[row][curr_col+1] *= 2
                        board[row][curr_col] = 0
                        merged[curr_col+1] = True
                        board_changed = True
        
        elif direction == "up":
            for col in range(self.cols):
                # Process each column top to bottom
                merged = [False] * self.rows
                for row in range(1, self.rows):
                    if board[row][col] == 0:
                        continue
                    
                    # Try to move up
                    curr_row = row
                    while curr_row > 0 and board[curr_row-1][col] == 0:
                        board[curr_row-1][col] = board[curr_row][col]
                        board[curr_row][col] = 0
                        curr_row -= 1
                        board_changed = True
                    
                    # Try to merge with the tile above
                    if curr_row > 0 and board[curr_row-1][col] == board[curr_row][col] and not merged[curr_row-1]:
                        board[curr_row-1][col] *= 2
                        board[curr_row][col] = 0
                        merged[curr_row-1] = True
                        board_changed = True
        
        elif direction == "down":
            for col in range(self.cols):
                # Process each column bottom to top
                merged = [False] * self.rows
                for row in range(self.rows-2, -1, -1):
                    if board[row][col] == 0:
                        continue
                    
                    # Try to move down
                    curr_row = row
                    while curr_row < self.rows-1 and board[curr_row+1][col] == 0:
                        board[curr_row+1][col] = board[curr_row][col]
                        board[curr_row][col] = 0
                        curr_row += 1
                        board_changed = True
                    
                    # Try to merge with the tile below
                    if curr_row < self.rows-1 and board[curr_row+1][col] == board[curr_row][col] and not merged[curr_row+1]:
                        board[curr_row+1][col] *= 2
                        board[curr_row][col] = 0
                        merged[curr_row+1] = True
                        board_changed = True
        
        return board_changed

    def move_tiles(self, direction):
        # Store the original board state to check if the move made a change
        original_board = self.get_state_matrix().copy()
        
        # Try the move on a copy of the board first to see if it's valid
        board_copy = original_board.copy()
        if not self.try_move(direction, board_copy):
            return False  # No valid move
            
        # If move is valid, make the actual move
        new_tiles = {}
        
        if direction == "left":
            for row in range(self.rows):
                merged = [False] * self.cols
                for col in range(self.cols):
                    key = f"{row}{col}"
                    if key in self.tiles:
                        tile = self.tiles[key]
                        curr_col = col
                        
                        # Move as far left as possible
                        while curr_col > 0 and f"{row}{curr_col-1}" not in new_tiles:
                            curr_col -= 1
                        
                        # Check for merge
                        if curr_col > 0:
                            left_key = f"{row}{curr_col-1}"
                            if left_key in new_tiles and new_tiles[left_key].value == tile.value and not merged[curr_col-1]:
                                # Merge with tile to the left
                                new_tiles[left_key].value *= 2
                                new_tiles[left_key].pulse_count = 3  # Trigger pulse animation for merged tile
                                self.current_score += new_tiles[left_key].value
                                merged[curr_col-1] = True
                                continue
                        
                        # Place tile in new position
                        new_key = f"{row}{curr_col}"
                        tile.row = row
                        tile.col = curr_col
                        tile.x = curr_col * self.tile_width
                        tile.y = row * self.tile_height + self.header_height
                        new_tiles[new_key] = tile
        
        elif direction == "right":
            for row in range(self.rows):
                merged = [False] * self.cols
                for col in range(self.cols-1, -1, -1):
                    key = f"{row}{col}"
                    if key in self.tiles:
                        tile = self.tiles[key]
                        curr_col = col
                        
                        # Move as far right as possible
                        while curr_col < self.cols-1 and f"{row}{curr_col+1}" not in new_tiles:
                            curr_col += 1
                        
                        # Check for merge
                        if curr_col < self.cols-1:
                            right_key = f"{row}{curr_col+1}"
                            if right_key in new_tiles and new_tiles[right_key].value == tile.value and not merged[curr_col+1]:
                                # Merge with tile to the right
                                new_tiles[right_key].value *= 2
                                new_tiles[right_key].pulse_count = 3  # Trigger pulse animation for merged tile
                                self.current_score += new_tiles[right_key].value
                                merged[curr_col+1] = True
                                continue
                        
                        # Place tile in new position
                        new_key = f"{row}{curr_col}"
                        tile.row = row
                        tile.col = curr_col
                        tile.x = curr_col * self.tile_width
                        tile.y = row * self.tile_height + self.header_height
                        new_tiles[new_key] = tile
        
        elif direction == "up":
            for col in range(self.cols):
                merged = [False] * self.rows
                for row in range(self.rows):
                    key = f"{row}{col}"
                    if key in self.tiles:
                        tile = self.tiles[key]
                        curr_row = row
                        
                        # Move as far up as possible
                        while curr_row > 0 and f"{curr_row-1}{col}" not in new_tiles:
                            curr_row -= 1
                        
                        # Check for merge
                        if curr_row > 0:
                            up_key = f"{curr_row-1}{col}"
                            if up_key in new_tiles and new_tiles[up_key].value == tile.value and not merged[curr_row-1]:
                                # Merge with tile above
                                new_tiles[up_key].value *= 2
                                new_tiles[up_key].pulse_count = 3  # Trigger pulse animation for merged tile
                                self.current_score += new_tiles[up_key].value
                                merged[curr_row-1] = True
                                continue
                        
                        # Place tile in new position
                        new_key = f"{curr_row}{col}"
                        tile.row = curr_row
                        tile.col = col
                        tile.x = col * self.tile_width
                        tile.y = curr_row * self.tile_height + self.header_height
                        new_tiles[new_key] = tile
        
        elif direction == "down":
            for col in range(self.cols):
                merged = [False] * self.rows
                for row in range(self.rows-1, -1, -1):
                    key = f"{row}{col}"
                    if key in self.tiles:
                        tile = self.tiles[key]
                        curr_row = row
                        
                        # Move as far down as possible
                        while curr_row < self.rows-1 and f"{curr_row+1}{col}" not in new_tiles:
                            curr_row += 1
                        
                        # Check for merge
                        if curr_row < self.rows-1:
                            down_key = f"{curr_row+1}{col}"
                            if down_key in new_tiles and new_tiles[down_key].value == tile.value and not merged[curr_row+1]:
                                # Merge with tile below
                                new_tiles[down_key].value *= 2
                                new_tiles[down_key].pulse_count = 3  # Trigger pulse animation for merged tile
                                self.current_score += new_tiles[down_key].value
                                merged[curr_row+1] = True
                                continue
                        
                        # Place tile in new position
                        new_key = f"{curr_row}{col}"
                        tile.row = curr_row
                        tile.col = col
                        tile.x = col * self.tile_width
                        tile.y = curr_row * self.tile_height + self.header_height
                        new_tiles[new_key] = tile
        
        # Update tiles dictionary
        self.tiles = new_tiles
        
        # Update score
        if self.current_score > self.high_score:
            self.high_score = self.current_score
        
        # Check if board changed
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
                # Check if AI button clicked - adjusted for new position
                if 30 <= mouse_pos[0] <= 190 and 90 <= mouse_pos[1] <= 130:
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
    def __init__(self, value, row, col, tile_width, tile_height, header_height):
        self.value = value
        self.row = row
        self.col = col
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.header_height = header_height
        self.x = col * tile_width
        self.y = row * tile_height + header_height
        self.margin = 10
        self.border_radius = 10
        self.tile_anim_scale = 1.0
        self.pulse_count = 0
        
    def draw(self, window, fonts, text_colors):
        # Calculate pulse animation effect
        pulse_scale = 1.0
        if self.pulse_count > 0:
            # Pulsating effect - make the tile grow slightly and then return to normal
            pulse_scale = 1.0 + (0.1 * (self.pulse_count / 3))
            self.pulse_count -= 0.1
            
        # Calculate draw position and dimensions with pulse effect
        x = self.x + self.margin + ((1 - pulse_scale) * self.tile_width / 2)
        y = self.y + self.margin + ((1 - pulse_scale) * self.tile_height / 2)
        width = (self.tile_width - 2 * self.margin) * pulse_scale
        height = (self.tile_height - 2 * self.margin) * pulse_scale
        
        # Draw tile
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(
            window, 
            Game2048.COLORS.get(self.value, Game2048.COLORS[2048]), 
            rect,
            border_radius=self.border_radius
        )
        
        # Draw text
        if self.value > 0:
            font = fonts.get(self.value, fonts[2048])
            text = font.render(str(self.value), True, text_colors.get(self.value, (255, 255, 255)))
            
            # Center text
            text_x = x + (width - text.get_width()) / 2
            text_y = y + (height - text.get_height()) / 2
            
            window.blit(text, (text_x, text_y))


if __name__ == "__main__":
    game = Game2048()
    game.run()
