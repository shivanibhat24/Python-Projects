import pygame
import sys
import random
import os

class RetroGameCreator:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Retro Game Creator")
        
        # Default game settings
        self.screen_width = 640
        self.screen_height = 480
        self.fps = 60
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        
        # Game properties
        self.game_type = "platformer"  # Default game type
        self.theme = "fantasy"  # Default theme
        self.difficulty = "medium"  # Default difficulty
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        
        # Fonts
        self.title_font = pygame.font.SysFont("monospace", 36)
        self.menu_font = pygame.font.SysFont("monospace", 24)
        self.text_font = pygame.font.SysFont("monospace", 16)
        
        # Game state
        self.game_state = "menu"  # menu, setup, playing, game_over
        
        # Player properties
        self.player_x = 100
        self.player_y = 100
        self.player_width = 32
        self.player_height = 32
        self.player_speed = 5
        self.player_jump = -12
        self.player_velocity_y = 0
        self.player_on_ground = False
        self.score = 0
        self.lives = 3
        
        # Game elements
        self.platforms = []
        self.enemies = []
        self.collectibles = []
        self.gravity = 0.5
        
        # Color palettes based on themes
        self.color_palettes = {
            "fantasy": [(0, 48, 59), (255, 207, 64), (37, 188, 36), (173, 64, 255)],
            "sci-fi": [(0, 0, 64), (0, 255, 255), (64, 64, 255), (255, 0, 128)],
            "western": [(101, 67, 33), (255, 201, 102), (222, 177, 45), (170, 117, 25)],
            "horror": [(20, 20, 20), (150, 0, 0), (50, 50, 50), (200, 200, 200)],
            "cyberpunk": [(10, 10, 40), (255, 0, 140), (0, 255, 255), (255, 255, 0)]
        }
        
        # Sprite representations (simple colored rectangles for now)
        self.sprite_types = {
            "player": self.WHITE,
            "enemy": self.RED,
            "platform": self.GREEN,
            "collectible": self.YELLOW
        }
    
    def create_game(self, game_type, theme, difficulty):
        """Setup a new game with the selected parameters"""
        self.game_type = game_type
        self.theme = theme
        self.difficulty = difficulty
        
        # Reset game elements
        self.platforms = []
        self.enemies = []
        self.collectibles = []
        self.score = 0
        self.lives = 3
        
        # Setup based on game type
        if game_type == "platformer":
            self.setup_platformer()
        elif game_type == "shooter":
            self.setup_shooter()
        elif game_type == "puzzle":
            self.setup_puzzle()
        
        # Set game state to playing
        self.game_state = "playing"
    
    def setup_platformer(self):
        """Setup a platformer game"""
        # Create floor
        floor = {"x": 0, "y": self.screen_height - 40, 
                 "width": self.screen_width, "height": 40}
        self.platforms.append(floor)
        
        # Create platforms based on difficulty
        num_platforms = {"easy": 5, "medium": 8, "hard": 12}[self.difficulty]
        
        for _ in range(num_platforms):
            platform = {
                "x": random.randint(0, self.screen_width - 100),
                "y": random.randint(100, self.screen_height - 100),
                "width": random.randint(60, 200),
                "height": 20
            }
            self.platforms.append(platform)
        
        # Create enemies based on difficulty
        num_enemies = {"easy": 2, "medium": 4, "hard": 7}[self.difficulty]
        
        for _ in range(num_enemies):
            enemy = {
                "x": random.randint(100, self.screen_width - 50),
                "y": random.randint(50, self.screen_height - 100),
                "width": 30,
                "height": 30,
                "speed": random.randint(1, 3),
                "direction": random.choice([-1, 1])
            }
            self.enemies.append(enemy)
        
        # Create collectibles
        num_collectibles = 10
        
        for _ in range(num_collectibles):
            collectible = {
                "x": random.randint(50, self.screen_width - 50),
                "y": random.randint(50, self.screen_height - 100),
                "width": 15,
                "height": 15,
                "collected": False
            }
            self.collectibles.append(collectible)
        
        # Position player
        self.player_x = 50
        self.player_y = self.screen_height - 100
        self.player_velocity_y = 0
    
    def setup_shooter(self):
        """Setup a shooter game"""
        # For shooter games, enemies come from top
        self.player_x = self.screen_width // 2 - self.player_width // 2
        self.player_y = self.screen_height - 100
        
        num_enemies = {"easy": 3, "medium": 6, "hard": 10}[self.difficulty]
        
        for _ in range(num_enemies):
            enemy = {
                "x": random.randint(50, self.screen_width - 50),
                "y": random.randint(50, 200),
                "width": 30,
                "height": 30,
                "speed": random.randint(1, 3),
                "direction": random.choice([-1, 1])
            }
            self.enemies.append(enemy)
    
    def setup_puzzle(self):
        """Setup a puzzle game"""
        # Simple puzzle with blocks to arrange
        num_blocks = {"easy": 4, "medium": 6, "hard": 9}[self.difficulty]
        
        for i in range(num_blocks):
            block = {
                "x": 100 + i * 60,
                "y": 100,
                "width": 50,
                "height": 50,
                "dragging": False,
                "goal_x": 100 + i * 60,
                "goal_y": 300
            }
            self.collectibles.append(block)  # Using collectibles as puzzle blocks
    
    def draw_menu(self):
        """Draw the main menu screen"""
        self.screen.fill(self.BLACK)
        
        # Draw title
        title = self.title_font.render("RETRO GAME CREATOR", True, self.WHITE)
        self.screen.blit(title, (self.screen_width//2 - title.get_width()//2, 80))
        
        # Draw game type options
        game_type_text = self.menu_font.render("Game Type:", True, self.WHITE)
        self.screen.blit(game_type_text, (100, 160))
        
        for i, game_type in enumerate(["platformer", "shooter", "puzzle"]):
            color = self.YELLOW if self.game_type == game_type else self.WHITE
            text = self.menu_font.render(game_type, True, color)
            self.screen.blit(text, (250, 160 + i * 30))
        
        # Draw theme options
        theme_text = self.menu_font.render("Theme:", True, self.WHITE)
        self.screen.blit(theme_text, (100, 280))
        
        for i, theme in enumerate(["fantasy", "sci-fi", "western", "horror", "cyberpunk"]):
            color = self.YELLOW if self.theme == theme else self.WHITE
            text = self.menu_font.render(theme, True, color)
            self.screen.blit(text, (250, 280 + i * 30))
        
        # Draw difficulty options
        difficulty_text = self.menu_font.render("Difficulty:", True, self.WHITE)
        self.screen.blit(difficulty_text, (400, 160))
        
        for i, difficulty in enumerate(["easy", "medium", "hard"]):
            color = self.YELLOW if self.difficulty == difficulty else self.WHITE
            text = self.menu_font.render(difficulty, True, color)
            self.screen.blit(text, (550, 160 + i * 30))
        
        # Draw start button
        pygame.draw.rect(self.screen, self.GREEN, (self.screen_width//2 - 75, 400, 150, 40))
        start_text = self.menu_font.render("START GAME", True, self.BLACK)
        self.screen.blit(start_text, (self.screen_width//2 - start_text.get_width()//2, 410))
    
    def draw_game(self):
        """Draw the game elements"""
        # Get the color palette for the current theme
        palette = self.color_palettes.get(self.theme, [(0, 0, 0), (255, 255, 255), (200, 200, 200), (100, 100, 100)])
        
        # Fill background with the theme's background color
        self.screen.fill(palette[0])
        
        # Draw platforms
        for platform in self.platforms:
            pygame.draw.rect(self.screen, palette[2], 
                             (platform["x"], platform["y"], platform["width"], platform["height"]))
        
        # Draw collectibles
        for collectible in self.collectibles:
            if not collectible.get("collected", False):
                pygame.draw.rect(self.screen, palette[1], 
                                 (collectible["x"], collectible["y"], 
                                  collectible["width"], collectible["height"]))
        
        # Draw enemies
        for enemy in self.enemies:
            pygame.draw.rect(self.screen, palette[3], 
                             (enemy["x"], enemy["y"], enemy["width"], enemy["height"]))
        
        # Draw player
        pygame.draw.rect(self.screen, self.WHITE, 
                         (self.player_x, self.player_y, self.player_width, self.player_height))
        
        # Draw score and lives
        score_text = self.text_font.render(f"SCORE: {self.score}", True, self.WHITE)
        self.screen.blit(score_text, (20, 20))
        
        lives_text = self.text_font.render(f"LIVES: {self.lives}", True, self.WHITE)
        self.screen.blit(lives_text, (self.screen_width - 150, 20))
    
    def draw_game_over(self):
        """Draw the game over screen"""
        self.screen.fill(self.BLACK)
        
        game_over_text = self.title_font.render("GAME OVER", True, self.RED)
        self.screen.blit(game_over_text, 
                         (self.screen_width//2 - game_over_text.get_width()//2, 180))
        
        score_text = self.menu_font.render(f"Final Score: {self.score}", True, self.WHITE)
        self.screen.blit(score_text, 
                         (self.screen_width//2 - score_text.get_width()//2, 240))
        
        restart_text = self.menu_font.render("Press SPACE to return to menu", True, self.WHITE)
        self.screen.blit(restart_text, 
                         (self.screen_width//2 - restart_text.get_width()//2, 300))
    
    def update_platformer(self):
        """Update game elements for platformer game"""
        keys = pygame.key.get_pressed()
        
        # Player horizontal movement
        if keys[pygame.K_LEFT]:
            self.player_x -= self.player_speed
        if keys[pygame.K_RIGHT]:
            self.player_x += self.player_speed
        
        # Keep player in bounds
        if self.player_x < 0:
            self.player_x = 0
        if self.player_x > self.screen_width - self.player_width:
            self.player_x = self.screen_width - self.player_width
        
        # Apply gravity
        self.player_velocity_y += self.gravity
        self.player_y += self.player_velocity_y
        
        # Check for platform collisions
        self.player_on_ground = False
        for platform in self.platforms:
            if (self.player_y + self.player_height >= platform["y"] and 
                self.player_y + self.player_height <= platform["y"] + platform["height"] and
                self.player_x + self.player_width > platform["x"] and 
                self.player_x < platform["x"] + platform["width"]):
                
                if self.player_velocity_y > 0:  # Only when falling
                    self.player_y = platform["y"] - self.player_height
                    self.player_velocity_y = 0
                    self.player_on_ground = True
        
        # Check for jumping
        if keys[pygame.K_SPACE] and self.player_on_ground:
            self.player_velocity_y = self.player_jump
            self.player_on_ground = False
        
        # Update enemies
        for enemy in self.enemies:
            enemy["x"] += enemy["speed"] * enemy["direction"]
            
            # Change direction if hitting screen edge
            if enemy["x"] <= 0 or enemy["x"] >= self.screen_width - enemy["width"]:
                enemy["direction"] *= -1
            
            # Check for collision with player
            if (self.player_x < enemy["x"] + enemy["width"] and
                self.player_x + self.player_width > enemy["x"] and
                self.player_y < enemy["y"] + enemy["height"] and
                self.player_y + self.player_height > enemy["y"]):
                
                self.lives -= 1
                self.player_x = 50
                self.player_y = self.screen_height - 100
                
                if self.lives <= 0:
                    self.game_state = "game_over"
        
        # Check for collectible collisions
        for collectible in self.collectibles:
            if (not collectible.get("collected", False) and
                self.player_x < collectible["x"] + collectible["width"] and
                self.player_x + self.player_width > collectible["x"] and
                self.player_y < collectible["y"] + collectible["height"] and
                self.player_y + self.player_height > collectible["y"]):
                
                collectible["collected"] = True
                self.score += 10
        
        # Check if player fell off screen
        if self.player_y > self.screen_height:
            self.lives -= 1
            self.player_x = 50
            self.player_y = self.screen_height - 100
            self.player_velocity_y = 0
            
            if self.lives <= 0:
                self.game_state = "game_over"
    
    def update_shooter(self):
        """Update game elements for shooter game"""
        keys = pygame.key.get_pressed()
        
        # Player horizontal movement
        if keys[pygame.K_LEFT]:
            self.player_x -= self.player_speed
        if keys[pygame.K_RIGHT]:
            self.player_x += self.player_speed
        
        # Keep player in bounds
        if self.player_x < 0:
            self.player_x = 0
        if self.player_x > self.screen_width - self.player_width:
            self.player_x = self.screen_width - self.player_width
            
        # Fire bullets when space is pressed
        if keys[pygame.K_SPACE] and len(self.collectibles) < 10:  # Limit number of bullets
            # Create a bullet (using collectibles list for simplicity)
            bullet = {
                "x": self.player_x + self.player_width // 2 - 5,
                "y": self.player_y,
                "width": 10,
                "height": 20,
                "collected": False
            }
            self.collectibles.append(bullet)
        
        # Update bullets
        for bullet in self.collectibles[:]:
            bullet["y"] -= 10  # Bullet speed
            
            # Remove bullets that go off-screen
            if bullet["y"] < 0:
                self.collectibles.remove(bullet)
                
            # Check for collisions with enemies
            for enemy in self.enemies[:]:
                if (bullet["x"] < enemy["x"] + enemy["width"] and
                    bullet["x"] + bullet["width"] > enemy["x"] and
                    bullet["y"] < enemy["y"] + enemy["height"] and
                    bullet["y"] + bullet["height"] > enemy["y"]):
                    
                    # Remove both bullet and enemy
                    if bullet in self.collectibles:
                        self.collectibles.remove(bullet)
                    if enemy in self.enemies:
                        self.enemies.remove(enemy)
                    self.score += 20
                    
                    # Spawn new enemy if all are destroyed
                    if len(self.enemies) == 0:
                        num_new_enemies = {"easy": 3, "medium": 6, "hard": 10}[self.difficulty]
                        for _ in range(num_new_enemies):
                            enemy = {
                                "x": random.randint(50, self.screen_width - 50),
                                "y": random.randint(50, 200),
                                "width": 30,
                                "height": 30,
                                "speed": random.randint(1, 3) + self.score // 100,  # Increase speed as score increases
                                "direction": random.choice([-1, 1])
                            }
                            self.enemies.append(enemy)
        
        # Update enemies
        for enemy in self.enemies:
            enemy["y"] += 1  # Enemies move down
            enemy["x"] += enemy["speed"] * enemy["direction"]
            
            # Change direction if hitting screen edge
            if enemy["x"] <= 0 or enemy["x"] >= self.screen_width - enemy["width"]:
                enemy["direction"] *= -1
            
            # Check for collision with player or if enemy reached bottom
            if ((self.player_x < enemy["x"] + enemy["width"] and
                 self.player_x + self.player_width > enemy["x"] and
                 self.player_y < enemy["y"] + enemy["height"] and
                 self.player_y + self.player_height > enemy["y"]) or
                enemy["y"] > self.screen_height):
                
                self.lives -= 1
                
                if enemy in self.enemies:
                    self.enemies.remove(enemy)
                    
                # Respawn enemy
                if len(self.enemies) < 3:
                    enemy = {
                        "x": random.randint(50, self.screen_width - 50),
                        "y": random.randint(50, 200),
                        "width": 30,
                        "height": 30,
                        "speed": random.randint(1, 3),
                        "direction": random.choice([-1, 1])
                    }
                    self.enemies.append(enemy)
                    
                if self.lives <= 0:
                    self.game_state = "game_over"
    
    def update_puzzle(self):
        """Update game elements for puzzle game"""
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()
        
        # Handle block dragging
        for block in self.collectibles:
            block_rect = pygame.Rect(block["x"], block["y"], block["width"], block["height"])
            
            if block_rect.collidepoint(mouse_pos):
                if mouse_pressed[0]:  # Left mouse button
                    block["dragging"] = True
                elif block["dragging"]:
                    block["dragging"] = False
                    
                    # Check if block is placed near its goal
                    if (abs(block["x"] - block["goal_x"]) < 30 and
                        abs(block["y"] - block["goal_y"]) < 30):
                        block["x"] = block["goal_x"]
                        block["y"] = block["goal_y"]
                        block["collected"] = True
                        self.score += 10
            
            # Move block with mouse if dragging
            if block["dragging"]:
                block["x"] = mouse_pos[0] - block["width"] // 2
                block["y"] = mouse_pos[1] - block["height"] // 2
        
        # Check if puzzle is solved
        all_collected = all(block.get("collected", False) for block in self.collectibles)
        if all_collected:
            self.score += 100
            self.setup_puzzle()  # Create new puzzle
    
    def handle_menu_click(self, pos):
        """Handle clicks in the menu screen"""
        x, y = pos
        
        # Game type selections
        for i, game_type in enumerate(["platformer", "shooter", "puzzle"]):
            if 250 <= x <= 400 and 160 + i * 30 <= y <= 190 + i * 30:
                self.game_type = game_type
        
        # Theme selections
        for i, theme in enumerate(["fantasy", "sci-fi", "western", "horror", "cyberpunk"]):
            if 250 <= x <= 400 and 280 + i * 30 <= y <= 310 + i * 30:
                self.theme = theme
        
        # Difficulty selections
        for i, difficulty in enumerate(["easy", "medium", "hard"]):
            if 550 <= x <= 650 and 160 + i * 30 <= y <= 190 + i * 30:
                self.difficulty = difficulty
        
        # Start button
        if self.screen_width//2 - 75 <= x <= self.screen_width//2 + 75 and 400 <= y <= 440:
            self.create_game(self.game_type, self.theme, self.difficulty)
    
    def run(self):
        """Main game loop"""
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.game_state == "menu":
                        self.handle_menu_click(event.pos)
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.game_state == "playing":
                            self.game_state = "menu"
                        else:
                            running = False
                    
                    if event.key == pygame.K_SPACE and self.game_state == "game_over":
                        self.game_state = "menu"
            
            # Update game state
            if self.game_state == "playing":
                if self.game_type == "platformer":
                    self.update_platformer()
                elif self.game_type == "shooter":
                    self.update_shooter()
                elif self.game_type == "puzzle":
                    self.update_puzzle()
            
            # Draw the appropriate screen
            if self.game_state == "menu":
                self.draw_menu()
            elif self.game_state == "playing":
                self.draw_game()
            elif self.game_state == "game_over":
                self.draw_game_over()
            
            # Update the display
            pygame.display.flip()
            self.clock.tick(self.fps)
        
        pygame.quit()

if __name__ == "__main__":
    game_creator = RetroGameCreator()
    game_creator.run()
