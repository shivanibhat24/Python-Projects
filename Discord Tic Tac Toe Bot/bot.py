import discord
from discord.ext import commands
import numpy as np
import random
import pickle
import os
from datetime import datetime

# Bot configuration
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.2
EXPLORATION_DECAY = 0.995
MIN_EXPLORATION_RATE = 0.01

# File to save Q-table
Q_TABLE_FILE = 'q_table.pkl'

# Game state
games = {}

class TicTacToe:
    def __init__(self, player_id):
        self.board = [' ' for _ in range(9)]
        self.player_id = player_id
        self.bot_symbol = 'O'
        self.player_symbol = 'X'
        self.current_turn = self.player_symbol  # Player goes first
        self.winner = None
        self.game_over = False
        self.moves_history = []
        
    def make_move(self, position, symbol):
        if 0 <= position < 9 and self.board[position] == ' ' and not self.game_over:
            self.board[position] = symbol
            self.moves_history.append((self.get_board_state(), position))
            self.current_turn = self.bot_symbol if symbol == self.player_symbol else self.player_symbol
            self.check_winner()
            return True
        return False
    
    def get_board_state(self):
        # Convert board to a tuple (immutable for dictionary key)
        return tuple(self.board)
    
    def get_available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']
    
    def check_winner(self):
        # Check for winning combinations
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
        for combo in winning_combinations:
            if self.board[combo[0]] != ' ' and self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]]:
                self.winner = self.board[combo[0]]
                self.game_over = True
                return
        
        # Check for tie
        if ' ' not in self.board:
            self.game_over = True
    
    def render_board(self):
        board_str = "```\n"
        for i in range(0, 9, 3):
            board_str += f" {self.board[i]} | {self.board[i+1]} | {self.board[i+2]} \n"
            if i < 6:
                board_str += "---+---+---\n"
        board_str += "```"
        
        # Add position reference
        ref_str = "```\n Position reference:\n"
        for i in range(0, 9, 3):
            ref_str += f" {i} | {i+1} | {i+2} \n"
            if i < 6:
                ref_str += "---+---+---\n"
        ref_str += "```"
        
        return board_str + "\n" + ref_str
    
    def get_reward(self, is_bot_move=True):
        if not self.game_over:
            return 0
        
        if self.winner == self.bot_symbol:
            return 1  # Bot wins
        elif self.winner == self.player_symbol:
            return -1  # Bot loses
        else:
            return 0.5  # Tie
            
class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.exploration_rate = EXPLORATION_RATE
        self.load_q_table()
    
    def load_q_table(self):
        if os.path.exists(Q_TABLE_FILE):
            try:
                with open(Q_TABLE_FILE, 'rb') as f:
                    self.q_table = pickle.load(f)
                print(f"Loaded Q-table with {len(self.q_table)} states")
            except Exception as e:
                print(f"Error loading Q-table: {e}")
        else:
            print("No Q-table found, starting fresh")
    
    def save_q_table(self):
        try:
            with open(Q_TABLE_FILE, 'wb') as f:
                pickle.dump(self.q_table, f)
            print(f"Saved Q-table with {len(self.q_table)} states")
        except Exception as e:
            print(f"Error saving Q-table: {e}")
    
    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        return self.q_table[state][action]
    
    def choose_action(self, game):
        state = game.get_board_state()
        available_moves = game.get_available_moves()
        
        if not available_moves:
            return None
        
        # Exploration: random move
        if random.random() < self.exploration_rate:
            return random.choice(available_moves)
        
        # Exploitation: best move according to Q-table
        best_value = float('-inf')
        best_actions = []
        
        for action in available_moves:
            q_value = self.get_q_value(state, action)
            if q_value > best_value:
                best_value = q_value
                best_actions = [action]
            elif q_value == best_value:
                best_actions.append(action)
        
        return random.choice(best_actions)
    
    def update_q_values(self, game):
        if not game.moves_history:
            return
        
        # Get the final reward
        final_reward = game.get_reward()
        
        # Update Q-values in reverse order
        for state, action in reversed(game.moves_history):
            if game.board[action] != game.bot_symbol:
                continue  # Skip player moves
                
            # Get current Q-value
            current_q = self.get_q_value(state, action)
            
            # Update Q-value
            self.q_table[state][action] = current_q + LEARNING_RATE * (final_reward - current_q)
        
        # Decay exploration rate
        self.exploration_rate = max(MIN_EXPLORATION_RATE, self.exploration_rate * EXPLORATION_DECAY)

# Initialize the agent
agent = QLearningAgent()

# Bot events
@bot.event
async def on_ready():
    print(f'Bot is ready! Logged in as {bot.user}')
    print(f'Current exploration rate: {agent.exploration_rate:.4f}')

@bot.command(name='tictactoe', aliases=['ttt'])
async def tictactoe(ctx):
    """Start a new game of Tic Tac Toe against the bot"""
    player_id = ctx.author.id
    
    if player_id in games:
        await ctx.send("You already have a game in progress! Use `!move <position>` to play or `!resign` to quit.")
        return
    
    # Create a new game
    games[player_id] = TicTacToe(player_id)
    
    await ctx.send(f"🎮 **New Tic Tac Toe game!**\nYou are X, I am O. You go first!\nUse `!move <position>` to place your mark.\n\n{games[player_id].render_board()}")

@bot.command(name='move')
async def move(ctx, position: int = None):
    """Make a move in your Tic Tac Toe game"""
    player_id = ctx.author.id
    
    if player_id not in games:
        await ctx.send("You don't have a game in progress! Start one with `!tictactoe`")
        return
    
    if position is None:
        await ctx.send("Please specify a position (0-8)")
        return
    
    game = games[player_id]
    
    # Player's turn
    if game.current_turn == game.player_symbol:
        if game.make_move(position, game.player_symbol):
            await ctx.send(f"You placed an X at position {position}.\n\n{game.render_board()}")
            
            if game.game_over:
                if game.winner == game.player_symbol:
                    await ctx.send("🎉 **You win!** Congratulations!")
                else:
                    await ctx.send("😐 **It's a tie!**")
                
                # Update Q-values
                agent.update_q_values(game)
                agent.save_q_table()
                del games[player_id]
                return
            
            # Bot's turn
            bot_move = agent.choose_action(game)
            game.make_move(bot_move, game.bot_symbol)
            
            await ctx.send(f"I place an O at position {bot_move}.\n\n{game.render_board()}")
            
            if game.game_over:
                if game.winner == game.bot_symbol:
                    await ctx.send("😈 **I win!** Better luck next time!")
                else:
                    await ctx.send("😐 **It's a tie!**")
                
                # Update Q-values
                agent.update_q_values(game)
                agent.save_q_table()
                del games[player_id]
        else:
            await ctx.send("Invalid move! Choose an empty position between 0-8.")
    else:
        await ctx.send("It's not your turn!")

@bot.command(name='resign')
async def resign(ctx):
    """Resign from your current Tic Tac Toe game"""
    player_id = ctx.author.id
    
    if player_id in games:
        del games[player_id]
        await ctx.send("Game resigned. Start a new one with `!tictactoe`")
    else:
        await ctx.send("You don't have a game in progress!")

@bot.command(name='stats')
async def stats(ctx):
    """Show bot statistics"""
    num_states = len(agent.q_table)
    total_values = sum(len(actions) for actions in agent.q_table.values())
    
    stats_message = (
        f"🤖 **Bot Statistics**\n"
        f"Learning rate: {LEARNING_RATE}\n"
        f"Exploration rate: {agent.exploration_rate:.4f}\n"
        f"States learned: {num_states}\n"
        f"Total Q-values: {total_values}\n"
        f"Last save: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    await ctx.send(stats_message)

# Run the bot
if __name__ == "__main__":
    # Replace TOKEN with your Discord bot token
    bot.run('YOUR_DISCORD_BOT_TOKEN')
