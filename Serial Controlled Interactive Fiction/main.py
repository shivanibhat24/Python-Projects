#!/usr/bin/env python3
"""
Serial Port Interactive Fiction Game
-----------------------------------
A text adventure game that communicates via serial port, inspired by classic
games like Zork but with ASCII art and interactive puzzles.
"""

import serial
import time
import textwrap
import random
import json
import os
from threading import Thread

class SerialTerminalGame:
    def __init__(self, port='/dev/ttyUSB0', baud_rate=9600):
        """Initialize the game with serial port settings"""
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None
        self.connected = False
        self.game_running = False
        
        # Game state
        self.current_location = "entrance"
        self.inventory = []
        self.game_state = {
            "lantern_lit": False,
            "trapdoor_open": False,
            "puzzle_solved": False,
            "key_found": False,
            "exit_unlocked": False
        }
        
        # Initialize game world
        self.init_game_world()
    
    def init_game_world(self):
        """Initialize the game world with locations, items, and descriptions"""
        self.locations = {
            "entrance": {
                "name": "Cave Entrance",
                "description": "You stand at the mouth of a dark, foreboding cave. Sunlight streams in from behind you, but darkness lies ahead.",
                "exits": {"north": "cavern", "east": "forest_path"},
                "items": ["lantern"],
                "ascii_art": """
                    /\\    /\\
                   /  \\__/  \\
                  /          \\
                 (            )
                  \\  ______  /
                   \\/      \\/
                    |      |
                    |      |
                    |      |
                """
            },
            "forest_path": {
                "name": "Forest Path",
                "description": "A narrow path winds through ancient trees. The cave entrance lies to the west.",
                "exits": {"west": "entrance", "east": "clearing"},
                "items": ["stick"],
                "ascii_art": """
                     /\\  /\\  /\\
                    /  \\/  \\/  \\
                   /            \\
                  /              \\
                 /                \\
                 |   ----------   |
                 |   |        |   |
                 |   |        |   |
                """
            },
            "clearing": {
                "name": "Forest Clearing",
                "description": "A peaceful clearing bathed in dappled sunlight. There's a strange symbol carved into a tree stump.",
                "exits": {"west": "forest_path"},
                "items": ["mushroom"],
                "ascii_art": """
                        .    *    .
                      *    *    *  
                    .   *   .    * 
                     \\       /
                      \\     /
                       \\___/
                       |   |
                       |___|
                """
            },
            "cavern": {
                "name": "Main Cavern",
                "description": "A vast cavern stretches before you. The air is damp and cool. There's a strange marking on the floor.",
                "exits": {"south": "entrance", "north": "passage", "east": "alcove"},
                "items": ["rock"],
                "requires_light": True,
                "ascii_art": """
                     ___________
                    /           \\
                   /             \\
                  |               |
                  |               |
                  |               |
                  |     _____     |
                  |    |     |    |
                  |____|     |____|
                """
            },
            "alcove": {
                "name": "Hidden Alcove",
                "description": "A small alcove cut into the cavern wall. It seems to be a natural shelf of some kind.",
                "exits": {"west": "cavern"},
                "items": ["key"],
                "requires_light": True,
                "ascii_art": """
                  ______________
                 /              \\
                |                |
                |      ____      |
                |     /    \\     |
                |    |      |    |
                |    |      |    |
                |____|      |____|
                """
            },
            "passage": {
                "name": "Narrow Passage",
                "description": "A narrow passage that leads deeper into the earth. The ceiling is low, and you need to crouch to move through.",
                "exits": {"south": "cavern", "north": "chamber"},
                "items": [],
                "requires_light": True,
                "ascii_art": """
                 __________________
                |                  |
                |                  |
                |__________________|
                |                  |
                |                  |
                |__________________|
                """
            },
            "chamber": {
                "name": "Underground Chamber",
                "description": "A large underground chamber with strange carvings on the walls. In the center is what appears to be a puzzle mechanism.",
                "exits": {"south": "passage", "down": "crypt"},
                "items": [],
                "requires_light": True,
                "trapdoor_hidden": True,
                "ascii_art": """
                 _____________________
                /                     \\
               /                       \\
              |                         |
              |                         |
              |           []            |
              |          [  ]           |
              |         [    ]          |
              |__________________________|
                """
            },
            "crypt": {
                "name": "Ancient Crypt",
                "description": "A crypt that seems untouched for centuries. There's an ornate door on the far wall with a keyhole.",
                "exits": {"up": "chamber", "door": "exit"},
                "items": ["amulet"],
                "requires_light": True,
                "door_locked": True,
                "ascii_art": """
                 _____________________
                /                     \\
               /                       \\
              |      ______________     |
              |     |              |    |
              |     |              |    |
              |     |              |    |
              |     |______________|    |
              |___________________________
                """
            },
            "exit": {
                "name": "Exit",
                "description": "You've found your way out of the cave system! The sunlight is blinding after your time underground.",
                "exits": {},
                "items": [],
                "is_end": True,
                "ascii_art": """
                    \\               /
                     \\             /
                      \\           /
                       \\_________/
                       |         |
                       |         |
                       |         |
                       |_________|
                """
            }
        }
        
        self.items = {
            "lantern": {
                "name": "lantern",
                "description": "An old brass lantern. It's not lit.",
                "can_take": True,
                "usable": True
            },
            "rock": {
                "name": "rock",
                "description": "A smooth, round rock. Might be useful for something.",
                "can_take": True,
                "usable": True
            },
            "stick": {
                "name": "stick",
                "description": "A sturdy wooden stick, about arm's length.",
                "can_take": True,
                "usable": True
            },
            "key": {
                "name": "key",
                "description": "An ancient brass key with unusual teeth.",
                "can_take": True,
                "usable": True
            },
            "mushroom": {
                "name": "mushroom",
                "description": "A small glowing mushroom. It gives off a faint blue light.",
                "can_take": True,
                "usable": True
            },
            "amulet": {
                "name": "amulet",
                "description": "A strange amulet with symbols that match those found on the walls.",
                "can_take": True,
                "usable": True
            }
        }

    def connect(self):
        """Attempt to establish a serial connection"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            self.connected = True
            self.send_message("Serial connection established!\r\n")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect to serial port: {e}")
            return False
    
    def disconnect(self):
        """Close the serial connection"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.connected = False
        self.game_running = False
    
    def send_message(self, message):
        """Send a message over the serial connection"""
        if not self.connected or not self.serial_conn:
            return
        
        # Format message with word wrapping
        wrapped_lines = []
        for line in message.split('\n'):
            if line.strip():
                wrapped = textwrap.fill(line, width=70)
                wrapped_lines.append(wrapped)
            else:
                wrapped_lines.append('')
        
        formatted_message = '\n'.join(wrapped_lines) + '\r\n'
        
        try:
            self.serial_conn.write(formatted_message.encode('utf-8'))
            self.serial_conn.flush()
        except serial.SerialException as e:
            print(f"Error sending message: {e}")
            self.disconnect()
    
    def receive_command(self):
        """Read a command from the serial connection"""
        if not self.connected or not self.serial_conn:
            return None
        
        buffer = ""
        while self.connected and self.game_running:
            try:
                if self.serial_conn.in_waiting > 0:
                    char = self.serial_conn.read(1).decode('utf-8')
                    if char == '\r' or char == '\n':
                        if buffer:
                            return buffer.strip()
                    else:
                        buffer += char
                        # Echo character back to terminal
                        self.serial_conn.write(char.encode('utf-8'))
                        self.serial_conn.flush()
            except serial.SerialException as e:
                print(f"Error receiving command: {e}")
                self.disconnect()
                return None
            time.sleep(0.01)
        
        return None
    
    def start_game(self):
        """Start the game"""
        if not self.connected:
            if not self.connect():
                return
        
        self.game_running = True
        
        # Display welcome message
        welcome_text = """
        =================================================
                    THE CAVERNS OF MYSTERY
        =================================================
        
        You are an adventurer exploring a mysterious cave system
        rumored to contain ancient treasures and forgotten knowledge.
        
        Use simple commands like:
        - look: examine your surroundings
        - go [direction]: move in a direction (north, south, east, west, up, down)
        - take [item]: pick up an item
        - drop [item]: drop an item from your inventory
        - inventory: check what you're carrying
        - use [item]: use an item in your inventory
        - use [item] on [target]: use an item on a target
        - examine [item/feature]: look at something more closely
        - help: show this help message
        - quit: exit the game
        
        =================================================
        Type 'look' to begin your adventure.
        """
        
        self.send_message(welcome_text)
        
        # Game loop
        while self.connected and self.game_running:
            command = self.receive_command()
            if command:
                self.process_command(command)
    
    def process_command(self, command):
        """Process a game command"""
        command = command.lower().strip()
        
        # Split the command into words
        words = command.split()
        
        if not words:
            self.send_message("Please enter a command.")
            return
        
        action = words[0]
        
        # Handle simple commands
        if action == "quit":
            self.send_message("Thanks for playing! Goodbye.")
            self.game_running = False
            return
        
        elif action == "help":
            self.show_help()
            return
        
        elif action == "look":
            self.look()
            return
        
        elif action == "inventory" or action == "i":
            self.show_inventory()
            return
        
        # Handle commands with targets
        if len(words) > 1:
            target = " ".join(words[1:])
            
            if action == "go":
                self.go(target)
            elif action == "take":
                self.take(target)
            elif action == "drop":
                self.drop(target)
            elif action == "examine":
                self.examine(target)
            elif action == "use":
                if len(words) > 3 and words[2] == "on":
                    # Format: use [item] on [target]
                    item = words[1]
                    target = " ".join(words[3:])
                    self.use_on(item, target)
                else:
                    # Format: use [item]
                    self.use(target)
            else:
                self.send_message(f"I don't understand '{command}'.")
        else:
            self.send_message(f"I don't understand '{command}'.")
    
    def show_help(self):
        """Display help information"""
        help_text = """
        Available commands:
        - look: examine your surroundings
        - go [direction]: move in a direction (north, south, east, west, up, down)
        - take [item]: pick up an item
        - drop [item]: drop an item from your inventory
        - inventory: check what you're carrying
        - use [item]: use an item in your inventory
        - use [item] on [target]: use an item on a target
        - examine [item/feature]: look at something more closely
        - help: show this help message
        - quit: exit the game
        """
        self.send_message(help_text)
    
    def look(self):
        """Look around the current location"""
        current = self.locations[self.current_location]
        
        # Check if location requires light and player doesn't have it
        if current.get("requires_light", False) and not self.game_state["lantern_lit"]:
            self.send_message("It's too dark to see anything. You need a light source.")
            return
        
        # Basic location information
        message = f"\n{current['ascii_art']}\n\n{current['name']}\n"
        message += f"{current['description']}\n"
        
        # List items in the location
        if current["items"]:
            item_list = [self.items[item]["name"] for item in current["items"]]
            message += f"\nYou can see: {', '.join(item_list)}.\n"
        
        # List exits
        if current["exits"]:
            exit_list = list(current["exits"].keys())
            message += f"\nExits: {', '.join(exit_list)}.\n"
            
            # Special case for trapdoor
            if self.current_location == "chamber" and self.game_state["trapdoor_open"]:
                message += "There's an open trapdoor in the floor.\n"
            elif self.current_location == "chamber" and not current.get("trapdoor_hidden", False):
                message += "There's a closed trapdoor in the floor.\n"
                
            # Special case for locked door
            if self.current_location == "crypt" and self.game_state["exit_unlocked"]:
                message += "The door is now unlocked.\n"
            elif self.current_location == "crypt" and current.get("door_locked", False):
                message += "The ornate door appears to be locked.\n"
        
        self.send_message(message)
    
    def go(self, direction):
        """Move in a direction"""
        current = self.locations[self.current_location]
        
        # Special case for trapdoor in chamber
        if self.current_location == "chamber" and direction == "down":
            if self.game_state["trapdoor_open"]:
                self.current_location = "crypt"
                self.look()
                return
            else:
                self.send_message("You can't go that way. There's no obvious path down.")
                return
        
        # Special case for locked door in crypt
        if self.current_location == "crypt" and direction == "door":
            if self.game_state["exit_unlocked"]:
                self.current_location = "exit"
                self.look()
                self.win_game()
                return
            else:
                self.send_message("The door is locked. You'll need a key to open it.")
                return
        
        # Standard movement
        if direction in current["exits"]:
            self.current_location = current["exits"][direction]
            self.look()
        else:
            self.send_message(f"You can't go {direction}.")
    
    def take(self, item_name):
        """Pick up an item"""
        current = self.locations[self.current_location]
        
        # Check if location requires light and player doesn't have it
        if current.get("requires_light", False) and not self.game_state["lantern_lit"]:
            self.send_message("It's too dark to find anything. You need a light source.")
            return
        
        for item_id in current["items"]:
            if self.items[item_id]["name"] == item_name:
                if self.items[item_id]["can_take"]:
                    current["items"].remove(item_id)
                    self.inventory.append(item_id)
                    
                    # Special case for key
                    if item_id == "key":
                        self.game_state["key_found"] = True
                    
                    self.send_message(f"You take the {item_name}.")
                else:
                    self.send_message(f"You can't take the {item_name}.")
                return
        
        self.send_message(f"There's no {item_name} here.")
    
    def drop(self, item_name):
        """Drop an item"""
        for item_id in self.inventory:
            if self.items[item_id]["name"] == item_name:
                self.inventory.remove(item_id)
                self.locations[self.current_location]["items"].append(item_id)
                self.send_message(f"You drop the {item_name}.")
                return
        
        self.send_message(f"You don't have a {item_name}.")
    
    def show_inventory(self):
        """Show the player's inventory"""
        if not self.inventory:
            self.send_message("Your inventory is empty.")
            return
        
        item_list = [self.items[item]["name"] for item in self.inventory]
        self.send_message(f"You are carrying: {', '.join(item_list)}.")
    
    def examine(self, target):
        """Examine something more closely"""
        current = self.locations[self.current_location]
        
        # Check if location requires light and player doesn't have it
        if current.get("requires_light", False) and not self.game_state["lantern_lit"]:
            self.send_message("It's too dark to see anything clearly. You need a light source.")
            return
        
        # Check inventory
        for item_id in self.inventory:
            if self.items[item_id]["name"] == target:
                self.send_message(self.items[item_id]["description"])
                return
        
        # Check items in location
        for item_id in current["items"]:
            if self.items[item_id]["name"] == target:
                self.send_message(self.items[item_id]["description"])
                return
        
        # Special cases by location
        if self.current_location == "chamber" and target in ["puzzle", "mechanism", "center"]:
            self.send_message("The puzzle consists of a series of concentric stone rings that can be rotated. There are symbols carved into each ring that seem to need alignment.")
            return
        
        if self.current_location == "clearing" and target in ["symbol", "tree", "stump"]:
            self.send_message("The symbol resembles a key inside a circle. Below it are the words: 'Light before darkness, knowledge before passage.'")
            return
        
        if self.current_location == "cavern" and target in ["marking", "floor"]:
            self.send_message("The marking appears to be a circular pattern with symbols matching those in the chamber puzzle.")
            return
        
        if self.current_location == "crypt" and target in ["door", "keyhole"]:
            self.send_message("The door is made of stone with intricate carvings. The keyhole appears to be shaped for a specific key.")
            return
        
        self.send_message(f"You don't see anything special about the {target}.")
    
    def use(self, item_name):
        """Use an item"""
        # Check if player has the item
        item_id = None
        for inv_item in self.inventory:
            if self.items[inv_item]["name"] == item_name:
                item_id = inv_item
                break
        
        if not item_id:
            self.send_message(f"You don't have a {item_name}.")
            return
        
        # Handle specific item usages
        if item_id == "lantern":
            self.game_state["lantern_lit"] = not self.game_state["lantern_lit"]
            if self.game_state["lantern_lit"]:
                self.send_message("You light the lantern. A warm glow illuminates the area around you.")
            else:
                self.send_message("You extinguish the lantern.")
            return
            
        elif item_id == "key" and self.current_location == "crypt":
            self.game_state["exit_unlocked"] = True
            self.send_message("You insert the key into the lock and turn it. The door unlocks with a satisfying click.")
            return
            
        elif item_id == "mushroom":
            self.send_message("The mushroom glows a bit brighter when handled, but it's not enough to light your way in dark areas.")
            return
            
        elif item_id == "amulet" and self.current_location == "chamber":
            if not self.game_state["puzzle_solved"]:
                self.send_message("You hold up the amulet to the puzzle. The symbols on the amulet glow, revealing the correct configuration for the puzzle!")
                self.game_state["puzzle_solved"] = True
                self.game_state["trapdoor_open"] = True
                self.locations["chamber"]["trapdoor_hidden"] = False
                return
        
        self.send_message(f"You're not sure how to use the {item_name} here.")
    
    def use_on(self, item_name, target):
        """Use an item on a target"""
        # Check if player has the item
        item_id = None
        for inv_item in self.inventory:
            if self.items[inv_item]["name"] == item_name:
                item_id = inv_item
                break
        
        if not item_id:
            self.send_message(f"You don't have a {item_name}.")
            return
        
        # Handle specific item usages
        if item_id == "stick" and target in ["puzzle", "mechanism"] and self.current_location == "chamber":
            self.send_message("You poke at the puzzle with the stick. It rotates slightly, but you can't get a good grip to solve it properly.")
            return
            
        elif item_id == "key" and target in ["door", "keyhole"] and self.current_location == "crypt":
            self.game_state["exit_unlocked"] = True
            self.send_message("You insert the key into the keyhole and turn it. The door unlocks with a satisfying click.")
            return
            
        elif item_id == "rock" and target in ["puzzle", "mechanism"] and self.current_location == "chamber":
            self.send_message("You try to use the rock on the puzzle, but it doesn't fit anywhere useful.")
            return
        
        self.send_message(f"You can't use the {item_name} on the {target}.")
    
    def win_game(self):
        """Handle game completion"""
        end_message = """
        =============================================
                    CONGRATULATIONS!
        =============================================
        
        You have successfully navigated the Caverns of Mystery
        and found your way to freedom! The sunlight feels
        warm on your face after your underground adventure.
        
        Thanks for playing!
        
        Would you like to play again? (yes/no)
        """
        self.send_message(end_message)
        
        # Wait for player response
        while self.connected and self.game_running:
            response = self.receive_command()
            if response:
                response = response.lower().strip()
                if response == "yes" or response == "y":
                    self.reset_game()
                    return
                elif response == "no" or response == "n":
                    self.send_message("Thanks for playing! Goodbye.")
                    self.game_running = False
                    return
                else:
                    self.send_message("Please answer yes or no.")
    
    def reset_game(self):
        """Reset the game to its initial state"""
        self.current_location = "entrance"
        self.inventory = []
        self.game_state = {
            "lantern_lit": False,
            "trapdoor_open": False,
            "puzzle_solved": False,
            "key_found": False,
            "exit_unlocked": False
        }
        
        # Reinitialize game world to reset item positions
        self.init_game_world()
        
        self.send_message("Starting a new game!\n")
        self.look()

def main():
    """Main function to run the game"""
    # Get port from command line argument or use default
    import sys
    port = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'
    
    print(f"Starting Serial Terminal Game on port {port}")
    game = SerialTerminalGame(port=port)
    
    try:
        game.start_game()
    except KeyboardInterrupt:
        print("Game interrupted by user.")
    finally:
        game.disconnect()
        print("Game ended. Serial connection closed.")

if __name__ == "__main__":
    main()
