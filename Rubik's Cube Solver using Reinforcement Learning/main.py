import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import torch
import torch.optim as optim
from tensordict import TensorDict

class RubiksCubeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rubik's Cube RL Solver")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.is_training = False
        self.train_thread = None
        self.episodes_trained = 0
        self.max_episodes = 1000
        self.current_state = None
        
        # Create environment and agent
        self.env = RubiksCubeEnv()  # Assuming this class exists
        self.env_wrapper = EnvWrapper(self.env)  # Assuming this class exists
        
        # Create DQN agent
        state_dim = self.env._get_state().shape[0]
        action_dim = self.env.action_space.n
        self.agent = DQNAgent(state_dim, action_dim)  # Assuming this class exists
        
        # Reset environment
        self.current_state = self.env_wrapper.reset()
        
        # Set up the UI layout
        self.setup_ui()
        
        # Start the UI update loop
        self.update_ui()
    
    def setup_ui(self):
        # Create frames
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        viz_frame = ttk.Frame(self.root, padding=10)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Control panel
        ttk.Label(control_frame, text="Training Parameters", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Episodes input
        episodes_frame = ttk.Frame(control_frame)
        episodes_frame.pack(fill=tk.X, pady=5)
        ttk.Label(episodes_frame, text="Episodes:").pack(side=tk.LEFT)
        self.episodes_var = tk.StringVar(value="1000")
        ttk.Entry(episodes_frame, textvariable=self.episodes_var, width=10).pack(side=tk.RIGHT)
        
        # Learning rate input
        lr_frame = ttk.Frame(control_frame)
        lr_frame.pack(fill=tk.X, pady=5)
        ttk.Label(lr_frame, text="Learning Rate:").pack(side=tk.LEFT)
        self.lr_var = tk.StringVar(value="0.001")
        ttk.Entry(lr_frame, textvariable=self.lr_var, width=10).pack(side=tk.RIGHT)
        
        # Gamma input
        gamma_frame = ttk.Frame(control_frame)
        gamma_frame.pack(fill=tk.X, pady=5)
        ttk.Label(gamma_frame, text="Discount Factor (γ):").pack(side=tk.LEFT)
        self.gamma_var = tk.StringVar(value="0.99")
        ttk.Entry(gamma_frame, textvariable=self.gamma_var, width=10).pack(side=tk.RIGHT)
        
        # Scramble moves
        scramble_frame = ttk.Frame(control_frame)
        scramble_frame.pack(fill=tk.X, pady=5)
        ttk.Label(scramble_frame, text="Scramble Moves:").pack(side=tk.LEFT)
        self.scramble_var = tk.StringVar(value="5")
        ttk.Entry(scramble_frame, textvariable=self.scramble_var, width=10).pack(side=tk.RIGHT)
        
        # Status display
        status_frame = ttk.LabelFrame(control_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=10)
        
        # Episodes
        episodes_status = ttk.Frame(status_frame)
        episodes_status.pack(fill=tk.X, pady=2)
        ttk.Label(episodes_status, text="Episodes Trained:").pack(side=tk.LEFT)
        self.episodes_label = ttk.Label(episodes_status, text="0")
        self.episodes_label.pack(side=tk.RIGHT)
        
        # Epsilon
        epsilon_status = ttk.Frame(status_frame)
        epsilon_status.pack(fill=tk.X, pady=2)
        ttk.Label(epsilon_status, text="Exploration Rate (ε):").pack(side=tk.LEFT)
        self.epsilon_label = ttk.Label(epsilon_status, text="1.0000")
        self.epsilon_label.pack(side=tk.RIGHT)
        
        # Avg reward
        reward_status = ttk.Frame(status_frame)
        reward_status.pack(fill=tk.X, pady=2)
        ttk.Label(reward_status, text="Avg Reward (10 ep):").pack(side=tk.LEFT)
        self.reward_label = ttk.Label(reward_status, text="0.00")
        self.reward_label.pack(side=tk.RIGHT)
        
        # Avg loss
        loss_status = ttk.Frame(status_frame)
        loss_status.pack(fill=tk.X, pady=2)
        ttk.Label(loss_status, text="Avg Loss (10 ep):").pack(side=tk.LEFT)
        self.loss_label = ttk.Label(loss_status, text="0.0000")
        self.loss_label.pack(side=tk.RIGHT)
        
        # Progress bar
        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        ttk.Label(progress_frame, text="Progress:").pack(anchor=tk.W)
        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100).pack(fill=tk.X)
        
        # Action buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        # Training button
        self.train_button = ttk.Button(btn_frame, text="Start Training", command=self.toggle_training)
        self.train_button.pack(fill=tk.X, pady=2)
        
        # Reset button
        ttk.Button(btn_frame, text="Reset Cube", command=self.reset_cube).pack(fill=tk.X, pady=2)
        
        # Scramble button
        ttk.Button(btn_frame, text="Scramble Cube", command=self.scramble_cube).pack(fill=tk.X, pady=2)
        
        # Solve button
        ttk.Button(btn_frame, text="Solve Cube", command=self.solve_cube).pack(fill=tk.X, pady=2)
        
        # Save/Load model buttons
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, pady=10)
        ttk.Button(model_frame, text="Save Model", command=self.save_model).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(model_frame, text="Load Model", command=self.load_model).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=2)
        
        # Visualization area
        viz_notebook = ttk.Notebook(viz_frame)
        viz_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Cube visualization tab
        cube_tab = ttk.Frame(viz_notebook)
        viz_notebook.add(cube_tab, text="Cube")
        
        self.cube_canvas = tk.Canvas(cube_tab, width=500, height=500, bg="white")
        self.cube_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Training plots tab
        plots_tab = ttk.Frame(viz_notebook)
        viz_notebook.add(plots_tab, text="Training Metrics")
        
        # Set up matplotlib figure
        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_title("Episode Rewards")
        self.ax1.set_xlabel("Episode")
        self.ax1.set_ylabel("Reward")
        self.reward_plot, = self.ax1.plot([], [], 'b-')
        
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title("Training Loss")
        self.ax2.set_xlabel("Training Step")
        self.ax2.set_ylabel("Loss")
        self.loss_plot, = self.ax2.plot([], [], 'r-')
        
        # Add figure to tkinter canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plots_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def update_ui(self):
        # Update cube visualization
        self.env.render(self.cube_canvas)
        
        # Update status labels
        self.episodes_label.config(text=str(self.episodes_trained))
        self.epsilon_label.config(text=f"{self.agent.epsilon:.4f}")
        
        # Update progress bar
        if self.is_training and self.max_episodes > 0:
            progress = (self.episodes_trained / self.max_episodes) * 100
            self.progress_var.set(progress)
        
        # Update plots
        if len(self.agent.rewards) > 0:
            avg_reward = sum(self.agent.rewards[-10:]) / min(len(self.agent.rewards), 10)
            self.reward_label.config(text=f"{avg_reward:.2f}")
            
            x = list(range(len(self.agent.rewards)))
            self.reward_plot.set_data(x, self.agent.rewards)
            self.ax1.relim()
            self.ax1.autoscale_view()
        
        if len(self.agent.losses) > 0:
            avg_loss = sum(self.agent.losses[-10:]) / min(len(self.agent.losses), 10)
            self.loss_label.config(text=f"{avg_loss:.4f}")
            
            x = list(range(len(self.agent.losses)))
            self.loss_plot.set_data(x, self.agent.losses)
            self.ax2.relim()
            self.ax2.autoscale_view()
        
        self.canvas.draw()
        
        # Schedule the next UI update
        self.root.after(100, self.update_ui)
    
    def reset_cube(self):
        self.current_state = self.env_wrapper.reset()
        self.env.render(self.cube_canvas)
    
    def scramble_cube(self):
        moves = int(self.scramble_var.get())
        self.env.scramble(moves)
        self.current_state = TensorDict(
            {"observation": torch.tensor(self.env._get_state(), dtype=torch.float32).unsqueeze(0)},
            batch_size=[1]
        )
        self.env.render(self.cube_canvas)
    
    def toggle_training(self):
        if self.is_training:
            self.stop_training()
        else:
            self.start_training()
    
    def start_training(self):
        if self.is_training:
            return
        
        self.is_training = True
        self.train_button.config(text="Stop Training")
        
        # Get training parameters
        self.max_episodes = int(self.episodes_var.get())
        learning_rate = float(self.lr_var.get())
        gamma = float(self.gamma_var.get())
        scramble_moves = int(self.scramble_var.get())
        
        # Update agent parameters
        self.agent.optimizer = optim.Adam(self.agent.q_network.parameters(), lr=learning_rate)
        self.agent.gamma = gamma
        
        # Start training in a separate thread
        self.train_thread = threading.Thread(target=self.train, args=(scramble_moves,))
        self.train_thread.daemon = True
        self.train_thread.start()
    
    def stop_training(self):
        self.is_training = False
        self.train_button.config(text="Start Training")
        
        if self.train_thread:
            self.train_thread.join(timeout=1.0)
            self.train_thread = None
    
    def train(self, scramble_moves):
        self.agent.rewards = []
        
        while self.is_training and self.episodes_trained < self.max_episodes:
            # Reset environment with scramble
            self.env.reset()
            self.env.scramble(scramble_moves)
            state = TensorDict(
                {"observation": torch.tensor(self.env._get_state(), dtype=torch.float32).unsqueeze(0)},
                batch_size=[1]
            )
            
            episode_reward = 0
            done = False
            steps = 0
            max_steps = 100
            
            while not done and steps < max_steps:
                # Select action
                action = self.agent.select_action(state["observation"])
                action_item = action.item()
                
                # Take action
                next_state_dict = self.env_wrapper.step(action_item)
                next_state = next_state_dict
                reward = next_state_dict["reward"].item()
                done = next_state_dict["done"].item()
                
                # Store experience
                self.agent.add_experience(state, action_item, next_state, reward, done)
                
                # Update agent
                loss = self.agent.update(batch_size=64)
                
                # Move to the next state
                state = next_state
                episode_reward += reward
                steps += 1
                
                # Small delay to not overwhelm CPU
                time.sleep(0.001)
                
                if not self.is_training:
                    break
            
            # Store episode reward
            self.agent.rewards.append(episode_reward)
            
            # Update episode counter
            self.episodes_trained += 1
            
            # Exit if training is stopped
            if not self.is_training:
                break
        
        if self.episodes_trained >= self.max_episodes:
            self.is_training = False
            self.train_button.config(text="Start Training")
    
    def solve_cube(self):
        if self.is_training:
            messagebox.showwarning("Warning", "Please stop training before solving.")
            return
        
        # Store current state
        current_state = self.current_state
        
        # Solve with RL agent
        done = False
        steps = 0
        max_steps = 50
        solution_found = False
        
        while not done and steps < max_steps:
            # Select best action
            action = self.agent.select_action(current_state["observation"], evaluate=True)
            action_item = action.item()
            
            # Take action
            next_state_dict = self.env_wrapper.step(action_item)
            reward = next_state_dict["reward"].item()
            done = next_state_dict["done"].item()
            
            # Move to the next state
            current_state = next_state_dict
            steps += 1
            
            # Render cube after each step with delay
            self.env.render(self.cube_canvas)
            self.root.update()
            time.sleep(0.5)  # Delay to visualize steps
            
            if done and reward > 0:
                solution_found = True
                break
        
        if solution_found:
            messagebox.showinfo("Success", f"Cube solved in {steps} steps!")
        else:
            messagebox.showinfo("Information", f"Could not solve cube in {max_steps} steps. Try training more.")
    
    def save_model(self):
        try:
            self.agent.save_model("rubiks_cube_model.pth")
            messagebox.showinfo("Success", "Model saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        try:
            self.agent.load_model("rubiks_cube_model.pth")
            messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

# Main entry point
if __name__ == "__main__":
    # Initialize Tkinter
    root = tk.Tk()
    app = RubiksCubeApp(root)
    root.mainloop()
