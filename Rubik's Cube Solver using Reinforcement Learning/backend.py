from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import json
import os
import cv2
from cube_model import CubeEnvironment, DQNAgent, train_model, solve_cube
import tensorflow as tf
import threading
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__, static_folder='static')

# Global variables for maintaining state
training_thread = None
is_training = False
current_progress = {
    'episodes_trained': 0,
    'total_episodes': 0,
    'epsilon': 1.0,
    'avg_reward': 0.0,
    'avg_loss': 0.0,
    'rewards': [],
    'losses': []
}

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Function to run training in a separate thread
def train_model_thread(episodes, learning_rate, gamma, scramble_moves):
    global is_training, current_progress
    
    # Create environment and agent
    env = CubeEnvironment()
    state_size = 324  # 54 stickers * 6 colors (one-hot)
    action_size = env.actions
    agent = DQNAgent(state_size, action_size)
    
    # Set hyperparameters
    agent.learning_rate = learning_rate
    agent.gamma = gamma
    
    batch_size = 32
    current_progress['total_episodes'] = episodes
    current_progress['episodes_trained'] = 0
    current_progress['rewards'] = []
    current_progress['losses'] = []
    
    for e in range(episodes):
        if not is_training:
            break
            
        # Reset the environment
        state = env.reset()
        
        # Scramble the cube
        env.scramble(scramble_moves)
        state = env._get_observation()
        
        # Reset accumulated reward for this episode
        total_reward = 0
        avg_loss = 0
        steps = 0
        
        while not env.done and steps < 50:
            # Choose an action
            action = agent.act(state)
            
            # Take the action
            next_state, reward, done, _ = env.step(action)
            
            # Remember the experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and rewards
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train the model
            loss = agent.replay(batch_size)
            if loss > 0:
                avg_loss += loss
        
        # Update target model periodically
        if e % 10 == 0:
            agent.update_target_model()
        
        # Calculate average loss
        avg_loss = avg_loss / steps if steps > 0 else 0
        
        # Store metrics
        current_progress['rewards'].append(total_reward)
        current_progress['losses'].append(avg_loss)
        
        # Update progress
        current_progress['episodes_trained'] = e + 1
        current_progress['epsilon'] = agent.epsilon
        
        # Calculate average reward over last 10 episodes
        recent_rewards = current_progress['rewards'][-10:] if current_progress['rewards'] else [0]
        current_progress['avg_reward'] = sum(recent_rewards) / len(recent_rewards)
        
        # Calculate average loss over last 10 episodes
        recent_losses = current_progress['losses'][-10:] if current_progress['losses'] else [0]
        current_progress['avg_loss'] = sum(recent_losses) / len(recent_losses)
        
        # Save model periodically
        if e % 100 == 0:
            agent.save(f"models/cube_model_{e}.h5")
    
    # Save final model
    agent.save("models/cube_model_final.h5")
    
    # Save metrics
    with open("models/training_metrics.json", "w") as f:
        json.dump({
            'rewards': current_progress['rewards'],
            'losses': current_progress['losses'],
            'episodes_trained': current_progress['episodes_trained'],
            'epsilon': current_progress['epsilon']
        }, f)
    
    is_training = False

# Route for serving the static HTML
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# API endpoint to start training
@app.route('/api/train', methods=['POST'])
def start_training():
    global training_thread, is_training, current_progress
    
    data = request.json
    episodes = int(data.get('episodes', 1000))
    learning_rate = float(data.get('learning_rate', 0.001))
    gamma = float(data.get('gamma', 0.99))
    scramble_moves = int(data.get('scramble_moves', 5))
    
    if is_training:
        return jsonify({'status': 'error', 'message': 'Training already in progress'})
    
    is_training = True
    training_thread = threading.Thread(
        target=train_model_thread, 
        args=(episodes, learning_rate, gamma, scramble_moves)
    )
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({'status': 'success', 'message': 'Training started'})

# API endpoint to stop training
@app.route('/api/stop_training', methods=['POST'])
def stop_training():
    global is_training
    
    if not is_training:
        return jsonify({'status': 'error', 'message': 'No training in progress'})
    
    is_training = False
    return jsonify({'status': 'success', 'message': 'Training stopped'})

# API endpoint to get training progress
@app.route('/api/progress', methods=['GET'])
def get_progress():
    global current_progress
    return jsonify({
        'episodes_trained': current_progress['episodes_trained'],
        'total_episodes': current_progress['total_episodes'],
        'epsilon': current_progress['epsilon'],
        'avg_reward': current_progress['avg_reward'],
        'avg_loss': current_progress['avg_loss'],
        'is_training': is_training
    })

# API endpoint to get training metrics for charts
@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    global current_progress
    
    # Calculate how many points to return to avoid overwhelming the browser
    rewards = current_progress['rewards']
    losses = current_progress['losses']
    
    max_points = 100
    step = max(1, len(rewards) // max_points)
    
    return jsonify({
        'rewards': rewards[::step],
        'losses': losses[::step],
        'episodes': list(range(0, len(rewards), step))
    })

# API endpoint to scramble the cube
@app.route('/api/scramble', methods=['POST'])
def scramble_cube():
    data = request.json
    moves = int(data.get('moves', 5))
    
    env = CubeEnvironment()
    env.reset()
    actions = env.scramble(moves)
    
    return jsonify({
        'state': env.get_state_dict(),
        'moves': actions
    })

# API endpoint to reset the cube
@app.route('/api/reset', methods=['POST'])
def reset_cube():
    env = CubeEnvironment()
    env.reset()
    
    return jsonify({
        'state': env.get_state_dict()
    })

# API endpoint to solve the cube
@app.route('/api/solve', methods=['POST'])
def solve_cube_endpoint():
    data = request.json
    cube_state = data.get('state', None)
    
    if not cube_state:
        return jsonify({'status': 'error', 'message': 'No cube state provided'})
    
    # Convert list representation to numpy arrays
    for face in cube_state:
        cube_state[face] = np.array(cube_state[face]).reshape(3, 3).tolist()
    
    # Check if we have a trained model
    model_path = 'models/cube_model_final.h5'
    if not os.path.exists(model_path):
        return jsonify({'status': 'error', 'message': 'No trained model found'})
    
    # Solve the cube
    try:
        result = solve_cube(cube_state, model_path)
        return jsonify({
            'status': 'success',
            'solution': result['solution'],
            'solved': result['solved'],
            'final_state': result['final_state']
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# API endpoint to save the model
@app.route('/api/save_model', methods=['POST'])
def save_model():
    data = request.json
    filename = data.get('filename', 'user_model')
    
    # Check if we have a trained model
    model_path = 'models/cube_model_final.h5'
    if not os.path.exists(model_path):
        return jsonify({'status': 'error', 'message': 'No trained model found'})
    
    # Save metrics
    with open(f"models/{filename}_metrics.json", "w") as f:
        json.dump({
            'episodes_trained': current_progress['episodes_trained'],
            'epsilon': current_progress['epsilon'],
            'avg_reward': current_progress['avg_reward'],
            'avg_loss': current_progress['avg_loss']
        }, f)
    
    return jsonify({
        'status': 'success',
        'message': 'Model saved',
        'filename': filename
    })

# API endpoint to load a model
@app.route('/api/load_model', methods=['POST'])
def load_model():
    global current_progress
    
    data = request.json
    filename = data.get('filename', 'user_model')
    
    # Check if model exists
    model_path = f"models/{filename}.h5"
    metrics_path = f"models/{filename}_metrics.json"
    
    if not os.path.exists(model_path) or not os.path.exists(metrics_path):
        return jsonify({'status': 'error', 'message': 'Model not found'})
    
    # Load metrics
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    # Update current progress
    current_progress['episodes_trained'] = metrics.get('episodes_trained', 0)
    current_progress['epsilon'] = metrics.get('epsilon', 1.0)
    current_progress['avg_reward'] = metrics.get('avg_reward', 0.0)
    current_progress['avg_loss'] = metrics.get('avg_loss', 0.0)
    
    return jsonify({
        'status': 'success',
        'message': 'Model loaded',
        'progress': {
            'episodes_trained': current_progress['episodes_trained'],
            'epsilon': current_progress['epsilon'],
            'avg_reward': current_progress['avg_reward'],
            'avg_loss': current_progress['avg_loss']
        }
    })

# API endpoint to process cube images
@app.route('/api/process_cube_images', methods=['POST'])
def process_cube_images():
    data = request.json
    images = data.get('images', {})
    
    if not images or len(images) != 6:
        return jsonify({'status': 'error', 'message': 'Incomplete cube faces provided'})
    
    cube_state = {}
    
    # Process each face image to detect colors
    for face, image_data in images.items():
        # Remove the data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode the base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Save for debugging
        image.save(f"uploads/{face}.png")
        
        # Convert to OpenCV format
        image_np = np.array(image)
        
        # Process the image to detect cube face
        face_colors = detect_cube_colors(image_np, face)
        cube_state[face] = face_colors
    
    # Return the detected cube state
    return jsonify({
        'status': 'success',
        'cube_state': cube_state
    })

def detect_cube_colors(image, face):
    """
    Simplified function to detect colors on a Rubik's cube face from an image.
    In a real implementation, this would use computer vision techniques.
    """
    # This is a placeholder - in a real app, we would:
    # 1. Use image segmentation to find the 9 stickers
    # 2. Detect the color of each sticker
    # 3. Map colors to the cube state
    
    # For this example, we'll just return a predefined state based on the face
    face_map = {
        'top': [0] * 9,      # White
        'bottom': [1] * 9,   # Yellow
        'front': [2] * 9,    # Red
        'back': [3] * 9,     # Orange
        'left': [4] * 9,     # Blue
        'right': [5] * 9     # Green
    }
    
    # Return the placeholder state
    return face_map.get(face, [0] * 9)

# API endpoint to perform a move on the cube
@app.route('/api/make_move', methods=['POST'])
def make_move():
    data = request.json
    cube_state = data.get('state', None)
    move = data.get('move', None)
    
    if not cube_state or move is None:
        return jsonify({'status': 'error', 'message': 'Missing cube state or move'})
    
    # Initialize environment with the current state
    env = CubeEnvironment()
    env.set_state(cube_state)
    
    # Perform the move
    try:
        new_state, _, done, _ = env.step(move)
        return jsonify({
            'status': 'success',
            'state': env.get_state_dict(),
            'solved': done
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# API endpoint to get available models
@app.route('/api/models', methods=['GET'])
def get_models():
    models = []
    
    # List model files in the models directory
    for filename in os.listdir('models'):
        if filename.endswith('.h5') and not filename.endswith('final.h5'):
            # Get the base name without extension
            base_name = filename[:-3]
            
            # Check if there's a corresponding metrics file
            metrics_path = f"models/{base_name}_metrics.json"
            if os.path.exists(metrics_path):
                # Load the metrics
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                
                models.append({
                    'name': base_name,
                    'episodes_trained': metrics.get('episodes_trained', 0),
                    'avg_reward': metrics.get('avg_reward', 0.0)
                })
            else:
                models.append({
                    'name': base_name,
                    'episodes_trained': 0,
                    'avg_reward': 0.0
                })
    
    return jsonify({'models': models})

# API endpoint to evaluate a model on random scrambles
@app.route('/api/evaluate', methods=['POST'])
def evaluate_model():
    data = request.json
    model_name = data.get('model', 'cube_model_final')
    num_tests = int(data.get('num_tests', 10))
    scramble_moves = int(data.get('scramble_moves', 5))
    
    # Check if model exists
    model_path = f"models/{model_name}.h5"
    if not os.path.exists(model_path):
        return jsonify({'status': 'error', 'message': 'Model not found'})
    
    # Initialize environment and load model
    env = CubeEnvironment()
    state_size = 324
    action_size = env.actions
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    
    # Statistics
    success_count = 0
    avg_steps = 0
    max_steps = 50  # Maximum steps per attempt
    
    # Run evaluations
    for i in range(num_tests):
        env.reset()
        env.scramble(scramble_moves)
        state = env._get_observation()
        
        solved = False
        steps = 0
        
        while steps < max_steps and not solved:
            # Choose best action (no exploration)
            action = agent.act(state, test=True)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            state = next_state
            steps += 1
            
            if done:
                solved = True
                success_count += 1
                avg_steps += steps
        
    # Calculate statistics
    success_rate = success_count / num_tests * 100
    avg_steps = avg_steps / success_count if success_count > 0 else 0
    
    return jsonify({
        'status': 'success',
        'model': model_name,
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'num_tests': num_tests,
        'scramble_moves': scramble_moves
    })

# API endpoint to export a visualization of the model's policy
@app.route('/api/visualize_policy', methods=['POST'])
def visualize_policy():
    data = request.json
    model_name = data.get('model', 'cube_model_final')
    
    # Check if model exists
    model_path = f"models/{model_name}.h5"
    if not os.path.exists(model_path):
        return jsonify({'status': 'error', 'message': 'Model not found'})
    
    # Initialize environment and load model
    env = CubeEnvironment()
    state_size = 324
    action_size = env.actions
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    
    # Generate a visualization of the policy for a few scrambled cubes
    visualizations = []
    
    for i in range(3):  # Generate 3 examples
        env.reset()
        env.scramble(5)  # 5 random moves
        
        # Get the current state
        state = env._get_observation()
        state_dict = env.get_state_dict()
        
        # Get Q-values for all actions
        q_values = agent.model.predict(np.expand_dims(state, axis=0))[0]
        
        # Sort actions by Q-value
        actions = list(range(action_size))
        action_q_pairs = list(zip(actions, q_values))
        sorted_actions = sorted(action_q_pairs, key=lambda x: x[1], reverse=True)
        
        # Get the top 3 actions
        top_actions = sorted_actions[:3]
        
        # Add to visualizations
        visualizations.append({
            'state': state_dict,
            'top_actions': [
                {
                    'action': action,
                    'action_name': env.action_to_string(action),
                    'q_value': float(q_value)
                }
                for action, q_value in top_actions
            ]
        })
    
    return jsonify({
        'status': 'success',
        'visualizations': visualizations
    })

# Main entry point
if __name__ == '__main__':
    app.run(host='192.168.1.100', port=5000, debug=True)