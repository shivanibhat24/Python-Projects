import traci
import random
import networkx as nx
import time
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import datetime
import argparse
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Constants
MIN_VISIT_INTERVAL = 30  # Minimum time (in minutes) between visits to the same location
PATROL_DURATION = 480    # Default patrol duration in minutes (8 hours)

class SmartCityPatrol:
    def __init__(self, network_file, historical_data=None, num_agents=3, 
                 num_priority_nodes=5, num_hops=3, visualize=True):
        """
        Initialize the Smart City Patrol system.
        
        Args:
            network_file: SUMO network configuration file
            historical_data: Path to historical incident data CSV (optional)
            num_agents: Number of patrol agents/vehicles
            num_priority_nodes: Number of priority nodes to monitor
            num_hops: Number of hops in rabbit walks
            visualize: Whether to enable visualization
        """
        self.network_file = network_file
        self.historical_data = historical_data
        self.num_agents = num_agents
        self.num_priority_nodes = num_priority_nodes
        self.num_hops = num_hops
        self.visualize = visualize
        
        # Data structures
        self.junctions = []
        self.junction_positions = {}
        self.patrol_graph = None
        self.priority_nodes = []
        self.non_priority_nodes = []
        self.rabbit_walks = {}
        self.agent_positions = {}
        self.node_idleness = {}
        self.incident_probabilities = {}
        self.incident_log = []
        self.last_visit_time = {}
        
        # ML model components
        self.time_clusters = None
        self.location_clusters = None

    # ---- Visualization Functions ----
    
    def mark_priority_node(self, junction_id, priority_level=1):
        """
        Mark priority node with color based on priority level.
        Higher priority levels have more intense colors.
        
        Args:
            junction_id: ID of the junction
            priority_level: 1-5 scale of priority (5 being highest)
        """
        if not self.visualize:
            return
            
        # Scale color intensity based on priority level
        color_intensity = min(255, 50 * priority_level)
        
        try:
            traci.gui.toggleSelection(junction_id, objType='junction', layer=0)
            traci.gui.addShape(junction_id + '_p', 0, 
                              shape=[traci.junction.getPosition(junction_id)], 
                              color=(0, 0, color_intensity, 255), 
                              fill=True)
        except:
            print(f"Warning: Could not mark junction {junction_id}")

    def mark_non_priority_node(self, junction_id):
        """Mark non-priority node with an orange circle."""
        if not self.visualize:
            return
            
        try:
            traci.gui.toggleSelection(junction_id, objType='junction', layer=0)
            traci.gui.addShape(junction_id + '_np', 0, 
                              shape=[traci.junction.getPosition(junction_id)], 
                              color=(255, 165, 0, 255), 
                              fill=True)
        except:
            print(f"Warning: Could not mark junction {junction_id}")

    def draw_rabbit_hop(self, agent_id, hop_nodes, color=(0, 255, 0, 255)):
        """Draw the path that the agent follows on the map."""
        if not self.visualize:
            return
            
        try:
            positions = [traci.junction.getPosition(node) for node in hop_nodes if node in self.junction_positions]
            if len(positions) > 1:
                traci.gui.addPolygon(agent_id + '_hops', shape=positions, color=color, fill=False, layer=1)
        except:
            print(f"Warning: Could not draw path for agent {agent_id}")

    # ---- Data Preprocessing ----
    
    def load_historical_data(self):
        """
        Load and preprocess historical incident data for pattern recognition.
        
        Returns:
            DataFrame with processed incident data
        """
        if not self.historical_data:
            print("No historical data provided. Using random probabilities.")
            return None
            
        try:
            # Load data from CSV
            data = pd.read_csv(self.historical_data)
            
            # Ensure required columns exist
            required_cols = ['timestamp', 'location_x', 'location_y', 'incident_type', 'severity']
            if not all(col in data.columns for col in required_cols):
                print("Historical data missing required columns. Using random probabilities.")
                return None
                
            # Convert timestamp to datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            
            # Process location data
            if 'junction_id' not in data.columns:
                # Map coordinates to nearest junctions
                data['junction_id'] = data.apply(
                    lambda row: self.find_nearest_junction(row['location_x'], row['location_y']), 
                    axis=1
                )
            
            print(f"Loaded {len(data)} historical incidents")
            return data
            
        except Exception as e:
            print(f"Error loading historical data: {e}")
            return None
    
    def find_nearest_junction(self, x, y):
        """Find the nearest junction to the given coordinates."""
        min_dist = float('inf')
        nearest = None
        
        for jid, pos in self.junction_positions.items():
            dist = ((pos[0] - x) ** 2 + (pos[1] - y) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest = jid
                
        return nearest

    # ---- Machine Learning Methods ----
    
    def train_incident_prediction_model(self, data):
        """
        Train a model to predict incident probabilities based on historical data.
        
        Args:
            data: DataFrame with historical incident data
        """
        if data is None or len(data) == 0:
            return
            
        # Temporal pattern analysis
        time_features = data[['hour', 'day_of_week']].values
        scaler = StandardScaler()
        time_features_scaled = scaler.fit_transform(time_features)
        
        # Cluster similar times using DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        self.time_clusters = dbscan.fit(time_features_scaled)
        
        # Location pattern analysis
        if 'junction_id' in data.columns:
            # Count incidents by junction
            junction_counts = data['junction_id'].value_counts()
            
            # Calculate incident probabilities by junction
            total_incidents = sum(junction_counts)
            for junction, count in junction_counts.items():
                if junction in self.junctions:
                    self.incident_probabilities[junction] = count / total_incidents
        
        print("Trained incident prediction model")
    
    def calculate_current_probabilities(self):
        """
        Calculate current incident probabilities based on time of day and historical patterns.
        
        Returns:
            Dictionary of junctions and their current probability scores
        """
        # Get current time features
        now = datetime.datetime.now()
        current_hour = now.hour
        current_day = now.weekday()
        
        # Default to historical probabilities
        current_probs = self.incident_probabilities.copy()
        
        # If we have time clusters, adjust probabilities based on time similarity
        if self.time_clusters is not None:
            current_time_feature = np.array([[current_hour, current_day]])
            
            # TODO: Use the time clusters to adjust probabilities
            # This would involve finding which cluster the current time belongs to
            # and adjusting probabilities based on that cluster's incident rates
            
        # For junctions with no historical data, assign a small default probability
        for junction in self.junctions:
            if junction not in current_probs:
                current_probs[junction] = 0.01
                
        return current_probs

    # ---- Patrolling Environment Setup ----
    
    def setup_environment(self):
        """Set up the patrolling environment and graph."""
        # Get junctions from SUMO
        self.junctions, self.junction_positions = self.get_junctions()
        
        # Create patrolling graph
        self.patrol_graph = self.create_patrol_graph(self.junctions)
        
        # Initialize idleness for all nodes
        self.node_idleness = {node: 0 for node in self.junctions}
        self.last_visit_time = {node: 0 for node in self.junctions}
        
        # Load historical data if available
        historical_data = self.load_historical_data()
        
        # Train prediction model
        self.train_incident_prediction_model(historical_data)
        
        # Determine priority nodes based on incident probabilities
        self.determine_priority_nodes()

    def get_junctions(self):
        """Fetch all junctions from the SUMO simulation."""
        junction_ids = traci.junction.getIDList()
        junction_positions = {jid: traci.junction.getPosition(jid) for jid in junction_ids}
        return junction_ids, junction_positions

    def create_patrol_graph(self, junctions):
        """
        Create a graph using SUMO's junctions as nodes with edges representing patrol routes.
        Uses a more sophisticated approach to create edges based on road connectivity.
        """
        G = nx.Graph()
        
        # Add junctions as nodes
        for junction in junctions:
            G.add_node(junction)
        
        # Add edges based on road connectivity
        edge_ids = traci.edge.getIDList()
        for edge_id in edge_ids:
            try:
                from_junction = traci.edge.getFromNode(edge_id)
                to_junction = traci.edge.getToNode(edge_id)
                
                if from_junction in junctions and to_junction in junctions:
                    # Add edge with length as weight
                    length = traci.edge.getLength(edge_id)
                    G.add_edge(from_junction, to_junction, weight=length)
            except:
                # Some edges might not have connected junctions
                continue
        
        # Ensure graph is connected
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            
        return G

    def determine_priority_nodes(self):
        """
        Determine priority nodes based on incident probabilities and current conditions.
        """
        # Get current incident probabilities
        current_probs = self.calculate_current_probabilities()
        
        # Sort junctions by probability
        sorted_junctions = sorted(current_probs.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N junctions as priority nodes
        self.priority_nodes = [j for j, _ in sorted_junctions[:self.num_priority_nodes]]
        
        # Remaining junctions are non-priority
        self.non_priority_nodes = list(set(self.junctions) - set(self.priority_nodes))
        
        # Visualize nodes on map
        for i, node in enumerate(self.priority_nodes):
            # Calculate priority level (1-5) based on position in the list
            priority_level = max(1, min(5, 5 - int(4 * i / len(self.priority_nodes))))
            self.mark_priority_node(node, priority_level)
            
        for node in self.non_priority_nodes:
            self.mark_non_priority_node(node)
            
        # Generate rabbit walks for patrolling
        self.rabbit_walks = self.generate_rabbit_walks()
        
        print(f"Determined {len(self.priority_nodes)} priority nodes")

    # ---- Rabbit Walk Algorithm ----
    
    def generate_rabbit_walks(self):
        """
        Generate Rabbit Walks between priority nodes with adaptive hops.
        Return a dictionary with generated patrol routes.
        """
        walks = {}
        
        for source in self.priority_nodes:
            walks[source] = []
            
            for target in self.priority_nodes:
                if source != target:
                    # Get shortest path between source and target
                    try:
                        shortest_path = nx.shortest_path(self.patrol_graph, source, target)
                        path_length = len(shortest_path)
                        
                        # For very short paths, use them directly
                        if path_length <= 2:
                            walks[source].append(shortest_path)
                            continue
                            
                        # Otherwise create rabbit walk with hops
                        for _ in range(3):  # Generate multiple options
                            walk = [source]
                            
                            # Select intermediate hops
                            for _ in range(self.num_hops):
                                # Choose next hop with preference for high idleness
                                candidates = list(self.patrol_graph.neighbors(walk[-1]))
                                if not candidates:
                                    # If no neighbors, choose a random junction
                                    candidates = self.non_priority_nodes
                                
                                # Weight candidates by idleness
                                weights = [self.node_idleness.get(node, 0) for node in candidates]
                                if sum(weights) == 0:
                                    weights = None  # Use uniform weights if all zeros
                                
                                next_node = random.choices(candidates, weights=weights, k=1)[0]
                                walk.append(next_node)
                            
                            # Ensure walk ends at target
                            if walk[-1] != target:
                                # Find path from last hop to target
                                try:
                                    final_segment = nx.shortest_path(self.patrol_graph, walk[-1], target)
                                    walk.extend(final_segment[1:])  # Skip first node as it's already in the walk
                                except:
                                    walk.append(target)  # Direct jump if no path
                            
                            walks[source].append(walk)
                    
                    except nx.NetworkXNoPath:
                        # If no path exists, create a random walk
                        walk = [source]
                        current = source
                        
                        for _ in range(self.num_hops):
                            neighbors = list(self.patrol_graph.neighbors(current))
                            if neighbors:
                                current = random.choice(neighbors)
                                walk.append(current)
                        
                        walk.append(target)
                        walks[source].append(walk)
        
        return walks

    # ---- Route Assignment and Patrol Logic ----
    
    def assign_route(self, agent_id, current_position):
        """
        Assign a patrol route to an agent based on current conditions.
        
        Args:
            agent_id: The ID of the patrol agent
            current_position: Current position of the agent
            
        Returns:
            Assigned route for the agent
        """
        # Convert edge to junction if needed
        if current_position not in self.junctions:
            # Find nearest junction to current position
            for junction in self.junctions:
                if current_position.startswith(junction):
                    current_position = junction
                    break
        
        if current_position not in self.junctions:
            print(f"Agent {agent_id} at unknown position {current_position}")
            return None
        
        # Update idleness when a node is visited
        self.update_node_idleness(current_position)
        
        # Log the visit
        self.log_visit(agent_id, current_position)
        
        # Decide next destination
        if current_position in self.priority_nodes and current_position in self.rabbit_walks:
            # Select the route with the highest reward
            available_routes = self.rabbit_walks[current_position]
            if not available_routes:
                return None
                
            # Calculate reward for each route
            route_rewards = []
            for route in available_routes:
                # Calculate reward based on idleness of nodes in the route
                idleness_reward = sum(self.node_idleness.get(node, 0) for node in route)
                # Add priority node bonus
                priority_bonus = sum(10 for node in route if node in self.priority_nodes)
                route_rewards.append(idleness_reward + priority_bonus)
            
            # Select route with highest reward
            selected_route = available_routes[route_rewards.index(max(route_rewards))]
            
            # Draw the route on the map
            color = (0, 255, 0, 255)  # Green for normal patrol
            self.draw_rabbit_hop(agent_id, selected_route, color)
            
            return selected_route
        else:
            # If not at a priority node, head to the nearest priority node
            if self.priority_nodes:
                nearest_priority = min(self.priority_nodes, 
                                      key=lambda x: nx.shortest_path_length(self.patrol_graph, 
                                                                          current_position, 
                                                                          x, 
                                                                          weight='weight'))
                try:
                    path = nx.shortest_path(self.patrol_graph, current_position, nearest_priority)
                    color = (255, 0, 0, 255)  # Red for direct path to priority
                    self.draw_rabbit_hop(agent_id, path, color)
                    return path
                except:
                    # If no path, choose a random priority node
                    return [current_position, random.choice(self.priority_nodes)]
            else:
                # If no priority nodes, patrol randomly
                random_node = random.choice(self.junctions)
                return [current_position, random_node]

    def update_node_idleness(self, node):
        """
        Update the idleness value when a node is visited.
        
        Args:
            node: The node that was just visited
        """
        current_time = self.get_current_simulation_time()
        
        # Calculate time since last visit
        time_since_last_visit = current_time - self.last_visit_time.get(node, 0)
        
        # Reset idleness for this node
        self.node_idleness[node] = 0
        
        # Update last visit time
        self.last_visit_time[node] = current_time
        
        # Increase idleness for all other nodes
        for other_node in self.junctions:
            if other_node != node:
                self.node_idleness[other_node] += 1

    def get_current_simulation_time(self):
        """Get the current simulation time in minutes."""
        return traci.simulation.getTime() / 60  # Convert seconds to minutes

    def log_visit(self, agent_id, node):
        """Log a visit for analysis."""
        current_time = self.get_current_simulation_time()
        self.incident_log.append({
            'time': current_time,
            'agent': agent_id,
            'node': node,
            'idleness': self.node_idleness.get(node, 0)
        })

    # ---- Incident Generation and Response ----
    
    def generate_incidents(self):
        """
        Probabilistically generate incidents based on current conditions.
        Returns list of new incidents.
        """
        new_incidents = []
        current_time = self.get_current_simulation_time()
        
        # Get current incident probabilities
        current_probs = self.calculate_current_probabilities()
        
        # Check for new incidents
        for junction, prob in current_probs.items():
            # Scale probability by time since last visit
            time_since_visit = current_time - self.last_visit_time.get(junction, 0)
            adjusted_prob = prob * (0.1 + min(1.0, time_since_visit / MIN_VISIT_INTERVAL))
            
            # Roll for incident
            if random.random() < adjusted_prob / 100:  # Convert to a smaller probability
                severity = random.choices([1, 2, 3, 4, 5], weights=[50, 30, 15, 4, 1])[0]
                new_incidents.append({
                    'time': current_time,
                    'location': junction,
                    'severity': severity,
                    'responded': False
                })
                
                # Make this a high priority node
                if junction not in self.priority_nodes:
                    self.priority_nodes.append(junction)
                    if junction in self.non_priority_nodes:
                        self.non_priority_nodes.remove(junction)
                    
                    # Mark as high priority
                    self.mark_priority_node(junction, severity)
                    
                    # Update rabbit walks
                    self.rabbit_walks = self.generate_rabbit_walks()
                
                print(f"New incident at {junction} with severity {severity}")
        
        return new_incidents

    def respond_to_incidents(self, incidents, agent_positions):
        """
        Check if agents have responded to incidents.
        
        Args:
            incidents: List of active incidents
            agent_positions: Dictionary of agent positions
            
        Returns:
            Updated list of incidents
        """
        responded = []
        
        for incident in incidents:
            location = incident['location']
            
            # Check if any agent is at the incident location
            for agent, position in agent_positions.items():
                if position == location and not incident['responded']:
                    incident['responded'] = True
                    incident['response_time'] = self.get_current_simulation_time() - incident['time']
                    incident['responding_agent'] = agent
                    responded.append(incident)
                    
                    # Log the response
                    print(f"Agent {agent} responded to incident at {location} after {incident['response_time']:.1f} minutes")
                    
                    # Remove from priority if it was added for this incident
                    if location in self.priority_nodes and location not in self.non_priority_nodes:
                        self.non_priority_nodes.append(location)
                        self.priority_nodes.remove(location)
                        self.mark_non_priority_node(location)
                    
                    break
        
        # Remove responded incidents
        active_incidents = [i for i in incidents if not i['responded']]
        
        return active_incidents, responded

    # ---- Main Simulation Logic ----
    
    def setup_agents(self):
        """Set up patrolling agents in the simulation."""
        # Create patrol vehicles
        for i in range(self.num_agents):
            agent_id = f"patrol_{i}"
            
            # Choose random starting position
            start_edge = random.choice(traci.edge.getIDList())
            
            try:
                # Add vehicle to simulation
                traci.vehicle.add(agent_id, routeID="", typeID="police")
                traci.vehicle.moveToXY(agent_id, edgeID=start_edge, lane=0, x=0, y=0, angle=0, keepRoute=2)
                
                # Set vehicle properties
                traci.vehicle.setColor(agent_id, (0, 0, 255, 255))  # Blue for police
                traci.vehicle.setMaxSpeed(agent_id, 15)  # 15 m/s = ~54 km/h
                
                # Initialize agent position
                self.agent_positions[agent_id] = start_edge
                
                print(f"Added patrol agent {agent_id}")
            except Exception as e:
                print(f"Error adding agent {agent_id}: {e}")

    def run_simulation(self):
        """Run the simulation with intelligent patrolling."""
        try:
            # Start SUMO GUI with the specified config file
            if self.visualize:
                traci.start(["sumo-gui", "-c", self.network_file])
            else:
                traci.start(["sumo", "-c", self.network_file])
            
            # Setup patrolling environment
            self.setup_environment()
            
            # Add patrol agents
            self.setup_agents()
            
            # Tracking variables
            step = 0
            active_incidents = []
            responded_incidents = []
            patrol_stats = {
                'visits_per_node': defaultdict(int),
                'response_times': []
            }
            
            # Run for the specified duration or until simulation ends
            max_steps = PATROL_DURATION * 60  # Convert minutes to seconds
            
            while step < max_steps and traci.simulation.getMinExpectedNumber() > 0:
                # Advance simulation
                traci.simulationStep()
                
                # Update agent positions
                for agent_id in list(self.agent_positions.keys()):
                    if agent_id in traci.vehicle.getIDList():
                        current_edge = traci.vehicle.getRoadID(agent_id)
                        
                        # Update position if changed
                        if current_edge != self.agent_positions.get(agent_id):
                            self.agent_positions[agent_id] = current_edge
                            
                            # Check if agent has reached a node/junction
                            for junction in self.junctions:
                                if current_edge.startswith(junction):
                                    # Reached a junction, assign new route
                                    route = self.assign_route(agent_id, junction)
                                    if route:
                                        # Navigate vehicle along the route
                                        self.navigate_agent(agent_id, route)
                                    
                                    # Update visit stats
                                    patrol_stats['visits_per_node'][junction] += 1
                                    break
                    else:
                        # Agent no longer in simulation
                        self.agent_positions.pop(agent_id, None)
                
                # Generate incidents (periodically)
                if step % 60 == 0:  # Check every minute
                    new_incidents = self.generate_incidents()
                    active_incidents.extend(new_incidents)
                
                # Check for incident responses
                active_incidents, newly_responded = self.respond_to_incidents(
                    active_incidents, self.agent_positions)
                
                # Record response times
                for incident in newly_responded:
                    patrol_stats['response_times'].append(incident['response_time'])
                
                responded_incidents.extend(newly_responded)
                
                # Adjust priority nodes periodically
                if step % 300 == 0:  # Every 5 minutes
                    self.determine_priority_nodes()
                
                step += 1
                time.sleep(0.01)  # Slow down for visualization
            
            # Save simulation results
            self.save_results(patrol_stats, responded_incidents)
            
            # Print summary
            self.print_summary(patrol_stats, responded_incidents)
            
        except Exception as e:
            print(f"Error during simulation: {e}")
        finally:
            traci.close()  # Close SUMO

    def navigate_agent(self, agent_id, route):
        """Navigate an agent along a route."""
        if len(route) < 2:
            return
            
        try:
            # Create a new route
            route_id = f"{agent_id}_route_{int(time.time())}"
            
            # Convert junction nodes to edges for vehicle movement
            edges = []
            for i in range(len(route) - 1):
                from_junction = route[i]
                to_junction = route[i+1]
                
                # Find edges connecting these junctions
                connecting_edges = []
                for edge_id in traci.edge.getIDList():
                    try:
                        if (traci.edge.getFromNode(edge_id) == from_junction and 
                            traci.edge.getToNode(edge_id) == to_junction):
                            connecting_edges.append(edge_id)
                    except:
                        continue
                
                if connecting_edges:
                    edges.append(random.choice(connecting_edges))
                
            # If we found a valid path
            if edges:
                # Add route to SUMO
                traci.route.add(route_id, edges)
                
                # Assign route to vehicle
                traci.vehicle.setRoute(agent_id, edges)
            else:
                # Fallback: just set a new target
                traci.vehicle.changeTarget(agent_id, self.get_random_edge())
        except Exception as e:
            print(f"Error navigating agent {agent_id}: {e}")

    def get_random_edge(self):
        """Get a random edge from the network."""
        edges = traci.edge.getIDList()
        return random.choice(edges)

    # ---- Results and Analysis ----
    
    def save_results(self, patrol_stats, responded_incidents):
        """Save simulation results to file."""
        results = {
            'simulation_config': {
                'network_file': self.network_file,
                'num_agents': self.num_agents,
                'num_priority_nodes': self.num_priority_nodes,
                'num_hops': self.num_hops
            },
            'patrol_stats': {
                'visits_per_node': dict(patrol_stats['visits_per_node']),
                'avg_response_time': np.mean(patrol_stats['response_times']) if patrol_stats['response_times'] else 0,
                'max_response_time': max(patrol_stats['response_times']) if patrol_stats['response_times'] else 0,
                'total_incidents': len(responded_incidents),
                'patrol_duration': self.get_current_simulation_time()
            },
            'incidents': [dict(incident) for incident in responded_incidents]
        }
        
        # Save to JSON file
        filename = f"patrol_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")

    def print_summary(self, patrol_stats, responded_incidents):
        """Print summary of patrol results."""
        print("\n===== PATROL SUMMARY =====")
        print(f"Total duration: {self.get_current_simulation_time():.1f} minutes")
        print(f"Number of patrol agents: {self.num_agents}")
        print(f"Total incidents: {len(responded_incidents)}")
        
        if patrol_stats['response_times']:
            print(f"Avg response time: {np.mean(patrol_stats['response_times']):.1f} minutes")
            print(f"Max response time: {max(patrol_stats['response_times']):.1f} minutes")
        
        # Most visited nodes
        most_visited = sorted(patrol_stats['visits_per_node'].items(), 
                             key=lambda x: x[1], reverse=True)[:5]
        print("\nMost visited locations:")
        for node, count in most_visited:
            print(f"  - {node}: {count} visits")
        
        # Least visited nodes
        least_visited = sorted(patrol_stats['visits_per_node'].items(), 
                              key=lambda x: x[1])[:5]
        print("\nLeast visited locations:")
        for node, count in least_visited:
            print(f"  - {node}: {count} visits")
        
        print("=========================\n")

# ---- Command Line Interface ----

def main():
    """Main function to run the program from command line."""
    parser = argparse.ArgumentParser(description="Smart City Patrol Simulation")
    
    parser.add_argument("--network", "-n", type=str, required=True,
                      help="SUMO network file (.net.xml)")
    parser.add_argument("--historical-data", "-d", type=str, default=None,
                      help="Historical incident data (CSV file)")
    parser.add_argument("--agents", "-a", type=int, default=3,
                      help="Number of patrol agents")
    parser.add_argument("--priority-nodes", "-p", type=int, default=5,
                      help="Number of priority nodes")
    parser.add_argument("--hops", type=int, default=3,
                      help="Number of hops in rabbit walks")
    parser.add_argument("--duration", type=int, default=PATROL_DURATION,
                      help="Patrol duration in minutes")
    parser.add_argument("--no-gui", action="store_true",
                      help="Run without visualization")
    
    args = parser.parse_args()
    
    # Update global patrol duration if specified
    global PATROL_DURATION
    PATROL_DURATION = args.duration
    
    # Create and run patrol simulation
    patrol = SmartCityPatrol(
        network_file=args.network,
        historical_data=args.historical_data,
        num_agents=args.agents,
        num_priority_nodes=args.priority_nodes,
        num_hops=args.hops,
        visualize=not args.no_gui
    )
    
    patrol.run_simulation()

# ---- Visualization and Analysis Utilities ----

def plot_patrol_heatmap(results_file):
    """
    Generate a heatmap visualization of patrol coverage from results.
    
    Args:
        results_file: Path to the patrol results JSON file
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract visit data
    visits = results['patrol_stats']['visits_per_node']
    
    # Convert to dataframe for plotting
    nodes = list(visits.keys())
    visit_counts = list(visits.values())
    
    # Sort by visit count
    sorted_indices = np.argsort(visit_counts)
    sorted_nodes = [nodes[i] for i in sorted_indices]
    sorted_visits = [visit_counts[i] for i in sorted_indices]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    plt.barh(sorted_nodes[-30:], sorted_visits[-30:], color='blue')
    plt.xlabel('Number of Visits')
    plt.ylabel('Location ID')
    plt.title('Patrol Coverage Heatmap (Top 30 Locations)')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('patrol_heatmap.png')
    plt.close()
    
    print("Patrol heatmap saved to patrol_heatmap.png")

def analyze_response_times(results_file):
    """
    Analyze and visualize incident response times.
    
    Args:
        results_file: Path to the patrol results JSON file
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract incidents and their response times
    incidents = results['incidents']
    
    if not incidents:
        print("No incidents to analyze")
        return
    
    # Extract response times and severities
    response_times = [inc['response_time'] for inc in incidents]
    severities = [inc['severity'] for inc in incidents]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot response times by severity
    plt.scatter(severities, response_times, alpha=0.7, s=100)
    plt.xlabel('Incident Severity')
    plt.ylabel('Response Time (minutes)')
    plt.title('Incident Response Times by Severity')
    
    # Add trend line
    z = np.polyfit(severities, response_times, 1)
    p = np.poly1d(z)
    plt.plot(range(1, 6), p(range(1, 6)), "r--", alpha=0.8)
    
    # Add statistics
    plt.figtext(0.15, 0.85, f"Average response time: {np.mean(response_times):.2f} minutes")
    plt.figtext(0.15, 0.82, f"Median response time: {np.median(response_times):.2f} minutes")
    plt.figtext(0.15, 0.79, f"Max response time: {max(response_times):.2f} minutes")
    plt.figtext(0.15, 0.76, f"Total incidents: {len(incidents)}")
    
    # Save figure
    plt.tight_layout()
    plt.savefig('response_times.png')
    plt.close()
    
    print("Response time analysis saved to response_times.png")

def generate_patrol_report(results_file):
    """
    Generate a comprehensive patrol report with insights and recommendations.
    
    Args:
        results_file: Path to the patrol results JSON file
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract key statistics
    config = results['simulation_config']
    stats = results['patrol_stats']
    incidents = results['incidents']
    
    # Calculate additional metrics
    coverage_percentage = len(stats['visits_per_node']) / 100  # Assuming total possible nodes
    avg_visits_per_node = np.mean(list(stats['visits_per_node'].values()))
    
    # Identify hotspots (nodes with high visit counts)
    visits_items = list(stats['visits_per_node'].items())
    sorted_visits = sorted(visits_items, key=lambda x: x[1], reverse=True)
    hotspots = sorted_visits[:5]
    
    # Identify coldspots (nodes with low visit counts)
    coldspots = sorted_visits[-5:]
    
    # Group incidents by severity
    severity_counts = {}
    for incident in incidents:
        severity = incident['severity']
        if severity not in severity_counts:
            severity_counts[severity] = 0
        severity_counts[severity] += 1
    
    # Create report file
    report_filename = f"patrol_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_filename, 'w') as f:
        f.write("=== SMART CITY PATROL REPORT ===\n\n")
        
        # Configuration
        f.write("CONFIGURATION:\n")
        f.write(f"- Network: {config['network_file']}\n")
        f.write(f"- Number of agents: {config['num_agents']}\n")
        f.write(f"- Priority nodes: {config['num_priority_nodes']}\n")
        f.write(f"- Rabbit walk hops: {config['num_hops']}\n\n")
        
        # Overall Statistics
        f.write("OVERALL STATISTICS:\n")
        f.write(f"- Patrol duration: {stats['patrol_duration']:.1f} minutes\n")
        f.write(f"- Total incidents: {stats['total_incidents']}\n")
        f.write(f"- Average response time: {stats['avg_response_time']:.1f} minutes\n")
        f.write(f"- Maximum response time: {stats['max_response_time']:.1f} minutes\n")
        f.write(f"- Coverage: {coverage_percentage:.1%} of the network\n")
        f.write(f"- Average visits per node: {avg_visits_per_node:.1f}\n\n")
        
        # Hotspots and Coldspots
        f.write("PATROL HOTSPOTS (most visited):\n")
        for node, visits in hotspots:
            f.write(f"- {node}: {visits} visits\n")
        f.write("\n")
        
        f.write("PATROL COLDSPOTS (least visited):\n")
        for node, visits in coldspots:
            f.write(f"- {node}: {visits} visits\n")
        f.write("\n")
        
        # Incident Analysis
        f.write("INCIDENT ANALYSIS:\n")
        for severity, count in sorted(severity_counts.items()):
            f.write(f"- Severity {severity}: {count} incidents\n")
        f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        if stats['avg_response_time'] > 10:
            f.write("- Consider increasing the number of patrol agents to improve response times\n")
        if len(coldspots) > 0 and coldspots[0][1] == 0:
            f.write("- Adjust patrol strategies to cover coldspots with zero visits\n")
        if config['num_priority_nodes'] < 10:
            f.write("- Consider increasing the number of priority nodes for better coverage\n")
        f.write("- Review historical incident data to optimize patrol routes\n")
        f.write("- Consider time-based patrol strategies for different times of day\n")
    
    print(f"Patrol report generated: {report_filename}")

# ---- Real-time Monitoring Interface ----

def create_monitoring_interface(patrol):
    """
    Create a simple matplotlib-based real-time monitoring interface for the patrol.
    
    Args:
        patrol: SmartCityPatrol instance being monitored
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Function to update the plots
    def update_plots(frame):
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        
        # Plot 1: Node idleness heatmap
        nodes = list(patrol.node_idleness.keys())[:40]  # Limit to avoid overcrowding
        idleness = [patrol.node_idleness[node] for node in nodes]
        
        # Sort by idleness
        sorted_indices = np.argsort(idleness)
        top_nodes = [nodes[i] for i in sorted_indices[-20:]]  # Top 20 most idle
        top_idleness = [idleness[i] for i in sorted_indices[-20:]]
        
        ax1.barh(top_nodes, top_idleness, color='orange')
        ax1.set_title('Top 20 Most Idle Locations')
        ax1.set_xlabel('Idleness Value')
        
        # Plot 2: Agent positions and incident locations
        agent_x = []
        agent_y = []
        
        # Get agent positions
        for agent, pos in patrol.agent_positions.items():
            if pos in patrol.junction_positions:
                x, y = patrol.junction_positions[pos]
                agent_x.append(x)
                agent_y.append(y)
        
        # Get priority node positions
        priority_x = []
        priority_y = []
        
        for node in patrol.priority_nodes:
            if node in patrol.junction_positions:
                x, y = patrol.junction_positions[node]
                priority_x.append(x)
                priority_y.append(y)
        
        # Plot agents and priority nodes
        ax2.scatter(agent_x, agent_y, c='blue', s=100, label='Agents')
        ax2.scatter(priority_x, priority_y, c='red', s=50, label='Priority Nodes')
        
        # Add some junction positions for reference
        all_x = []
        all_y = []
        
        for pos in patrol.junction_positions.values():
            all_x.append(pos[0])
            all_y.append(pos[1])
        
        ax2.scatter(all_x, all_y, c='gray', s=10, alpha=0.3)
        ax2.set_title('Patrol Map')
        ax2.legend()
        
        # Make sure plots don't change scale
        ax2.set_xlim(min(all_x), max(all_x))
        ax2.set_ylim(min(all_y), max(all_y))
    
    # Create animation
    ani = FuncAnimation(fig, update_plots, interval=1000)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
