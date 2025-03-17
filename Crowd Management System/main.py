import numpy as np
import cv2
import threading
import time
import datetime
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
import json
import os

warnings.filterwarnings('ignore')

class CrowdManagementSystem:
    """
    AI-Based Crowd Management System for Public Events
    Monitors crowd density, predicts potential bottlenecks, and suggests actions
    to prevent overcrowding incidents.
    """
    
    def __init__(self, venue_map, max_capacity, danger_threshold=0.8, warning_threshold=0.6):
        """
        Initialize the crowd management system.
        
        Args:
            venue_map (dict): Map of the venue with zones and their capacities.
            max_capacity (int): Maximum capacity of the venue.
            danger_threshold (float): Threshold for danger level (0.0-1.0).
            warning_threshold (float): Threshold for warning level (0.0-1.0).
        """
        self.venue_map = venue_map
        self.max_capacity = max_capacity
        self.danger_threshold = danger_threshold
        self.warning_threshold = warning_threshold
        
        # Initialize crowd density data structures
        self.zone_density = {zone: 0 for zone in venue_map}
        self.zone_history = {zone: deque(maxlen=100) for zone in venue_map}
        self.total_crowd = 0
        
        # Initialize model for crowd flow prediction
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_model_trained = False
        
        # Initialize alerts and logs
        self.alerts = []
        self.logs = []
        
        # System status
        self.is_running = False
        self.monitoring_thread = None
        
        # Load historical data if available
        self.historical_data = self._load_historical_data()
        
        # Initialize notification system
        self.notification_subscribers = []
        
        print("Crowd Management System initialized successfully.")
        
    def _load_historical_data(self):
        """Load historical crowd data if available."""
        try:
            if os.path.exists('historical_crowd_data.csv'):
                return pd.read_csv('historical_crowd_data.csv')
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading historical data: {e}")
            return pd.DataFrame()
    
    def start_monitoring(self):
        """Start the crowd monitoring system."""
        if self.is_running:
            print("System is already running.")
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self._log_event("System started monitoring.")
        print("Crowd monitoring started.")
    
    def stop_monitoring(self):
        """Stop the crowd monitoring system."""
        if not self.is_running:
            print("System is not running.")
            return
        
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        self._log_event("System stopped monitoring.")
        print("Crowd monitoring stopped.")
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        while self.is_running:
            # In a real system, this would get data from cameras or sensors
            # For simulation, we'll use random data
            self._update_crowd_density_simulation()
            
            # Analyze crowd density
            self._analyze_crowd_density()
            
            # Predict future crowd flow
            self._predict_crowd_flow()
            
            # Check for alerts
            self._check_alerts()
            
            # Save data for historical analysis
            self._save_current_state()
            
            # Wait for next monitoring cycle
            time.sleep(5)
    
    def _update_crowd_density_simulation(self):
        """
        Simulate crowd density updates from cameras/sensors.
        In a real system, this would be replaced with actual sensor data.
        """
        # Simulate crowd movement between zones
        for zone in self.venue_map:
            # Random fluctuation in crowd density
            fluctuation = np.random.normal(0, 0.05 * self.venue_map[zone])
            self.zone_density[zone] = max(0, min(
                self.zone_density[zone] + fluctuation,
                self.venue_map[zone]
            ))
            self.zone_history[zone].append(self.zone_density[zone])
        
        # Update total crowd count
        self.total_crowd = sum(self.zone_density.values())
    
    def update_crowd_density(self, zone, count):
        """
        Update crowd density for a specific zone with actual data.
        
        Args:
            zone (str): The zone to update.
            count (int): The number of people in the zone.
        """
        if zone not in self.venue_map:
            print(f"Zone {zone} not found in venue map.")
            return
        
        self.zone_density[zone] = count
        self.zone_history[zone].append(count)
        self.total_crowd = sum(self.zone_density.values())
        
        # Analyze after update
        self._analyze_crowd_density()
        self._check_alerts()
    
    def _analyze_crowd_density(self):
        """Analyze current crowd density across all zones."""
        capacity_percentages = {
            zone: (self.zone_density[zone] / self.venue_map[zone]) 
            for zone in self.venue_map
        }
        
        # Identify bottlenecks
        bottlenecks = {
            zone: percentage 
            for zone, percentage in capacity_percentages.items() 
            if percentage > self.warning_threshold
        }
        
        # Calculate overall venue capacity usage
        overall_capacity_usage = self.total_crowd / self.max_capacity
        
        analysis = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_capacity": overall_capacity_usage,
            "zone_capacities": capacity_percentages,
            "bottlenecks": bottlenecks
        }
        
        return analysis
    
    def _predict_crowd_flow(self):
        """Predict future crowd flow based on historical data."""
        if len(self.historical_data) < 50 and not self.is_model_trained:
            # Not enough data to make reliable predictions
            return None
        
        # Prepare data for prediction
        # In a real system, this would include time features, event schedule, etc.
        if not self.is_model_trained and len(self.historical_data) >= 50:
            self._train_prediction_model()
        
        # Make predictions for each zone
        predictions = {}
        for zone in self.venue_map:
            if len(self.zone_history[zone]) < 10:
                continue
                
            # Create features from recent history
            features = list(self.zone_history[zone])[-10:]
            scaled_features = self.scaler.transform([features])
            
            # Predict next 3 time steps
            next_values = []
            current_features = scaled_features.copy()
            for i in range(3):
                next_val = self.model.predict(current_features)[0]
                next_values.append(next_val)
                
                # Update features for next prediction
                current_features = np.roll(current_features, -1)
                current_features[0][-1] = next_val
            
            predictions[zone] = next_values
        
        return predictions
    
    def _train_prediction_model(self):
        """Train the prediction model using historical data."""
        try:
            # Prepare training data
            X = []
            y = []
            
            # Create sequences of 10 time steps to predict the next value
            for zone in self.venue_map:
                zone_data = list(self.zone_history[zone])
                if len(zone_data) < 11:
                    continue
                    
                for i in range(len(zone_data) - 10):
                    X.append(zone_data[i:i+10])
                    y.append(zone_data[i+10])
            
            if len(X) < 10:
                return
                
            X = np.array(X)
            y = np.array(y)
            
            # Scale the data
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Train the model
            self.model.fit(X_scaled, y)
            self.is_model_trained = True
            
            self._log_event("Prediction model trained successfully.")
        except Exception as e:
            self._log_event(f"Error training prediction model: {e}")
    
    def _check_alerts(self):
        """Check for alert conditions and generate alerts if needed."""
        capacity_percentages = {
            zone: (self.zone_density[zone] / self.venue_map[zone]) 
            for zone in self.venue_map
        }
        
        new_alerts = []
        
        # Check individual zones
        for zone, percentage in capacity_percentages.items():
            if percentage >= self.danger_threshold:
                alert = {
                    "level": "DANGER",
                    "zone": zone,
                    "capacity": percentage,
                    "message": f"Zone {zone} is at {percentage:.1%} capacity. Immediate action required!",
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                new_alerts.append(alert)
            elif percentage >= self.warning_threshold:
                alert = {
                    "level": "WARNING",
                    "zone": zone,
                    "capacity": percentage,
                    "message": f"Zone {zone} is at {percentage:.1%} capacity. Monitor closely.",
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                new_alerts.append(alert)
        
        # Check overall venue capacity
        overall_capacity = self.total_crowd / self.max_capacity
        if overall_capacity >= self.danger_threshold:
            alert = {
                "level": "DANGER",
                "zone": "Overall",
                "capacity": overall_capacity,
                "message": f"Venue is at {overall_capacity:.1%} capacity. Consider restricting entry.",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            new_alerts.append(alert)
        elif overall_capacity >= self.warning_threshold:
            alert = {
                "level": "WARNING",
                "zone": "Overall",
                "capacity": overall_capacity,
                "message": f"Venue is at {overall_capacity:.1%} capacity. Prepare for crowd control measures.",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            new_alerts.append(alert)
        
        # Add new alerts to the list
        for alert in new_alerts:
            self.alerts.append(alert)
            self._notify_alert(alert)
        
        return new_alerts
    
    def _save_current_state(self):
        """Save current state for historical analysis."""
        current_state = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "zone_density": self.zone_density.copy(),
            "total_crowd": self.total_crowd
        }
        
        # In a real system, this would be saved to a database
        # For simulation, we'll append to a list
        if not hasattr(self, 'historical_states'):
            self.historical_states = []
        
        self.historical_states.append(current_state)
        
        # Periodically save to CSV
        if len(self.historical_states) % 20 == 0:
            self._save_historical_data()
    
    def _save_historical_data(self):
        """Save historical data to a CSV file."""
        try:
            if not hasattr(self, 'historical_states') or not self.historical_states:
                return
                
            # Convert to DataFrame
            data = []
            for state in self.historical_states:
                row = {"timestamp": state["timestamp"], "total_crowd": state["total_crowd"]}
                for zone, density in state["zone_density"].items():
                    row[f"zone_{zone}"] = density
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv('historical_crowd_data.csv', index=False)
            
            self._log_event("Historical data saved successfully.")
        except Exception as e:
            self._log_event(f"Error saving historical data: {e}")
    
    def _log_event(self, message):
        """Log an event."""
        log_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": message
        }
        self.logs.append(log_entry)
        print(f"[{log_entry['timestamp']}] {message}")
    
    def get_venue_status(self):
        """Get the current status of the venue."""
        analysis = self._analyze_crowd_density()
        status = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_crowd": self.total_crowd,
            "max_capacity": self.max_capacity,
            "capacity_percentage": self.total_crowd / self.max_capacity,
            "zone_status": {
                zone: {
                    "current": self.zone_density[zone],
                    "capacity": self.venue_map[zone],
                    "percentage": self.zone_density[zone] / self.venue_map[zone]
                } for zone in self.venue_map
            },
            "alerts": [alert for alert in self.alerts[-5:]] if self.alerts else []
        }
        return status
    
    def subscribe_to_alerts(self, callback):
        """
        Subscribe to alerts.
        
        Args:
            callback (function): A function that will be called when an alert is generated.
        """
        self.notification_subscribers.append(callback)
        self._log_event(f"New alert subscriber added. Total subscribers: {len(self.notification_subscribers)}")
    
    def _notify_alert(self, alert):
        """Notify all subscribers about a new alert."""
        for subscriber in self.notification_subscribers:
            try:
                subscriber(alert)
            except Exception as e:
                self._log_event(f"Error notifying subscriber: {e}")
    
    def get_recommended_actions(self):
        """Get recommended actions based on current venue status."""
        status = self.get_venue_status()
        recommendations = []
        
        # Check overall capacity
        if status["capacity_percentage"] > self.danger_threshold:
            recommendations.append({
                "priority": "HIGH",
                "action": "Temporarily stop admitting new visitors",
                "reason": "Venue is over danger threshold capacity"
            })
            recommendations.append({
                "priority": "HIGH",
                "action": "Activate emergency crowd management protocols",
                "reason": "Prevent potential overcrowding incidents"
            })
        elif status["capacity_percentage"] > self.warning_threshold:
            recommendations.append({
                "priority": "MEDIUM",
                "action": "Slow down admission rate",
                "reason": "Venue is approaching capacity"
            })
            recommendations.append({
                "priority": "MEDIUM",
                "action": "Prepare staff for crowd control measures",
                "reason": "Pre-emptive preparation for potential crowding"
            })
        
        # Check individual zones
        for zone, data in status["zone_status"].items():
            if data["percentage"] > self.danger_threshold:
                recommendations.append({
                    "priority": "HIGH",
                    "action": f"Redirect visitors away from zone {zone}",
                    "reason": f"Zone {zone} is at {data['percentage']:.1%} capacity"
                })
                recommendations.append({
                    "priority": "HIGH",
                    "action": f"Open alternative pathways around zone {zone}",
                    "reason": "Reduce pressure on overcrowded area"
                })
            elif data["percentage"] > self.warning_threshold:
                recommendations.append({
                    "priority": "MEDIUM",
                    "action": f"Station additional staff at zone {zone}",
                    "reason": f"Zone {zone} is approaching capacity limit"
                })
                recommendations.append({
                    "priority": "MEDIUM",
                    "action": f"Begin soft crowd redistribution from zone {zone}",
                    "reason": "Prevent zone from reaching critical capacity"
                })
        
        return recommendations
    
    def generate_report(self):
        """Generate a comprehensive report of the venue status and history."""
        if not hasattr(self, 'historical_states') or not self.historical_states:
            return {"error": "No historical data available for report generation."}
        
        # Current status
        current_status = self.get_venue_status()
        
        # Historical analysis
        zone_peaks = {}
        busiest_times = {}
        
        for zone in self.venue_map:
            zone_data = [state["zone_density"][zone] for state in self.historical_states]
            max_density = max(zone_data)
            max_index = zone_data.index(max_density)
            zone_peaks[zone] = {
                "max_density": max_density,
                "time": self.historical_states[max_index]["timestamp"],
                "percentage": max_density / self.venue_map[zone]
            }
        
        # Overall peak
        total_data = [state["total_crowd"] for state in self.historical_states]
        max_total = max(total_data)
        max_total_index = total_data.index(max_total)
        overall_peak = {
            "max_crowd": max_total,
            "time": self.historical_states[max_total_index]["timestamp"],
            "percentage": max_total / self.max_capacity
        }
        
        # Alert history
        alert_summary = {
            "total_alerts": len(self.alerts),
            "warnings": len([a for a in self.alerts if a["level"] == "WARNING"]),
            "dangers": len([a for a in self.alerts if a["level"] == "DANGER"]),
            "most_recent": self.alerts[-1] if self.alerts else None
        }
        
        report = {
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "monitoring_duration": (
                datetime.datetime.now() - 
                datetime.datetime.strptime(self.historical_states[0]["timestamp"], "%Y-%m-%d %H:%M:%S")
            ).total_seconds() / 3600,  # hours
            "current_status": current_status,
            "zone_peaks": zone_peaks,
            "overall_peak": overall_peak,
            "alert_summary": alert_summary,
            "recommendations": self.get_recommended_actions()
        }
        
        return report
    
    def visualize_crowd_density(self):
        """
        Generate a visualization of current crowd density.
        Returns a dictionary with plot data that can be rendered.
        """
        zone_labels = list(self.venue_map.keys())
        current_values = [self.zone_density[zone] for zone in zone_labels]
        capacity_values = [self.venue_map[zone] for zone in zone_labels]
        
        percentages = [current / capacity * 100 for current, capacity in zip(current_values, capacity_values)]
        
        # Prepare color coding
        colors = []
        for percentage in percentages:
            if percentage >= self.danger_threshold * 100:
                colors.append('red')
            elif percentage >= self.warning_threshold * 100:
                colors.append('orange')
            else:
                colors.append('green')
        
        # Create plot data
        plot_data = {
            "labels": zone_labels,
            "current_values": current_values,
            "capacity_values": capacity_values,
            "percentages": percentages,
            "colors": colors,
            "title": "Current Crowd Density by Zone",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return plot_data
    
    def visualize_historical_trend(self):
        """
        Generate historical trend visualization data.
        Returns a dictionary with plot data that can be rendered.
        """
        if not hasattr(self, 'historical_states') or len(self.historical_states) < 2:
            return {"error": "Not enough historical data for trend visualization."}
        
        timestamps = [state["timestamp"] for state in self.historical_states]
        total_crowd = [state["total_crowd"] for state in self.historical_states]
        
        # Zone-specific data
        zone_data = {}
        for zone in self.venue_map:
            zone_data[zone] = [state["zone_density"][zone] for state in self.historical_states]
        
        # Create plot data
        plot_data = {
            "timestamps": timestamps,
            "total_crowd": total_crowd,
            "zone_data": zone_data,
            "max_capacity": self.max_capacity,
            "danger_threshold": self.max_capacity * self.danger_threshold,
            "warning_threshold": self.max_capacity * self.warning_threshold,
            "title": "Historical Crowd Density Trends",
        }
        
        return plot_data


class VenueVisualization:
    """Class for visualizing venue and crowd data."""
    
    def __init__(self, crowd_system):
        """
        Initialize the visualization system.
        
        Args:
            crowd_system (CrowdManagementSystem): The crowd management system to visualize.
        """
        self.crowd_system = crowd_system
        
    def render_venue_map(self, output_file=None):
        """
        Render the venue map with current crowd density.
        
        Args:
            output_file (str, optional): If provided, save the visualization to this file.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get crowd density data
        data = self.crowd_system.visualize_crowd_density()
        
        # Simplified venue visualization (in reality, this would be a proper floor plan)
        zones = data["labels"]
        values = data["percentages"]
        colors = data["colors"]
        
        # Create a simple grid layout
        n_zones = len(zones)
        cols = int(np.ceil(np.sqrt(n_zones)))
        rows = int(np.ceil(n_zones / cols))
        
        for i, (zone, value, color) in enumerate(zip(zones, values, colors)):
            row = i // cols
            col = i % cols
            
            # Create a rectangle for each zone
            rect = plt.Rectangle(
                (col, row), 0.9, 0.9, 
                color=color, alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add zone label and capacity info
            ax.text(
                col + 0.45, row + 0.5, 
                f"{zone}\n{value:.1f}%",
                ha="center", va="center", fontsize=12, fontweight="bold"
            )
        
        # Set plot limits and properties
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title and legend
        plt.title("Venue Crowd Density Map", fontsize=16)
        
        # Create legend elements
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', edgecolor='black', label='Normal'),
            Patch(facecolor='orange', edgecolor='black', label='Warning'),
            Patch(facecolor='red', edgecolor='black', label='Danger')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add timestamp
        plt.figtext(0.02, 0.02, f"Generated: {data['timestamp']}", fontsize=8)
        
        if output_file:
            plt.savefig(output_file)
            return output_file
        else:
            return fig
    
    def render_historical_trend(self, output_file=None):
        """
        Render historical crowd trends.
        
        Args:
            output_file (str, optional): If provided, save the visualization to this file.
        """
        data = self.crowd_system.visualize_historical_trend()
        if "error" in data:
            return {"error": data["error"]}
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot total crowd
        ax.plot(
            range(len(data["timestamps"])), 
            data["total_crowd"], 
            'k-', 
            label='Total Crowd', 
            linewidth=2
        )
        
        # Plot threshold lines
        ax.axhline(
            y=data["warning_threshold"], 
            color='orange', 
            linestyle='--', 
            label=f'Warning Threshold ({self.crowd_system.warning_threshold*100}%)'
        )
        ax.axhline(
            y=data["danger_threshold"], 
            color='red', 
            linestyle='--', 
            label=f'Danger Threshold ({self.crowd_system.danger_threshold*100}%)'
        )
        
        # Axis labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of People')
        ax.set_title(data["title"], fontsize=16)
        
        # Set x-axis ticks
        n_ticks = min(10, len(data["timestamps"]))
        tick_indices = np.linspace(0, len(data["timestamps"])-1, n_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([data["timestamps"][i] for i in tick_indices], rotation=45)
        
        # Add legend
        ax.legend()
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            return output_file
        else:
            return fig


class EventManagementInterface:
    """Interface for event managers to interact with the crowd management system."""
    
    def __init__(self, crowd_system):
        """
        Initialize the interface.
        
        Args:
            crowd_system (CrowdManagementSystem): The crowd management system to interface with.
        """
        self.crowd_system = crowd_system
        self.visualization = VenueVisualization(crowd_system)
        
        # Register for alerts
        self.crowd_system.subscribe_to_alerts(self._handle_alert)
        
        # Alert history for UI
        self.ui_alerts = []
    
    def _handle_alert(self, alert):
        """Handle incoming alerts."""
        print(f"ALERT: {alert['level']} - {alert['message']}")
        self.ui_alerts.append(alert)
    
    def get_dashboard_data(self):
        """Get data for the management dashboard."""
        status = self.crowd_system.get_venue_status()
        recommendations = self.crowd_system.get_recommended_actions()
        
        # Get visualization data
        crowd_density_vis = self.crowd_system.visualize_crowd_density()
        
        dashboard_data = {
            "venue_status": status,
            "recommendations": recommendations,
            "alerts": self.ui_alerts[-5:],  # Most recent 5 alerts
            "visualizations": {
                "crowd_density": crowd_density_vis
            }
        }
        
        return dashboard_data
    
    def manual_update_zone(self, zone, count):
        """
        Manually update the count for a specific zone.
        
        Args:
            zone (str): The zone to update.
            count (int): The new count for the zone.
        """
        self.crowd_system.update_crowd_density(zone, count)
        return {"status": "success", "message": f"Updated zone {zone} to {count} people."}
    
    def generate_summary_report(self):
        """Generate a summary report of the event."""
        report = self.crowd_system.generate_report()
        return report
    
    def export_data(self, format="json"):
        """
        Export crowd management data.
        
        Args:
            format (str): The format to export data in ("json" or "csv").
        """
        if not hasattr(self.crowd_system, 'historical_states'):
            return {"error": "No data to export."}
        
        if format.lower() == "json":
            export_data = {
                "venue_map": self.crowd_system.venue_map,
                "max_capacity": self.crowd_system.max_capacity,
                "thresholds": {
                    "warning": self.crowd_system.warning_threshold,
                    "danger": self.crowd_system.danger_threshold
                },
                "historical_data": self.crowd_system.historical_states,
                "alerts": self.crowd_system.alerts
            }
            
            with open("crowd_management_export.json", "w") as f:
                json.dump(export_data, f, indent=2)
            
            return {"status": "success", "file": "crowd_management_export.json"}
        
        elif format.lower() == "csv":
            # Convert to DataFrame and export
            data = []
            for state in self.crowd_system.historical_states:
                row = {"timestamp": state["timestamp"], "total_crowd": state["total_crowd"]}
                for zone, density in state["zone_density"].items():
                    row[f"zone_{zone}"] = density
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv("crowd_management_export.csv", index=False)
            
            return {"status": "success", "file": "crowd_management_export.csv"}
        
        else:
            return {"error": f"Unsupported export format: {format}"}


# Example usage
if __name__ == "__main__":
    # Define venue map - zones and their capacities
    venue_map = {
        "Entrance": 200,
        "Main Hall": 800,
        "Stage Front": 500,
        "Food Court": 300,
        "East Wing": 250,
        "West Wing": 250,
        "VIP Area": 100,
        "Restrooms": 150
    }
    
    # Initialize the system
    crowd_system = CrowdManagementSystem(
        venue_map=venue_map,
        max_capacity=2000,
        danger_threshold=0.85,
        warning_threshold=0.7
    )
    
    # Start monitoring
    crowd_system.start_monitoring()
    
    # Create management interface
    management = EventManagementInterface(crowd_system)
    
    # Simulate some crowd updates
    import random
    for _ in range(10):
        for zone in venue_map:
           # Random crowd density between 40% and 90% of capacity
            capacity = venue_map[zone]
            count = int(random.uniform(0.4, 0.9) * capacity)
            crowd_system.update_crowd_density(zone, count)
        time.sleep(2)
    
    # Get current status
    print("\nCurrent Venue Status:")
    status = crowd_system.get_venue_status()
    print(json.dumps(status, indent=2))
    
    # Get recommendations
    print("\nRecommended Actions:")
    recommendations = crowd_system.get_recommended_actions()
    for rec in recommendations:
        print(f"[{rec['priority']}] {rec['action']} - {rec['reason']}")
    
    # Generate visualizations
    vis = VenueVisualization(crowd_system)
    map_fig = vis.render_venue_map("venue_map.png")
    trend_fig = vis.render_historical_trend("crowd_trends.png")
    
    # Generate report
    print("\nGenerating Event Report...")
    report = crowd_system.generate_report()
    print(f"Report generated at {report['generated_at']}")
    print(f"Monitoring duration: {report['monitoring_duration']:.2f} hours")
    print(f"Peak crowd: {report['overall_peak']['max_crowd']} people ({report['overall_peak']['percentage']:.1%} of capacity)")
    
    # Export data
    export_result = management.export_data(format="json")
    print(f"\nData exported to {export_result.get('file', 'unknown')}")
    
    # Stop monitoring
    crowd_system.stop_monitoring()
    print("\nCrowd monitoring stopped.")


class CrowdDetector:
    """
    Class to handle crowd detection from camera feeds.
    In a real system, this would use computer vision models for crowd detection.
    """
    
    def __init__(self, camera_config):
        """
        Initialize the crowd detector.
        
        Args:
            camera_config (dict): Configuration for cameras in the venue.
        """
        self.camera_config = camera_config
        self.cameras = {}
        
        # In a real system, this would load ML models for crowd detection
        print("Initializing crowd detection system...")
        
    def initialize_cameras(self):
        """Initialize camera connections."""
        for camera_id, config in self.camera_config.items():
            try:
                # In a real system, this would connect to actual cameras
                self.cameras[camera_id] = {
                    "id": camera_id,
                    "location": config["location"],
                    "zone": config["zone"],
                    "status": "connected",
                    "last_frame": None
                }
                print(f"Camera {camera_id} connected successfully.")
            except Exception as e:
                print(f"Failed to connect to camera {camera_id}: {e}")
    
    def process_camera_feed(self, camera_id):
        """
        Process a camera feed to detect crowd density.
        
        Args:
            camera_id (str): The ID of the camera to process.
            
        Returns:
            dict: Results of crowd detection.
        """
        if camera_id not in self.cameras:
            return {"error": f"Camera {camera_id} not found."}
        
        # In a real system, this would grab a frame from the camera and process it
        # For simulation, we'll generate random data
        
        camera = self.cameras[camera_id]
        zone = camera["zone"]
        zone_capacity = 0  # This would come from the venue map
        
        # Simulated crowd detection
        detected_count = int(random.uniform(0.2, 0.8) * zone_capacity)
        
        # In a real system, we'd also detect things like movement patterns, queue formation, etc.
        results = {
            "camera_id": camera_id,
            "zone": zone,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detected_count": detected_count,
            "confidence": random.uniform(0.8, 0.95),
            "density_map": None  # In a real system, this would be a heatmap
        }
        
        return results


class CrowdPredictionModel:
    """Class for more advanced crowd prediction models."""
    
    def __init__(self, historical_data=None):
        """
        Initialize the prediction model.
        
        Args:
            historical_data (pd.DataFrame, optional): Historical crowd data.
        """
        self.historical_data = historical_data
        self.model = None
        
    def train_model(self, features, target):
        """
        Train the prediction model.
        
        Args:
            features (array-like): Training features.
            target (array-like): Target values.
        """
        # In a real system, this could use more sophisticated models
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(features, target)
        self.model = model
        
        return model
    
    def predict_crowd_flow(self, current_state, event_schedule, weather_data):
        """
        Predict future crowd flow based on current state and external factors.
        
        Args:
            current_state (dict): Current crowd state.
            event_schedule (dict): Schedule of events that might affect crowd flow.
            weather_data (dict): Weather data that might affect attendance.
            
        Returns:
            dict: Predictions for future crowd flow.
        """
        if self.model is None:
            return {"error": "Model not trained."}
        
        # In a real system, this would preprocess the inputs and make predictions
        # For simulation, we'll return a simplified prediction
        
        predictions = {}
        
        # Example prediction logic
        for zone in current_state["zone_density"]:
            current_density = current_state["zone_density"][zone]
            zone_capacity = 0  # This would come from venue map
            
            # Predict next hour based on simplistic factors
            next_hour_prediction = current_density
            
            # Factor in event schedule
            upcoming_events = [e for e in event_schedule if e["zone"] == zone]
            if upcoming_events:
                # Events tend to increase crowd density
                next_hour_prediction *= 1.2
            
            # Factor in weather
            if weather_data.get("rain_probability", 0) > 0.5:
                # Rain tends to push people indoors
                if zone.endswith("indoor"):
                    next_hour_prediction *= 1.15
                else:
                    next_hour_prediction *= 0.8
            
            # Ensure prediction doesn't exceed capacity
            next_hour_prediction = min(next_hour_prediction, zone_capacity)
            
            predictions[zone] = next_hour_prediction
        
        return predictions


class EmergencyResponseSystem:
    """System for managing emergency situations related to crowd control."""
    
    def __init__(self, crowd_system):
        """
        Initialize the emergency response system.
        
        Args:
            crowd_system (CrowdManagementSystem): The crowd management system to interface with.
        """
        self.crowd_system = crowd_system
        self.emergency_status = "normal"
        self.emergency_protocols = {
            "overcrowding": {
                "actions": [
                    "Close venue entrances",
                    "Open all emergency exits",
                    "Direct security to key choke points",
                    "Make PA announcement requesting calm and cooperation",
                    "Notify emergency services if situation escalates"
                ],
                "contacts": ["Security Chief", "Event Manager", "Medical Team"]
            },
            "medical": {
                "actions": [
                    "Clear path for medical personnel",
                    "Establish medical treatment area",
                    "Make PA announcement for medical professionals",
                    "Contact emergency services"
                ],
                "contacts": ["Medical Team", "Security Chief", "Emergency Services"]
            },
            "security": {
                "actions": [
                    "Isolate affected area",
                    "Initiate security protocol",
                    "Make PA announcement if needed",
                    "Contact law enforcement if needed"
                ],
                "contacts": ["Security Chief", "Event Manager", "Law Enforcement"]
            }
        }
        
        # Register for alerts to trigger emergency responses
        self.crowd_system.subscribe_to_alerts(self._monitor_alerts)
    
    def _monitor_alerts(self, alert):
        """Monitor incoming alerts for potential emergency situations."""
        if alert["level"] == "DANGER" and alert["capacity"] > 0.95:
            self.trigger_emergency_protocol("overcrowding")
    
    def trigger_emergency_protocol(self, emergency_type):
        """
        Trigger an emergency protocol.
        
        Args:
            emergency_type (str): The type of emergency.
            
        Returns:
            dict: Response with emergency protocol details.
        """
        if emergency_type not in self.emergency_protocols:
            return {"error": f"Unknown emergency type: {emergency_type}"}
        
        # Set emergency status
        self.emergency_status = "active"
        
        # Get protocol
        protocol = self.emergency_protocols[emergency_type]
        
        # Log the emergency
        self.crowd_system._log_event(f"EMERGENCY: {emergency_type} protocol activated")
        
        # In a real system, this would trigger actual emergency notifications
        response = {
            "emergency_type": emergency_type,
            "status": "activated",
            "protocol": protocol,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return response
    
    def clear_emergency(self):
        """Clear the emergency status."""
        prev_status = self.emergency_status
        self.emergency_status = "normal"
        
        self.crowd_system._log_event(f"Emergency cleared. Previous status: {prev_status}")
        
        return {
            "status": "cleared",
            "previous_status": prev_status,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def get_emergency_status(self):
        """Get the current emergency status."""
        return {
            "status": self.emergency_status,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


class CrowdManagementAPI:
    """API for external systems to interact with the crowd management system."""
    
    def __init__(self, crowd_system):
        """
        Initialize the API.
        
        Args:
            crowd_system (CrowdManagementSystem): The crowd management system to interface with.
        """
        self.crowd_system = crowd_system
        self.management = EventManagementInterface(crowd_system)
        self.emergency = EmergencyResponseSystem(crowd_system)
        
    def get_system_status(self):
        """Get the overall system status."""
        venue_status = self.crowd_system.get_venue_status()
        emergency_status = self.emergency.get_emergency_status()
        
        return {
            "venue_status": venue_status,
            "emergency_status": emergency_status,
            "system_running": self.crowd_system.is_running,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def get_crowd_density(self, zone=None):
        """
        Get crowd density information.
        
        Args:
            zone (str, optional): If provided, get data for specific zone. Otherwise, get all zones.
            
        Returns:
            dict: Crowd density information.
        """
        status = self.crowd_system.get_venue_status()
        
        if zone:
            if zone not in status["zone_status"]:
                return {"error": f"Zone {zone} not found."}
            return {"zone": zone, "status": status["zone_status"][zone]}
        
        return {"zones": status["zone_status"]}
    
    def get_recommendations(self):
        """Get current recommendations."""
        return self.crowd_system.get_recommended_actions()
    
    def get_alerts(self, count=5):
        """
        Get recent alerts.
        
        Args:
            count (int): Number of recent alerts to return.
            
        Returns:
            list: Recent alerts.
        """
        return self.crowd_system.alerts[-count:] if self.crowd_system.alerts else []
    
    def update_zone_capacity(self, zone, capacity):
        """
        Update the capacity of a zone.
        
        Args:
            zone (str): The zone to update.
            capacity (int): The new capacity.
            
        Returns:
            dict: Response with update status.
        """
        if zone not in self.crowd_system.venue_map:
            return {"error": f"Zone {zone} not found."}
        
        old_capacity = self.crowd_system.venue_map[zone]
        self.crowd_system.venue_map[zone] = capacity
        
        # Recalculate max capacity
        self.crowd_system.max_capacity = sum(self.crowd_system.venue_map.values())
        
        self.crowd_system._log_event(f"Zone {zone} capacity updated from {old_capacity} to {capacity}")
        
        return {
            "status": "success",
            "zone": zone,
            "old_capacity": old_capacity,
            "new_capacity": capacity
        }
    
    def start_system(self):
        """Start the crowd management system."""
        if self.crowd_system.is_running:
            return {"status": "already_running"}
        
        self.crowd_system.start_monitoring()
        return {"status": "started"}
    
    def stop_system(self):
        """Stop the crowd management system."""
        if not self.crowd_system.is_running:
            return {"status": "not_running"}
        
        self.crowd_system.stop_monitoring()
        return {"status": "stopped"}
    
    def trigger_emergency(self, emergency_type):
        """
        Trigger an emergency protocol.
        
        Args:
            emergency_type (str): The type of emergency.
            
        Returns:
            dict: Response with emergency protocol details.
        """
        return self.emergency.trigger_emergency_protocol(emergency_type)
    
    def clear_emergency(self):
        """Clear the emergency status."""
        return self.emergency.clear_emergency()


# Extended example usage
def run_extended_example():
    """Run an extended example of the crowd management system."""
    # Define venue map - zones and their capacities
    venue_map = {
        "Entrance": 200,
        "Main Hall": 800,
        "Stage Front": 500,
        "Food Court": 300,
        "East Wing": 250,
        "West Wing": 250,
        "VIP Area": 100,
        "Restrooms": 150
    }
    
    # Initialize the system
    crowd_system = CrowdManagementSystem(
        venue_map=venue_map,
        max_capacity=2000,
        danger_threshold=0.85,
        warning_threshold=0.7
    )
    
    # Create API
    api = CrowdManagementAPI(crowd_system)
    
    # Start the system
    api.start_system()
    
    # Simulate crowd updates
    print("\nSimulating crowd updates...")
    for iteration in range(10):
        print(f"\nIteration {iteration + 1}")
        
        # Update random zones
        zones = list(venue_map.keys())
        update_zones = random.sample(zones, k=min(3, len(zones)))
        
        for zone in update_zones:
            capacity = venue_map[zone]
            # Gradually increase crowd density
            base_percentage = min(0.4 + (iteration * 0.05), 0.9)
            count = int(random.uniform(base_percentage, base_percentage + 0.1) * capacity)
            crowd_system.update_crowd_density(zone, count)
            print(f"Updated {zone}: {count} people ({count/capacity:.1%} capacity)")
        
        # Get current status
        status = api.get_system_status()
        print(f"Total crowd: {status['venue_status']['total_crowd']} people")
        print(f"Capacity: {status['venue_status']['total_crowd'] / status['venue_status']['max_capacity']:.1%}")
        
        # Get recommendations if any
        recommendations = api.get_recommendations()
        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations[:2]:  # Show only top 2
                print(f"[{rec['priority']}] {rec['action']}")
        
        # Trigger emergency if needed
        if status['venue_status']['capacity_percentage'] > 0.9:
            print("\n*** TRIGGERING EMERGENCY PROTOCOL ***")
            emergency_response = api.trigger_emergency("overcrowding")
            print(f"Emergency protocol activated: {emergency_response['emergency_type']}")
            
            # Clear emergency after handling
            api.clear_emergency()
        
        time.sleep(1)
    
    # Generate report
    print("\nGenerating final report...")
    report = crowd_system.generate_report()
    
    print(f"\nEvent Summary:")
    print(f"- Duration: {report['monitoring_duration']:.2f} hours")
    print(f"- Peak crowd: {report['overall_peak']['max_crowd']} people ({report['overall_peak']['percentage']:.1%} capacity)")
    print(f"- Total alerts: {report['alert_summary']['total_alerts']} ({report['alert_summary']['warnings']} warnings, {report['alert_summary']['dangers']} dangers)")
    
    # Export data
    export_result = api.management.export_data(format="json")
    print(f"\nData exported to {export_result.get('file', 'unknown')}")
    
    # Stop system
    api.stop_system()
    print("\nCrowd management system stopped.")

if __name__ == "__main__":
    run_extended_example()
