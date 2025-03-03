import tkinter as tk
from tkinter import ttk, messagebox, font
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
from PIL import Image, ImageTk
import os
from ttkthemes import ThemedTk
import io

# Create necessary directories if they don't exist
os.makedirs("assets", exist_ok=True)

class UltraModernUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Flight Delay Predictor")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1E1E2E")
        
        # Set custom theme
        self.root.set_theme("equilux")
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#1E1E2E")
        self.style.configure("TLabel", background="#1E1E2E", foreground="#FFFFFF", font=("Segoe UI", 10))
        self.style.configure("TButton", background="#6C5CE7", foreground="#FFFFFF", font=("Segoe UI", 10, "bold"))
        self.style.configure("Header.TLabel", background="#1E1E2E", foreground="#FFFFFF", font=("Segoe UI", 16, "bold"))
        self.style.configure("Title.TLabel", background="#1E1E2E", foreground="#FFFFFF", font=("Segoe UI", 24, "bold"))
        self.style.configure("Card.TFrame", background="#2D2D44", relief="flat", borderwidth=0)
        self.style.configure("TEntry", font=("Segoe UI", 10))
        self.style.configure("TCombobox", font=("Segoe UI", 10))
        
        # Configure custom ttk widgets
        self.style.map("TButton", 
            background=[("active", "#8A7CE8"), ("pressed", "#5546D6")],
            foreground=[("active", "#FFFFFF"), ("pressed", "#FFFFFF")])
        
        # Sample data
        self.airlines = ["Airline A", "Airline B", "Airline C", "Airline D", "Airline E"]
        self.days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        self.airports = ["ORD", "ATL", "DFW", "LAX", "JFK", "MIA", "SFO", "SEA", "DEN", "IAH"]
        
        # Create main layout
        self.create_header()
        self.create_main_content()
        
        # Initialize plots with dummy data
        self.create_dashboard_charts()
    
    def create_header(self):
        """Create the application header"""
        header_frame = ttk.Frame(self.root, style="TFrame")
        header_frame.pack(fill="x", padx=20, pady=(20, 0))
        
        # App title
        title_label = ttk.Label(header_frame, text="Flight Delay Predictor", style="Title.TLabel")
        title_label.pack(side="left", pady=10)
        
        # Model selection
        model_frame = ttk.Frame(header_frame, style="TFrame")
        model_frame.pack(side="right", pady=10)
        
        ttk.Label(model_frame, text="Model:", style="TLabel").pack(side="left", padx=(0, 5))
        model_combo = ttk.Combobox(model_frame, values=["XGBoost", "Random Forest", "Neural Network"], width=15, state="readonly")
        model_combo.current(0)
        model_combo.pack(side="left")
        
    def create_main_content(self):
        """Create the main content area"""
        main_frame = ttk.Frame(self.root, style="TFrame")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Create left panel (input form)
        left_panel = ttk.Frame(main_frame, style="Card.TFrame")
        left_panel.pack(side="left", fill="y", padx=(0, 10), ipadx=20, ipady=20, expand=False)
        
        # Prediction form title
        ttk.Label(left_panel, text="Flight Information", style="Header.TLabel").pack(pady=(20, 30), padx=20)
        
        # Create form fields
        form_frame = ttk.Frame(left_panel, style="Card.TFrame")
        form_frame.pack(fill="x", padx=20, pady=0)
        
        # Airline
        ttk.Label(form_frame, text="Airline:", style="TLabel").pack(anchor="w", pady=(0, 5))
        self.airline_combo = ttk.Combobox(form_frame, values=self.airlines, width=25, state="readonly")
        self.airline_combo.current(0)
        self.airline_combo.pack(fill="x", pady=(0, 15))
        
        # Day of Week
        ttk.Label(form_frame, text="Day of Week:", style="TLabel").pack(anchor="w", pady=(0, 5))
        self.day_combo = ttk.Combobox(form_frame, values=self.days_of_week, width=25, state="readonly")
        self.day_combo.current(0)
        self.day_combo.pack(fill="x", pady=(0, 15))
        
        # Origin and Destination
        origin_dest_frame = ttk.Frame(form_frame, style="Card.TFrame")
        origin_dest_frame.pack(fill="x", pady=(0, 15))
        
        # Origin
        origin_frame = ttk.Frame(origin_dest_frame, style="Card.TFrame")
        origin_frame.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Label(origin_frame, text="Origin:", style="TLabel").pack(anchor="w", pady=(0, 5))
        self.origin_combo = ttk.Combobox(origin_frame, values=self.airports, width=10, state="readonly")
        self.origin_combo.current(0)
        self.origin_combo.pack(fill="x")
        
        # Destination
        dest_frame = ttk.Frame(origin_dest_frame, style="Card.TFrame")
        dest_frame.pack(side="right", fill="x", expand=True, padx=(5, 0))
        ttk.Label(dest_frame, text="Destination:", style="TLabel").pack(anchor="w", pady=(0, 5))
        self.dest_combo = ttk.Combobox(dest_frame, values=self.airports, width=10, state="readonly")
        self.dest_combo.current(3)
        self.dest_combo.pack(fill="x")
        
        # Departure Time
        ttk.Label(form_frame, text="Departure Time (24hr):", style="TLabel").pack(anchor="w", pady=(0, 5))
        self.dep_time_entry = ttk.Entry(form_frame)
        self.dep_time_entry.insert(0, "1200")
        self.dep_time_entry.pack(fill="x", pady=(0, 15))
        
        # Date selection
        date_frame = ttk.Frame(form_frame, style="Card.TFrame")
        date_frame.pack(fill="x", pady=(0, 20))
        
        # Month
        month_frame = ttk.Frame(date_frame, style="Card.TFrame")
        month_frame.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Label(month_frame, text="Month:", style="TLabel").pack(anchor="w", pady=(0, 5))
        self.month_combo = ttk.Combobox(month_frame, values=[str(i) for i in range(1, 13)], width=5, state="readonly")
        self.month_combo.current(0)
        self.month_combo.pack(fill="x")
        
        # Day
        day_frame = ttk.Frame(date_frame, style="Card.TFrame")
        day_frame.pack(side="right", fill="x", expand=True, padx=(5, 0))
        ttk.Label(day_frame, text="Day:", style="TLabel").pack(anchor="w", pady=(0, 5))
        self.day_entry_combo = ttk.Combobox(day_frame, values=[str(i) for i in range(1, 32)], width=5, state="readonly")
        self.day_entry_combo.current(0)
        self.day_entry_combo.pack(fill="x")
        
        # Predict button
        predict_button = ttk.Button(form_frame, text="PREDICT DELAY", command=self.predict_delay)
        predict_button.pack(fill="x", pady=(15, 0))
        
        # Results area
        self.result_frame = ttk.Frame(left_panel, style="Card.TFrame")
        self.result_frame.pack(fill="x", padx=20, pady=(30, 20))
        
        ttk.Label(self.result_frame, text="Prediction Result", style="Header.TLabel").pack(pady=(10, 20))
        
        # Placeholder for prediction result
        self.result_label = ttk.Label(self.result_frame, text="Submit flight details to predict delay", style="TLabel")
        self.result_label.pack(pady=(0, 10))
        
        # Progress bar placeholder
        self.progress_frame = ttk.Frame(self.result_frame, style="Card.TFrame")
        self.progress_frame.pack(fill="x", padx=10, pady=(0, 20))
        
        self.progress = ttk.Progressbar(self.progress_frame, mode="determinate", length=200)
        self.progress.pack(fill="x")
        
        # Right panel - Dashboard
        right_panel = ttk.Frame(main_frame, style="TFrame")
        right_panel.pack(side="right", fill="both", expand=True)
        
        # Dashboard title
        ttk.Label(right_panel, text="Flight Delay Analytics", style="Header.TLabel").pack(pady=(0, 20))
        
        # Charts container
        self.charts_frame = ttk.Frame(right_panel, style="TFrame")
        self.charts_frame.pack(fill="both", expand=True)
        
        # Top row charts
        top_charts_frame = ttk.Frame(self.charts_frame, style="TFrame")
        top_charts_frame.pack(fill="x", expand=True)
        
        # Chart 1 - Airline Delay %
        self.airline_chart_frame = ttk.Frame(top_charts_frame, style="Card.TFrame")
        self.airline_chart_frame.pack(side="left", fill="both", expand=True, padx=(0, 5), pady=(0, 5))
        
        # Chart 2 - Day of Week Delay %
        self.day_chart_frame = ttk.Frame(top_charts_frame, style="Card.TFrame")
        self.day_chart_frame.pack(side="right", fill="both", expand=True, padx=(5, 0), pady=(0, 5))
        
        # Bottom chart - Origin Airport Delay %
        self.origin_chart_frame = ttk.Frame(self.charts_frame, style="Card.TFrame")
        self.origin_chart_frame.pack(fill="both", expand=True, pady=(5, 0))
    
    def create_dashboard_charts(self):
        """Create the visualization charts for the dashboard"""
        # Sample data
        airline_delay_data = {
            'Airline A': 18.5,
            'Airline B': 15.2,
            'Airline C': 12.7,
            'Airline D': 10.9,
            'Airline E': 8.3
        }
        
        day_of_week_data = {
            'Monday': 14.2,
            'Tuesday': 12.5,
            'Wednesday': 11.8,
            'Thursday': 13.4,
            'Friday': 16.7,
            'Saturday': 10.3,
            'Sunday': 15.1
        }
        
        origin_airport_data = {
            'ORD': 19.7,
            'ATL': 18.2,
            'DFW': 17.5,
            'LAX': 16.3,
            'JFK': 15.8,
            'MIA': 14.9,
            'SFO': 14.2,
            'SEA': 13.8,
            'DEN': 12.9,
            'IAH': 12.4
        }
        
        # Create airline chart
        self.create_bar_chart(
            self.airline_chart_frame,
            "Delay Risk by Airline",
            list(airline_delay_data.keys()),
            list(airline_delay_data.values()),
            "#6C5CE7"
        )
        
        # Create day of week chart
        self.create_line_chart(
            self.day_chart_frame,
            "Delay by Day of Week",
            list(day_of_week_data.keys()),
            list(day_of_week_data.values()),
            "#00BFFF"
        )
        
        # Create origin airport chart
        self.create_bar_chart(
            self.origin_chart_frame,
            "Top Delay-Prone Origin Airports",
            list(origin_airport_data.keys())[:5],
            list(origin_airport_data.values())[:5],
            "#FF6B6B"
        )
    
    def create_bar_chart(self, parent, title, x_data, y_data, color):
        """Create a bar chart visualization"""
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        fig.patch.set_facecolor('#2D2D44')
        ax.set_facecolor('#2D2D44')
        
        # Plot data
        bars = ax.bar(x_data, y_data, color=color, alpha=0.8)
        
        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{height:.1f}%', ha='center', va='bottom', color='white')
        
        # Customize the plot
        ax.set_title(title, color='white', fontsize=12, pad=10)
        ax.set_xlabel('', color='white')
        ax.set_ylabel('Delay %', color='white')
        ax.tick_params(axis='x', colors='white', rotation=45)
        ax.tick_params(axis='y', colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)
    
    def create_line_chart(self, parent, title, x_data, y_data, color):
        """Create a line chart visualization"""
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        fig.patch.set_facecolor('#2D2D44')
        ax.set_facecolor('#2D2D44')
        
        # Plot data
        ax.plot(x_data, y_data, marker='o', linewidth=3, color=color)
        
        # Add data labels
        for i, v in enumerate(y_data):
            ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', color='white')
        
        # Customize the plot
        ax.set_title(title, color='white', fontsize=12, pad=10)
        ax.set_xlabel('', color='white')
        ax.set_ylabel('Delay %', color='white')
        ax.tick_params(axis='x', colors='white', rotation=45)
        ax.tick_params(axis='y', colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)
    
    def predict_delay(self):
        """Predict flight delay based on form inputs"""
        # Get form values
        airline = self.airline_combo.get()
        day_of_week = self.day_combo.get()
        origin = self.origin_combo.get()
        destination = self.dest_combo.get()
        
        try:
            dep_time = int(self.dep_time_entry.get())
            if dep_time < 0 or dep_time > 2359:
                messagebox.showerror("Input Error", "Departure time must be between 0000 and 2359")
                return
        except ValueError:
            messagebox.showerror("Input Error", "Departure time must be a number")
            return
        
        # Simulate prediction (would connect to your XGBoost model)
        # For now, generate a random probability with some logic based on inputs
        base_probability = 0.2
        
        # Adjust probability based on airline
        airline_factors = {
            "Airline A": 0.15,
            "Airline B": 0.10,
            "Airline C": 0.05,
            "Airline D": -0.05,
            "Airline E": -0.10
        }
        base_probability += airline_factors.get(airline, 0)
        
        # Adjust probability based on day of week
        day_factors = {
            "Monday": 0.05,
            "Tuesday": -0.05,
            "Wednesday": -0.07,
            "Thursday": 0.0,
            "Friday": 0.12,
            "Saturday": -0.10,
            "Sunday": 0.08
        }
        base_probability += day_factors.get(day_of_week, 0)
        
        # Add some random variation
        final_probability = min(max(base_probability + random.uniform(-0.05, 0.05), 0.05), 0.95)
        
        # Update progress bar
        self.progress["value"] = final_probability * 100
        
        # Update result text
        if final_probability < 0.3:
            result_text = f"Low risk of delay: {final_probability:.1%}"
            self.result_label.configure(text=result_text, foreground="#00CC66")
        elif final_probability < 0.6:
            result_text = f"Medium risk of delay: {final_probability:.1%}"
            self.result_label.configure(text=result_text, foreground="#FFCC00")
        else:
            result_text = f"High risk of delay: {final_probability:.1%}"
            self.result_label.configure(text=result_text, foreground="#FF6666")
        
        # Add recommendation
        if final_probability >= 0.5:
            recommendation = ttk.Label(
                self.result_frame, 
                text="Consider alternative arrangements or buffer time.",
                style="TLabel",
                foreground="#FF9999"
            )
            
            # Remove previous recommendation if exists
            for widget in self.result_frame.winfo_children():
                if widget != self.result_label and widget != self.progress_frame:
                    widget.destroy()
                    
            recommendation.pack(pady=(0, 10))

# Main application
if __name__ == "__main__":
    root = ThemedTk(theme="equilux")
    app = UltraModernUI(root)
    root.mainloop()
