import tkinter as tk
from tkinter import messagebox
import time
import threading
import sys

# Check if winsound is available (it's Windows-specific)
try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False
    print("winsound module not available. Sound alerts will be disabled.")


class PomodoroTimer:
    """
    A Pomodoro Timer application with a Tkinter GUI.
    It allows users to set work, short break, and long break durations,
    and provides start, pause, and reset functionalities.
    Sound alerts are played at the end of each phase.
    """

    def __init__(self, master):
        """
        Initializes the PomodoroTimer application.

        Args:
            master: The root Tkinter window.
        """
        self.master = master
        master.title("Pomodoro Timer Bot")
        master.geometry("500x450") # Set a fixed size for the window
        master.resizable(False, False) # Make the window not resizable
        master.config(bg="#2c3e50") # Dark background for a modern look

        # --- Timer Variables ---
        self.work_duration = 25 * 60  # Default 25 minutes in seconds
        self.short_break_duration = 5 * 60  # Default 5 minutes in seconds
        self.long_break_duration = 15 * 60  # Default 15 minutes in seconds

        self.current_time_left = self.work_duration
        self.current_phase = "work"  # States: "work", "short_break", "long_break"
        self.pomodoros_completed = 0
        self.is_running = False
        self.timer_thread = None
        self.stop_event = threading.Event() # Event to signal the thread to stop

        # --- GUI Elements ---

        # Title Label
        self.title_label = tk.Label(master, text="Pomodoro Timer", font=("Inter", 24, "bold"), fg="#ecf0f1", bg="#2c3e50")
        self.title_label.pack(pady=20)

        # Phase Label
        self.phase_label = tk.Label(master, text="Work Time!", font=("Inter", 18, "italic"), fg="#f39c12", bg="#2c3e50")
        self.phase_label.pack(pady=10)

        # Time Display Label
        self.time_label = tk.Label(master, text=self._format_time(self.current_time_left),
                                   font=("Inter", 48, "bold"), fg="#ecf0f1", bg="#34495e",
                                   padx=20, pady=10, relief="solid", bd=2, highlightbackground="#f39c12", highlightthickness=2)
        self.time_label.pack(pady=20)

        # Pomodoros Completed Label
        self.pomodoro_count_label = tk.Label(master, text=f"Pomodoros: {self.pomodoros_completed}",
                                             font=("Inter", 14), fg="#bdc3c7", bg="#2c3e50")
        self.pomodoro_count_label.pack(pady=10)

        # Buttons Frame
        self.button_frame = tk.Frame(master, bg="#2c3e50")
        self.button_frame.pack(pady=10)

        self.start_button = tk.Button(self.button_frame, text="Start", command=self._start_timer,
                                       font=("Inter", 14, "bold"), bg="#27ae60", fg="white",
                                       activebackground="#2ecc71", activeforeground="white",
                                       width=10, height=2, relief="raised", bd=3, cursor="hand2")
        self.start_button.grid(row=0, column=0, padx=10, pady=5)

        self.pause_button = tk.Button(self.button_frame, text="Pause", command=self._pause_timer,
                                       font=("Inter", 14, "bold"), bg="#e67e22", fg="white",
                                       activebackground="#f39c12", activeforeground="white",
                                       width=10, height=2, relief="raised", bd=3, state=tk.DISABLED, cursor="hand2")
        self.pause_button.grid(row=0, column=1, padx=10, pady=5)

        self.reset_button = tk.Button(self.button_frame, text="Reset", command=self._reset_timer,
                                       font=("Inter", 14, "bold"), bg="#c0392b", fg="white",
                                       activebackground="#e74c3c", activeforeground="white",
                                       width=10, height=2, relief="raised", bd=3, cursor="hand2")
        self.reset_button.grid(row=0, column=2, padx=10, pady=5)

        # Settings Frame
        self.settings_frame = tk.LabelFrame(master, text="Set Durations (minutes)", font=("Inter", 12, "bold"),
                                            fg="#ecf0f1", bg="#2c3e50", bd=2, relief="groove", padx=10, pady=10)
        self.settings_frame.pack(pady=15, padx=20, fill="x")

        tk.Label(self.settings_frame, text="Work:", font=("Inter", 12), fg="#ecf0f1", bg="#2c3e50").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.work_entry = tk.Entry(self.settings_frame, width=5, font=("Inter", 12), justify="center", bg="#ecf0f1", fg="#2c3e50")
        self.work_entry.insert(0, str(self.work_duration // 60))
        self.work_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(self.settings_frame, text="Short Break:", font=("Inter", 12), fg="#ecf0f1", bg="#2c3e50").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.short_break_entry = tk.Entry(self.settings_frame, width=5, font=("Inter", 12), justify="center", bg="#ecf0f1", fg="#2c3e50")
        self.short_break_entry.insert(0, str(self.short_break_duration // 60))
        self.short_break_entry.grid(row=0, column=3, padx=5, pady=5)

        tk.Label(self.settings_frame, text="Long Break:", font=("Inter", 12), fg="#ecf0f1", bg="#2c3e50").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.long_break_entry = tk.Entry(self.settings_frame, width=5, font=("Inter", 12), justify="center", bg="#ecf0f1", fg="#2c3e50")
        self.long_break_entry.insert(0, str(self.long_break_duration // 60))
        self.long_break_entry.grid(row=0, column=5, padx=5, pady=5)

        self.set_times_button = tk.Button(self.settings_frame, text="Apply", command=self._set_times,
                                         font=("Inter", 12), bg="#3498db", fg="white",
                                         activebackground="#2980b9", activeforeground="white",
                                         relief="raised", bd=2, cursor="hand2")
        self.set_times_button.grid(row=0, column=6, padx=10, pady=5)

        self._update_display()

    def _format_time(self, seconds):
        """
        Formats seconds into MM:SS string.
        """
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def _update_display(self):
        """
        Updates the time and phase labels on the GUI.
        This method is called from the main thread using after().
        """
        self.time_label.config(text=self._format_time(self.current_time_left))
        self.pomodoro_count_label.config(text=f"Pomodoros: {self.pomodoros_completed}")

        if self.current_phase == "work":
            self.phase_label.config(text="Work Time!", fg="#f39c12")
        elif self.current_phase == "short_break":
            self.phase_label.config(text="Short Break!", fg="#2ecc71")
        elif self.current_phase == "long_break":
            self.phase_label.config(text="Long Break!", fg="#3498db")

    def _start_timer(self):
        """
        Starts or resumes the timer.
        Initializes a new timer thread if one is not running.
        """
        if not self.is_running:
            self.is_running = True
            self.stop_event.clear()  # Clear the stop event for a new run
            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.set_times_button.config(state=tk.DISABLED) # Disable settings while running

            # If current_time_left is 0, it means a phase just ended or it's a fresh start.
            # Set initial time based on current phase.
            if self.current_time_left <= 0:
                if self.current_phase == "work":
                    self.current_time_left = self.work_duration
                elif self.current_phase == "short_break":
                    self.current_time_left = self.short_break_duration
                elif self.current_phase == "long_break":
                    self.current_time_left = self.long_break_duration
                self._update_display() # Update immediately to show new time

            self.timer_thread = threading.Thread(target=self._run_timer)
            self.timer_thread.daemon = True # Allow the program to exit even if thread is running
            self.timer_thread.start()

    def _pause_timer(self):
        """
        Pauses the running timer.
        Signals the timer thread to stop.
        """
        if self.is_running:
            self.is_running = False
            self.stop_event.set()  # Set the stop event
            self.start_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)
            self.set_times_button.config(state=tk.NORMAL) # Enable settings when paused

    def _reset_timer(self):
        """
        Resets the timer to its initial state (work phase, 0 pomodoros).
        """
        self._pause_timer() # First, ensure the timer is paused and thread is signalled to stop
        self.current_time_left = self.work_duration
        self.current_phase = "work"
        self.pomodoros_completed = 0
        self._update_display()
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.set_times_button.config(state=tk.NORMAL) # Enable settings after reset

    def _run_timer(self):
        """
        This method runs in a separate thread to handle the timer countdown.
        It updates the GUI via master.after() to ensure thread safety.
        """
        while self.current_time_left > 0 and not self.stop_event.is_set():
            time.sleep(1)
            if not self.is_running or self.stop_event.is_set():
                # If paused or stop event set during sleep, break loop
                break
            self.current_time_left -= 1
            self.master.after(0, self._update_display) # Update GUI safely from main thread

        if self.current_time_left <= 0 and self.is_running:
            # Timer finished naturally (not paused/stopped)
            self._play_sound()
            self.master.after(0, self._switch_phase) # Switch phase safely from main thread
        elif self.stop_event.is_set():
            # Timer was stopped/paused
            pass # Do nothing, just exit the thread gracefully

    def _switch_phase(self):
        """
        Switches the timer to the next phase (work, short break, or long break).
        """
        if self.current_phase == "work":
            self.pomodoros_completed += 1
            if self.pomodoros_completed % 4 == 0:
                self.current_phase = "long_break"
                self.current_time_left = self.long_break_duration
                messagebox.showinfo("Pomodoro Timer", "Time for a Long Break!")
            else:
                self.current_phase = "short_break"
                self.current_time_left = self.short_break_duration
                messagebox.showinfo("Pomodoro Timer", "Time for a Short Break!")
        else: # It's a break (short or long)
            self.current_phase = "work"
            self.current_time_left = self.work_duration
            messagebox.showinfo("Pomodoro Timer", "Time to Work!")

        self._update_display()
        # Automatically restart timer for the new phase if it was running
        if self.is_running:
            self._start_timer() # This will start a new thread for the new phase

    def _play_sound(self):
        """
        Plays a simple beep sound.
        Uses winsound on Windows, otherwise prints a message.
        """
        if WINSOUND_AVAILABLE:
            try:
                winsound.Beep(1000, 500)  # Frequency 1000 Hz, Duration 500 ms
            except Exception as e:
                print(f"Error playing sound with winsound: {e}")
        else:
            print("Timer finished! (Sound alert not available)")

    def _set_times(self):
        """
        Reads the duration values from the entry fields and updates the timer settings.
        Validates input to ensure they are positive integers.
        """
        try:
            new_work = int(self.work_entry.get()) * 60
            new_short = int(self.short_break_entry.get()) * 60
            new_long = int(self.long_break_entry.get()) * 60

            if new_work <= 0 or new_short <= 0 or new_long <= 0:
                raise ValueError("Durations must be positive integers.")

            self.work_duration = new_work
            self.short_break_duration = new_short
            self.long_break_duration = new_long

            # If timer is not running, update current_time_left to new work duration
            if not self.is_running and self.current_phase == "work":
                self.current_time_left = self.work_duration
                self._update_display()

            messagebox.showinfo("Settings Updated", "Timer durations have been updated successfully!")

        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter valid positive numbers for durations.\n{e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PomodoroTimer(root)
    root.mainloop()

