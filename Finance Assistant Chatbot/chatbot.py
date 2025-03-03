import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import re
import json
import os

class FinanceEnvironment:
    """Environment for the finance agent to interact with user data."""
    
    def __init__(self):
        # Financial state parameters
        self.income = 0
        self.expenses = {}
        self.savings = 0
        self.investments = {}
        self.debt = {}
        self.financial_goals = []
        
        # State space definition
        self.state_size = 15  # Various financial indicators
        
        # Action space: different financial recommendations
        self.actions = [
            "reduce_expenses",
            "increase_savings",
            "pay_debt",
            "invest_stocks",
            "invest_bonds",
            "invest_index_funds",
            "emergency_fund",
            "retirement_planning",
            "budget_adjustment",
            "income_increase"
        ]
        self.action_size = len(self.actions)
        
        # Reward metrics
        self.previous_net_worth = 0
        
    def reset(self):
        """Reset the environment with current user data."""
        self._calculate_state()
        return self.state
    
    def _calculate_state(self):
        """Calculate the current financial state."""
        # Total expenses
        total_expenses = sum(self.expenses.values())
        
        # Total debt
        total_debt = sum(self.debt.values())
        
        # Total investments
        total_investments = sum(self.investments.values())
        
        # Calculate financial ratios and indicators
        if self.income > 0:
            expense_ratio = min(total_expenses / self.income, 1) if self.income > 0 else 1
            savings_ratio = min(self.savings / self.income, 1) if self.income > 0 else 0
            debt_to_income = min(total_debt / self.income, 1) if self.income > 0 else 1
        else:
            expense_ratio = 1
            savings_ratio = 0
            debt_to_income = 1
            
        # Define emergency months (savings / monthly expenses)
        emergency_months = self.savings / (total_expenses/12) if total_expenses > 0 else 0
        emergency_months = min(emergency_months, 12) / 12  # Normalize to 0-1
        
        # Investment diversification score (simplified)
        investment_types = len(self.investments)
        diversification = min(investment_types / 5, 1)  # Normalize assuming 5 types is fully diversified
        
        # Goal achievement progress
        goals_progress = sum(goal.get("progress", 0) for goal in self.financial_goals) / max(len(self.financial_goals), 1)
        
        # Construct the state vector
        self.state = np.array([
            self.income / 10000,  # Normalized income
            expense_ratio,
            savings_ratio,
            debt_to_income,
            emergency_months,
            diversification,
            goals_progress,
            total_debt / 10000,  # Normalized debt
            total_investments / 10000,  # Normalized investments
            self.savings / 10000,  # Normalized savings
            len(self.financial_goals) / 5,  # Normalized goal count
            total_expenses / self.income if self.income > 0 else 1,  # Expense to income ratio
            min(self.investments.get("stocks", 0) / max(total_investments, 1), 1),  # Stock allocation
            min(self.investments.get("bonds", 0) / max(total_investments, 1), 1),  # Bond allocation
            min(self.investments.get("index_funds", 0) / max(total_investments, 1), 1)  # Index fund allocation
        ])
        
        return self.state
    
    def step(self, action):
        """Take action and return new state, reward, and done flag."""
        action_taken = self.actions[action]
        
        # Calculate current net worth
        current_net_worth = self.calculate_net_worth()
        
        # Take action (in real application, this would involve user interaction)
        # For simulation, we'll assume each action has some effect on the financial state
        self._simulate_action_effect(action_taken)
        
        # Recalculate state
        new_state = self._calculate_state()
        
        # Calculate reward
        new_net_worth = self.calculate_net_worth()
        financial_health_change = new_net_worth - self.previous_net_worth
        
        # Reward is based on improvement in financial health
        reward = financial_health_change
        
        # Add bonuses for specific good financial practices
        if action_taken == "emergency_fund" and self.state[4] < 0.5:  # If emergency fund is low
            reward += 1
        elif action_taken == "pay_debt" and self.state[3] > 0.4:  # If debt-to-income is high
            reward += 1
        elif action_taken == "reduce_expenses" and self.state[1] > 0.7:  # If expense ratio is high
            reward += 1
        
        # Update previous net worth
        self.previous_net_worth = new_net_worth
        
        # In this simulation, episodes don't end
        done = False
        
        return new_state, reward, done, {"action": action_taken}
    
    def _simulate_action_effect(self, action):
        """Simulate the effect of taking an action on the financial state."""
        # This is a simplified simulation for the agent to learn
        if action == "reduce_expenses":
            for category in self.expenses:
                self.expenses[category] *= 0.95  # Reduce expenses by 5%
                
        elif action == "increase_savings":
            savings_increase = 0.05 * self.income
            self.savings += savings_increase
            
        elif action == "pay_debt":
            debt_payment = 0.1 * self.savings
            if debt_payment > 0 and self.debt:
                highest_interest_debt = max(self.debt.items(), key=lambda x: x[1].get("interest_rate", 0))[0]
                self.debt[highest_interest_debt]["balance"] -= debt_payment
                self.savings -= debt_payment
                if self.debt[highest_interest_debt]["balance"] <= 0:
                    del self.debt[highest_interest_debt]
            
        elif action == "invest_stocks":
            investment_amount = 0.1 * self.savings
            self.savings -= investment_amount
            self.investments["stocks"] = self.investments.get("stocks", 0) + investment_amount
            
        elif action == "invest_bonds":
            investment_amount = 0.1 * self.savings
            self.savings -= investment_amount
            self.investments["bonds"] = self.investments.get("bonds", 0) + investment_amount
            
        elif action == "invest_index_funds":
            investment_amount = 0.1 * self.savings
            self.savings -= investment_amount
            self.investments["index_funds"] = self.investments.get("index_funds", 0) + investment_amount
            
        elif action == "emergency_fund":
            if self.income > 0:
                monthly_expenses = sum(self.expenses.values()) / 12
                target_emergency = monthly_expenses * 6  # 6 months of expenses
                if self.savings < target_emergency:
                    # Redirect some money to savings
                    savings_increase = 0.1 * self.income
                    self.savings += savings_increase
            
        elif action == "retirement_planning":
            investment_amount = 0.05 * self.income
            self.investments["retirement"] = self.investments.get("retirement", 0) + investment_amount
            
        elif action == "budget_adjustment":
            # Rebalance budget to recommended percentages
            if self.income > 0:
                # Ideal: 50% needs, 30% wants, 20% savings
                total_expenses = sum(self.expenses.values())
                if total_expenses > 0.8 * self.income:
                    for category in self.expenses:
                        self.expenses[category] *= 0.8 * self.income / total_expenses
            
        elif action == "income_increase":
            # Simulate pursuing income increase opportunities
            self.income *= 1.02  # 2% increase
    
    def calculate_net_worth(self):
        """Calculate the current net worth."""
        assets = self.savings + sum(self.investments.values())
        liabilities = sum(debt_info.get("balance", 0) for debt_info in self.debt.values())
        return assets - liabilities
    
    def update_from_user_data(self, user_data):
        """Update environment with real user data."""
        if "income" in user_data:
            self.income = user_data["income"]
        if "expenses" in user_data:
            self.expenses = user_data["expenses"]
        if "savings" in user_data:
            self.savings = user_data["savings"]
        if "investments" in user_data:
            self.investments = user_data["investments"]
        if "debt" in user_data:
            self.debt = user_data["debt"]
        if "financial_goals" in user_data:
            self.financial_goals = user_data["financial_goals"]
        
        # Update state
        self._calculate_state()
        self.previous_net_worth = self.calculate_net_worth()


class DQNAgent:
    """Deep Q-Learning agent for financial recommendations."""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        """Build a neural network model for DQN."""
        model = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model."""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose an action based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Train model with experiences from memory."""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
        targets = rewards + self.gamma * (np.amax(self.target_model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)
        
        ind = np.array([i for i in range(batch_size)])
        targets_full[ind, actions] = targets
        
        self.model.fit(states, targets_full, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load model weights."""
        self.model.load_weights(name)
        
    def save(self, name):
        """Save model weights."""
        self.model.save_weights(name)


class FinanceAgent:
    """Main finance agent with chat interface."""
    
    def __init__(self):
        self.environment = FinanceEnvironment()
        self.agent = DQNAgent(self.environment.state_size, self.environment.action_size)
        self.user_data = self._load_user_data()
        self.intents = self._create_intents()
        self.last_state = None
        self.last_action = None
        
    def _load_user_data(self):
        """Load user financial data from file or create default."""
        try:
            if os.path.exists('user_financial_data.json'):
                with open('user_financial_data.json', 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading user data: {e}")
        
        # Return default data structure
        return {
            "income": 0,
            "expenses": {},
            "savings": 0,
            "investments": {},
            "debt": {},
            "financial_goals": []
        }
    
    def _save_user_data(self):
        """Save user financial data to file."""
        with open('user_financial_data.json', 'w') as f:
            json.dump(self.user_data, f, indent=2)
    
    def _create_intents(self):
        """Create intent patterns for chat functionality."""
        return {
            "greeting": [
                r"hello", r"hi", r"hey", r"good morning", r"good afternoon",
                r"greetings", r"howdy"
            ],
            "add_income": [
                r"(set|update) income to (\d+)",
                r"my income is (\d+)",
                r"i (make|earn) (\d+)"
            ],
            "add_expense": [
                r"add (?:an |a )?expense (?:of )?(\d+) for ([a-zA-Z ]+)",
                r"i spend (\d+) on ([a-zA-Z ]+)",
                r"(\d+) (?:for|on) ([a-zA-Z ]+)"
            ],
            "add_savings": [
                r"(?:i have|add|set) (\d+) in savings",
                r"update savings to (\d+)",
                r"savings (?:is|are) (\d+)"
            ],
            "add_investment": [
                r"add (?:an |a )?investment (?:of )?(\d+) in ([a-zA-Z ]+)",
                r"i have (\d+) invested in ([a-zA-Z ]+)",
                r"(\d+) invested in ([a-zA-Z ]+)"
            ],
            "add_debt": [
                r"add (?:a )?debt (?:of )?(\d+) for ([a-zA-Z ]+)(?: with interest (\d+(?:\.\d+)?)%)?",
                r"i owe (\d+) (?:for|on) ([a-zA-Z ]+)(?: at (\d+(?:\.\d+)?)%)?",
                r"(\d+) debt for ([a-zA-Z ]+)(?: at (\d+(?:\.\d+)?)%)?",
            ],
            "add_goal": [
                r"add goal to ([a-zA-Z ]+) (?:by|in) (\d+) (?:months|years)",
                r"i want to ([a-zA-Z ]+) (?:by|in) (\d+) (?:months|years)",
                r"goal: ([a-zA-Z ]+) (?:by|in) (\d+) (?:months|years)"
            ],
            "view_summary": [
                r"summary", r"overview", r"status", r"financial state",
                r"how (?:am i|are my finances) doing", r"show me my finances",
                r"what's my financial situation"
            ],
            "get_recommendation": [
                r"what should i do", r"give me advice", r"recommend something",
                r"help me", r"suggestions", r"recommendations", r"how can i improve",
                r"financial advice"
            ],
            "train_agent": [
                r"train", r"learn", r"improve yourself", r"get smarter",
                r"update your model"
            ],
            "goodbye": [
                r"bye", r"goodbye", r"see you", r"exit", r"quit", r"end"
            ]
        }
    
    def _match_intent(self, message):
        """Match user message to an intent."""
        for intent, patterns in self.intents.items():
            for pattern in patterns:
                match = re.search(pattern, message.lower())
                if match:
                    return intent, match
        return "unknown", None
    
    def get_recommended_action(self):
        """Get recommendation from the DQN agent."""
        if self.last_state is None:
            self.last_state = self.environment.reset()
        
        # Get action from agent
        action_idx = self.agent.act(self.last_state)
        action = self.environment.actions[action_idx]
        self.last_action = action_idx
        
        return action
    
    def give_feedback(self, reward):
        """Give feedback to the agent about its last recommendation."""
        if self.last_state is not None and self.last_action is not None:
            # Take the action in the environment
            next_state, env_reward, done, _ = self.environment.step(self.last_action)
            
            # Combine environment reward with user feedback
            total_reward = env_reward + reward
            
            # Store in memory
            self.agent.remember(self.last_state, self.last_action, total_reward, next_state, done)
            
            # Update last state
            self.last_state = next_state
    
    def train(self, episodes=10):
        """Train the DQN agent."""
        # Update environment with latest user data
        self.environment.update_from_user_data(self.user_data)
        
        for episode in range(episodes):
            state = self.environment.reset()
            total_reward = 0
            
            for step in range(100):  # Limit steps per episode
                action = self.agent.act(state)
                next_state, reward, done, _ = self.environment.step(action)
                total_reward += reward
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                
                if len(self.agent.memory) > 32:
                    self.agent.replay(32)
                
                if done:
                    break
            
            # Update target network every episode
            if episode % 5 == 0:
                self.agent.update_target_model()
        
        return "Training completed. I've improved my recommendations based on your financial situation."
    
    def format_action_as_advice(self, action):
        """Convert an action to natural language advice."""
        advice_map = {
            "reduce_expenses": "I recommend looking for ways to reduce your expenses. Consider reviewing your subscriptions, finding cheaper alternatives, or cutting non-essential spending.",
            
            "increase_savings": "You should focus on increasing your savings rate. Try to save a bit more each month by setting up automatic transfers to your savings account.",
            
            "pay_debt": "Prioritize paying down your debt, especially high-interest debt like credit cards. This will improve your financial health in the long run.",
            
            "invest_stocks": "Consider allocating some of your savings to stock investments for long-term growth. Diversify your portfolio across different companies and sectors.",
            
            "invest_bonds": "Adding bonds to your investment portfolio could provide more stability. They generally offer lower returns but with less risk than stocks.",
            
            "invest_index_funds": "Index funds are a great way to invest with built-in diversification. Consider adding these to your investment strategy for long-term wealth building.",
            
            "emergency_fund": "Focus on building or strengthening your emergency fund. Aim for 3-6 months of essential expenses in an easily accessible account.",
            
            "retirement_planning": "It's important to think about your retirement. Consider increasing contributions to retirement accounts like 401(k)s or IRAs.",
            
            "budget_adjustment": "Your budget may need some rebalancing. Try to follow the 50/30/20 rule: 50% for needs, 30% for wants, and 20% for savings and debt repayment.",
            
            "income_increase": "Looking for ways to increase your income would significantly improve your financial situation. Consider asking for a raise, developing new skills, or starting a side hustle."
        }
        
        return advice_map.get(action, "I recommend reviewing your overall financial strategy.")
    
    def generate_response(self, message):
        """Generate response based on user message."""
        intent, match = self._match_intent(message)
        
        if intent == "greeting":
            return "Hello! I'm your financial assistant. How can I help you manage your finances today?"
        
        elif intent == "add_income":
            income = int(match.group(2))
            self.user_data["income"] = income
            self.environment.update_from_user_data(self.user_data)
            self._save_user_data()
            return f"Got it! Your income is set to ${income}."
        
        elif intent == "add_expense":
            amount = int(match.group(1))
            category = match.group(2).strip()
            self.user_data["expenses"][category] = amount
            self.environment.update_from_user_data(self.user_data)
            self._save_user_data()
            return f"Added ${amount} as an expense for {category}."
        
        elif intent == "add_savings":
            amount = int(match.group(1))
            self.user_data["savings"] = amount
            self.environment.update_from_user_data(self.user_data)
            self._save_user_data()
            return f"Updated your savings to ${amount}."
        
        elif intent == "add_investment":
            amount = int(match.group(1))
            category = match.group(2).strip()
            self.user_data["investments"][category] = amount
            self.environment.update_from_user_data(self.user_data)
            self._save_user_data()
            return f"Added ${amount} investment in {category}."
        
        elif intent == "add_debt":
            amount = int(match.group(1))
            category = match.group(2).strip()
            interest = float(match.group(3)) if match.group(3) else 0
            self.user_data["debt"][category] = {"balance": amount, "interest_rate": interest}
            self.environment.update_from_user_data(self.user_data)
            self._save_user_data()
            interest_text = f" at {interest}% interest" if interest else ""
            return f"Added ${amount} debt for {category}{interest_text}."
        
        elif intent == "add_goal":
            goal_desc = match.group(1).strip()
            timeframe = int(match.group(2))
            unit = "months" if "month" in message else "years"
            self.user_data["financial_goals"].append({"description": goal_desc, "timeframe": timeframe, "unit": unit, "progress": 0})
            self.environment.update_from_user_data(self.user_data)
            self._save_user_data()
            return f"Added your goal to {goal_desc} in {timeframe} {unit}."
        
        elif intent == "view_summary":
            total_expenses = sum(self.user_data["expenses"].values())
            total_investments = sum(self.user_data["investments"].values())
            total_debt = sum(debt_info["balance"] for debt_info in self.user_data["debt"].values())
            net_worth = self.user_data["savings"] + total_investments - total_debt
            
            summary = f"Financial Summary:\n"
            summary += f"Income: ${self.user_data['income']}/month\n"
            summary += f"Total Expenses: ${total_expenses}/month\n"
            summary += f"Savings: ${self.user_data['savings']}\n"
            summary += f"Investments: ${total_investments}\n"
            summary += f"Debt: ${total_debt}\n"
            summary += f"Net Worth: ${net_worth}\n"
            
            if self.user_data["income"] > 0:
                savings_rate = (self.user_data["income"] - total_expenses) / self.user_data["income"] * 100
                summary += f"Savings Rate: {savings_rate:.1f}%\n"
            
            return summary
        
        elif intent == "get_recommendation":
            if sum(len(data) for data in [self.user_data["expenses"], self.user_data["investments"], self.user_data["debt"]]) == 0:
                return "I need more information about your finances to give personalized recommendations. Try telling me about your income, expenses, savings, investments, or debt."
            
            action = self.get_recommended_action()
            advice = self.format_action_as_advice(action)
            return f"Based on your financial situation, {advice} Would this advice be helpful for you?"
        
        elif intent == "train_agent":
            return self.train()
        
        elif intent == "goodbye":
            return "Goodbye! Take care of your finances, and I'll be here when you need more advice."
        
        else:
            return "I'm not sure I understand. You can tell me about your income, expenses, savings, investments, or debt. Or ask for recommendations on what to do next."


class FinanceChatbot:
    """Chat interface for the finance agent."""
    
    def __init__(self):
        self.agent = FinanceAgent()
        self.context = []
    
    def process_message(self, message):
        """Process user message and return response."""
        # Add to context
        self.context.append({"role": "user", "content": message})
        
        # Get response from agent
        response = self.agent.generate_response(message)
        
        # Add to context
        self.context.append({"role": "assistant", "content": response})
        
        return response

# Example usage of the chatbot
def main():
    chatbot = FinanceChatbot()
    print("Financial Assistant: Hello! I'm your AI financial advisor. I can help you manage your budget, savings, and investments.")
    print("Financial Assistant: Tell me about your income, expenses, savings, or ask for recommendations.")
    print("\nCommands: ")
    print("- Set income: 'my income is 5000'")
    print("- Add expense: 'I spend 1200 on rent'")
    print("- Add savings: 'I have 10000 in savings'")
    print("- Add debt: 'I owe 15000 for student loans at 4.5%'")
    print("- Add investment: 'I have 20000 invested in stocks'")
    print("- Add goal: 'I want to buy a house in 5 years'")
    print("- Get summary: 'show me my finances'")
    print("- Get recommendation: 'what should I do?'")
    print("- Type 'exit' to quit")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Financial Assistant: Goodbye! Take care of your finances.")
            break
        
        response = chatbot.process_message(user_input)
        print(f"Financial Assistant: {response}")

if __name__ == "__main__":
    main()
