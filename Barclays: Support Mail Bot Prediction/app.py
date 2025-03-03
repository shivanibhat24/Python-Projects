import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import pickle
from PIL import Image
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64
from support_mail_classifier import SupportMailClassifier

# Set page config
st.set_page_config(
    page_title="Support Mail Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for dark mode
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Custom CSS for light/dark mode
def apply_custom_css():
    if st.session_state.dark_mode:
        # Dark mode
        st.markdown("""
        <style>
        .main {
            background-color: #1E1E1E;
            color: #F0F0F0;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 10px 24px;
            transition: all 0.3s;
        }
        .stButton button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .css-1d391kg, .css-keje6w {
            background-color: #2D2D2D;
        }
        .st-bw {
            background-color: #2D2D2D;
        }
        .st-cd {
            border-color: #555;
        }
        .stTextInput input, .stTextArea textarea {
            background-color: #333;
            color: #F0F0F0;
            border: 1px solid #555;
        }
        .stSelectbox div, .stMultiselect div {
            background-color: #333;
            color: #F0F0F0;
        }
        .dashboard-container {
            background-color: #2D2D2D;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: #3D3D3D;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            text-align: center;
            transition: transform 0.3s ease-in-out;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.3);
        }
        .metric-title {
            font-size: 1.2em;
            color: #CCC;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #4CAF50;
        }
        .card-subtitle {
            color: #AAA;
            font-size: 0.9em;
        }
        .css-18e3th9 {
            padding-top: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Light mode
        st.markdown("""
        <style>
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 10px 24px;
            transition: all 0.3s;
        }
        .stButton button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .dashboard-container {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            text-align: center;
            transition: transform 0.3s ease-in-out;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        }
        .metric-title {
            font-size: 1.2em;
            color: #555;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #4CAF50;
        }
        .card-subtitle {
            color: #777;
            font-size: 0.9em;
        }
        .css-18e3th9 {
            padding-top: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)

apply_custom_css()

# Sample knowledge base data
knowledge_base = {
    "login_issues": {
        "solution": "Try clearing your browser cache and cookies, or use incognito mode. If problems persist, reset your password.",
        "urls": ["https://support.example.com/login-troubleshooting", "https://support.example.com/browser-cache"]
    },
    "password_reset": {
        "solution": "Click 'Forgot Password' on the login page and follow the instructions sent to your email.",
        "urls": ["https://support.example.com/reset-password"]
    },
    "performance": {
        "solution": "Check your internet connection speed. If it's normal, try refreshing the page or clearing your cache.",
        "urls": ["https://support.example.com/performance-issues", "https://support.example.com/system-requirements"]
    },
    "error": {
        "solution": "Note the error code and check our error code guide. Most common issues can be resolved by refreshing or logging out and back in.",
        "urls": ["https://support.example.com/error-codes"]
    },
    "release_info": {
        "solution": "Check our release schedule on the main support page. We typically release updates on the first Monday of each month.",
        "urls": ["https://support.example.com/release-schedule"]
    },
    "system_crash": {
        "solution": "Clear your browser cache, update your browser to the latest version, and ensure your system meets our requirements.",
        "urls": ["https://support.example.com/crash-troubleshooting"]
    },
    "billing": {
        "solution": "Access your billing information from the account settings page. For specific issues, please contact our billing department.",
        "urls": ["https://support.example.com/billing-faq"]
    },
    "data_export": {
        "solution": "Go to Settings > Data Management > Export. You can select the data format and date range for your export.",
        "urls": ["https://support.example.com/data-export-guide"]
    },
    "dashboard": {
        "solution": "Try refreshing your dashboard. If data is still missing, check if there are any ongoing system notifications.",
        "urls": ["https://support.example.com/dashboard-troubleshooting"]
    },
    "account_setup": {
        "solution": "Follow our account setup guide. Make sure to verify your email and complete all required profile fields.",
        "urls": ["https://support.example.com/account-setup-guide"]
    }
}

# Sample response templates
response_templates = {
    "login_issues": "Dear Customer, Thank you for contacting support. To resolve your login issue, please try clearing your browser cache and cookies or use incognito mode. If the problem persists, we recommend resetting your password. You can find detailed instructions at our knowledge base. Let us know if you need further assistance.",
    "password_reset": "Dear Customer, To reset your password, please click the 'Forgot Password' link on the login page. You'll receive an email with instructions to create a new password. If you don't receive the email, please check your spam folder. Please let us know if you need additional help.",
    "performance": "Dear Customer, We understand you're experiencing performance issues. First, please check your internet connection speed. If your connection is stable, try refreshing the page or clearing your browser cache. Our system requirements can be found in our knowledge base. Please contact us again if these steps don't resolve your issue.",
    "error": "Dear Customer, We apologize for the error you encountered. Please note the error code if available, as this helps us diagnose the issue. Most common errors can be resolved by refreshing the page or logging out and back in. Our error code guide provides specific solutions for each error. Let us know if you need further assistance.",
    "release_info": "Dear Customer, Thank you for your inquiry about our upcoming release. Our release schedule is available on our main support page. We typically release updates on the first Monday of each month, with patch releases as needed. You can subscribe to our newsletter for announcements about new features and improvements.",
    "system_crash": "Dear Customer, We're sorry to hear about the system crash you experienced. To resolve this issue, please try clearing your browser cache, updating your browser to the latest version, and ensuring your system meets our requirements. If the issue persists, please provide us with details about when the crash occurred and what actions you were performing at the time.",
    "billing": "Dear Customer, For billing-related inquiries, you can access your billing information from the account settings page. This includes your payment history, current plan, and options to update your payment method. For specific billing issues, please contact our dedicated billing department with your account details and a description of your concern.",
    "data_export": "Dear Customer, To export your data, go to Settings > Data Management > Export in your account. You can select the data format (CSV, Excel, or JSON) and specify the date range for your export. Large exports may take some time to process, and you'll receive an email notification when your export is ready for download.",
    "dashboard": "Dear Customer, If you're experiencing issues with your dashboard, first try refreshing the page. If data is still missing or incorrect, check if there are any system notifications about ongoing maintenance or known issues. You can also try clearing your browser cache or using a different browser to access the dashboard.",
    "account_setup": "Dear Customer, To set up your account, please follow our comprehensive account setup guide. Make sure to verify your email address and complete all required profile fields. If you're setting up multiple users, you'll need administrator privileges. Don't hesitate to contact us if you encounter any difficulties during the setup process."
}

# Function to load the model
@st.cache_resource
def load_model():
    # In a real application, check if the model file exists
    # If not, you might want to train the model or show an error
    try:
        classifier = SupportMailClassifier()
        classifier.load_model("support_mail_classifier.pkl")
        return classifier
    except:
        # If model doesn't exist, create a new one with sample data
        sample_data = {
            "emails": [
                "I can't login to my account, it says invalid password",
                "How do I reset my password?",
                "The application is very slow today",
                "I'm getting an error when I try to upload a file",
                "When will the new version be released?",
                "The system crashed while processing my report",
                "I need to update my billing information",
                "How do I export my data?",
                "The dashboard is not showing current data",
                "I need help setting up my account"
            ],
            "categories": [
                "login_issues",
                "password_reset",
                "performance",
                "error",
                "release_info",
                "system_crash",
                "billing",
                "data_export",
                "dashboard",
                "account_setup"
            ]
        }
        
        classifier = SupportMailClassifier()
        classifier.train(sample_data["emails"], sample_data["categories"])
        classifier.save_model("support_mail_classifier.pkl")
        return classifier

# Function to get sample data for visualization
def get_sample_data():
    categories = list(knowledge_base.keys())
    
    # Daily distribution data
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_counts = {}
    np.random.seed(42)
    for category in categories:
        daily_counts[category] = np.random.randint(5, 30, size=7).tolist()
    
    # Response time data
    response_times = {}
    for category in categories:
        response_times[category] = np.random.normal(4, 2, 30).tolist()  # hours
    
    # Category distribution
    category_distribution = {}
    for category in categories:
        category_distribution[category] = np.random.randint(50, 200)
    
    # Resolution rate
    resolution_rates = {}
    for category in categories:
        resolution_rates[category] = np.random.uniform(0.7, 0.95)
    
    return {
        "daily_counts": daily_counts,
        "response_times": response_times,
        "category_distribution": category_distribution,
        "resolution_rates": resolution_rates
    }

# Get the classifier model
model = load_model()

# Get sample visualization data
viz_data = get_sample_data()

# Sidebar
with st.sidebar:
    st.image("https://www.example.com/logo.png", width=50)  # Replace with your actual logo
    st.title("Support Mail Classifier")
    
    # Dark mode toggle
    st.write("## Theme Settings")
    if st.toggle("Dark Mode", value=st.session_state.dark_mode):
        st.session_state.dark_mode = True
        apply_custom_css()
        st.rerun()
    else:
        st.session_state.dark_mode = False
        apply_custom_css()
        
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigate to:",
        ["Dashboard", "Email Classifier", "Knowledge Base", "Analytics", "Settings"]
    )
    
    st.markdown("---")
    
    # Sample statistics
    st.write("## Quick Stats")
    st.write("📊 Emails today: 128")
    st.write("⚡ Auto-resolved: 76")
    st.write("⏱️ Avg. response time: 3.5h")
    
    st.markdown("---")
    st.write("Made with ❤️ by Your Team")

# Main content
if page == "Dashboard":
    st.title("📊 Support Dashboard")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Total Emails</div>
            <div class="metric-value">547</div>
            <div class="card-subtitle">Last 7 days</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Auto-Resolved</div>
            <div class="metric-value">324</div>
            <div class="card-subtitle">59.2% of total</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Avg. Response Time</div>
            <div class="metric-value">3.2h</div>
            <div class="card-subtitle">↓ 0.8h from last week</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Categories</div>
            <div class="metric-value">10</div>
            <div class="card-subtitle">Active knowledge bases</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Daily Email Volume")
        
        # Create data for the chart
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        total_by_day = [0] * 7
        
        for category, counts in viz_data["daily_counts"].items():
            for i in range(7):
                total_by_day[i] += counts[i]
        
        fig = px.bar(
            x=days, 
            y=total_by_day,
            labels={"x": "Day", "y": "Number of Emails"},
            color_discrete_sequence=["#4CAF50"]
        )
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Category Distribution")
        
        # Create data for the chart
        categories = list(viz_data["category_distribution"].keys())
        counts = list(viz_data["category_distribution"].values())
        
        fig = px.pie(
            values=counts, 
            names=categories,
            hole=0.4,
        )
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent emails
    st.subheader("Recent Support Emails")
    
    sample_emails = [
        {
            "id": "E-12345",
            "subject": "Cannot login to account",
            "received": "2025-03-03 09:23:45",
            "category": "login_issues",
            "status": "Auto-Resolved"
        },
        {
            "id": "E-12346",
            "subject": "Need to reset my password",
            "received": "2025-03-03 08:15:22",
            "category": "password_reset",
            "status": "Auto-Resolved"
        },
        {
            "id": "E-12347",
            "subject": "Application is extremely slow",
            "received": "2025-03-03 07:54:11",
            "category": "performance",
            "status": "Pending"
        },
        {
            "id": "E-12348",
            "subject": "Error when uploading files",
            "received": "2025-03-02 16:42:05",
            "category": "error",
            "status": "Forwarded"
        },
        {
            "id": "E-12349",
            "subject": "When is the new version coming out?",
            "received": "2025-03-02 14:30:19",
            "category": "release_info",
            "status": "Auto-Resolved"
        }
    ]
    
    df = pd.DataFrame(sample_emails)
    st.dataframe(df, use_container_width=True)

elif page == "Email Classifier":
    st.title("📧 Email Classifier")
    
    # Email input form
    st.write("### Enter Email Content")
    email_text = st.text_area("Paste email content here:", height=200)
    
    # Email metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        subject = st.text_input("Subject:", "")
    with col2:
        sender = st.text_input("From:", "")
    with col3:
        timestamp = st.text_input("Received:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    analyze_btn = st.button("Analyze Email", type="primary")
    
    if analyze_btn and email_text:
        with st.spinner("Analyzing email content..."):
            # Predict category
            prediction = model.predict(email_text)
            category = prediction["category"]
            confidence = prediction["confidence"]
            
            # Display results
            st.success(f"Analysis complete! Identified category: **{category}**")
            
            st.write("### Prediction Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Category:** {category}")
                st.write(f"**Confidence:** {confidence:.2%}")
                st.write("**Resolution:** Auto-resolution available")
                
                # Knowledge base info
                if category in knowledge_base:
                    kb = knowledge_base[category]
                    st.write("**Standard Solution:**")
                    st.write(kb["solution"])
                    st.write("**Knowledge Base Links:**")
                    for url in kb["urls"]:
                        st.write(f"- [{url.split('/')[-1].replace('-', ' ').title()}]({url})")
            
            with col2:
                # Confidence visualization
                probabilities = prediction["all_probabilities"]
                categories = list(probabilities.keys())
                values = list(probabilities.values())
                
                # Sort by probability
                sorted_indices = np.argsort(values)[::-1]
                sorted_categories = [categories[i] for i in sorted_indices]
                sorted_values = [values[i] for i in sorted_indices]
                
                # Take top 5
                top_n = 5
                fig = px.bar(
                    x=[f"{cat} ({val:.2%})" for cat, val in zip(sorted_categories[:top_n], sorted_values[:top_n])], 
                    y=sorted_values[:top_n],
                    labels={"x": "Category", "y": "Confidence"},
                    color=sorted_values[:top_n],
                    color_continuous_scale=["#CCFFCC", "#00CC00"]
                )
                fig.update_layout(
                    xaxis_title="", 
                    yaxis_title="Confidence",
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display suggested response
            st.write("### Suggested Response")
            if category in response_templates:
                response = response_templates[category]
                st.text_area("Auto-generated response:", response, height=200)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.button("Send Response")
                with col2:
                    st.button("Edit & Send")
                with col3:
                    st.button("Forward to Agent")
            else:
                st.write("No response template available for this category.")

elif page == "Knowledge Base":
    st.title("📚 Knowledge Base")
    
    # Search box
    search = st.text_input("Search knowledge base:")
    
    # Categories
    st.write("### Categories")
    category_cols = st.columns(5)
    
    categories = list(knowledge_base.keys())
    for i, cat in enumerate(categories):
        col_idx = i % 5
        with category_cols[col_idx]:
            # Format the category name to look nicer
            formatted_name = cat.replace("_", " ").title()
            if st.button(f"{formatted_name} ({viz_data['category_distribution'][cat]})", use_container_width=True):
                st.session_state.selected_category = cat
    
    # Display selected category
    st.write("### Solutions")
    
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = "login_issues"
        
    selected = st.session_state.selected_category
    
    if selected in knowledge_base:
        kb = knowledge_base[selected]
        formatted_name = selected.replace("_", " ").title()
        
        st.markdown(f"""
        <div class="dashboard-container">
            <h3>{formatted_name}</h3>
            <p><strong>Solution:</strong> {kb["solution"]}</p>
            <p><strong>Resolution Rate:</strong> {viz_data["resolution_rates"][selected]:.2%}</p>
            <p><strong>Knowledge Base Links:</strong></p>
            <ul>
                {"".join([f'<li><a href="{url}">{url.split("/")[-1].replace("-", " ").title()}</a></li>' for url in kb["urls"]])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample emails in this category
        st.write("#### Sample Emails in this Category")
        
        sample_emails = [
            f"I've been trying to {selected.replace('_', ' ')} but it's not working properly.",
            f"Could you please help me with {selected.replace('_', ' ')}? I've tried everything.",
            f"Is there a guide for {selected.replace('_', ' ')}? I'm completely stuck.",
            f"I need urgent assistance with {selected.replace('_', ' ')}. It's affecting my work."
        ]
        
        for email in sample_emails:
            st.markdown(f"""
            <div style="padding: 10px; border-left: 3px solid #4CAF50; margin-bottom: 10px;">
                <p>{email}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Standard response
        st.
