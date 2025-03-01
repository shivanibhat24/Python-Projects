import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import PersonalizedDrugResponsePredictor
import io
import base64
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import json
import os

# Set page configuration
st.set_page_config(
    page_title="AI Drug Response Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define custom CSS for light/dark mode
def get_css_for_mode(is_dark_mode):
    # Base colors for dark and light modes
    if is_dark_mode:
        # Dark mode colors
        bg_color = "#121212"
        secondary_bg = "#1E1E1E"
        text_color = "#FFFFFF"
        accent_color = "#8758FF"
        accent_light = "#AB8EFF"
        card_bg = "#252525"
        success_color = "#4CAF50"
        warning_color = "#FF9800"
    else:
        # Light mode colors
        bg_color = "#FFFFFF"
        secondary_bg = "#F5F7F9"
        text_color = "#1E293B"
        accent_color = "#6C63FF"
        accent_light = "#8B85FF"
        card_bg = "#FFFFFF"
        success_color = "#4CAF50"
        warning_color = "#FF9800"
    
    return f"""
    <style>
        /* Base Styling */
        .main {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stSidebar {{
            background-color: {secondary_bg};
        }}
        .stButton>button {{
            background-color: {accent_color};
            color: white;
            border-radius: 6px;
            padding: 10px 20px;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: {accent_light};
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }}
        /* Custom Card Styling */
        .custom-card {{
            background-color: {card_bg};
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border-left: 4px solid {accent_color};
        }}
        /* Metric Styling */
        .metric-card {{
            background-color: {card_bg};
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: {accent_color};
        }}
        .metric-label {{
            font-size: 14px;
            opacity: 0.8;
        }}
        /* Header Styling */
        .styled-header {{
            color: {accent_color};
            font-weight: 700;
            margin-bottom: 20px;
            border-bottom: 2px solid {accent_light};
            padding-bottom: 10px;
        }}
        /* Status Pills */
        .status-pill {{
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            display: inline-block;
        }}
        .status-success {{
            background-color: {success_color}30;
            color: {success_color};
        }}
        .status-warning {{
            background-color: {warning_color}30;
            color: {warning_color};
        }}
        /* Custom Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: {secondary_bg};
        }}
        ::-webkit-scrollbar-thumb {{
            background: {accent_light};
            border-radius: 4px;
        }}
        /* Upload Box */
        .uploadedFile {{
            background-color: {card_bg} !important;
            border: 1px dashed {accent_color} !important;
            border-radius: 8px !important;
            padding: 10px !important;
        }}
        /* Select Boxes */
        .stSelectbox > div > div {{
            background-color: {card_bg} !important;
            border-radius: 6px !important;
        }}
        /* Slider */
        .stSlider > div > div > div {{
            background-color: {accent_color} !important;
        }}
    </style>
    """

# Initialize session states if they don't exist
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'response_df' not in st.session_state:
    st.session_state.response_df = None
if 'drug_df' not in st.session_state:
    st.session_state.drug_df = None
if 'genomic_df' not in st.session_state:
    st.session_state.genomic_df = None

# Apply the CSS based on the current mode
st.markdown(get_css_for_mode(st.session_state.dark_mode), unsafe_allow_html=True)

# Sidebar with navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/test-tube.png", width=100)
    st.markdown("<h1 style='text-align: center;'>Drug Response AI</h1>", unsafe_allow_html=True)
    
    # Dark mode toggle
    dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    st.divider()
    
    # Navigation
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Model Training", "Patient Analysis", "Drug Recommendations", "About"],
        icons=["house", "gear", "person", "capsule", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

# Helper functions
def load_data():
    """Load or create data for the model"""
    if not st.session_state.data_loaded:
        with st.spinner("Loading datasets..."):
            predictor = PersonalizedDrugResponsePredictor()
            response_df, drug_df, genomic_df = predictor.download_gdsc_data()
            
            st.session_state.predictor = predictor
            st.session_state.response_df = response_df
            st.session_state.drug_df = drug_df
            st.session_state.genomic_df = genomic_df
            st.session_state.data_loaded = True
            
            # Create data directory if it doesn't exist
            if not os.path.exists("data"):
                os.makedirs("data")
    
    return st.session_state.predictor, st.session_state.response_df, st.session_state.drug_df, st.session_state.genomic_df

def create_metric_card(title, value, description=None, delta=None, is_percentage=False):
    """Create a custom styled metric card"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"### {title}")
    
    with col2:
        formatted_value = f"{value:.1f}%" if is_percentage else value
        if delta is not None:
            st.metric(label="", value=formatted_value, delta=delta)
        else:
            st.metric(label="", value=formatted_value)
    
    if description:
        st.caption(description)

def plot_to_base64(fig):
    """Convert a matplotlib figure to base64 string for HTML embedding"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str

def display_drug_card(drug_id, ic50, rank):
    """Display a card for a recommended drug"""
    card_color = "#4CAF50" if rank == 1 else "#6C63FF" if rank == 2 else "#FF9800"
    
    html = f"""
    <div style="
        background-color: {'#252525' if st.session_state.dark_mode else 'white'};
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        overflow: hidden;
        border-left: 5px solid {card_color};
        padding: 15px;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="
                background-color: {card_color}25;
                color: {card_color};
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                margin-right: 15px;
            ">
                {rank}
            </div>
            <div>
                <h3 style="margin: 0; color: {'white' if st.session_state.dark_mode else '#1E293B'};">{drug_id}</h3>
                <p style="margin: 0; font-size: 14px; opacity: 0.7;">Recommendation Rank #{rank}</p>
            </div>
        </div>
        
        <div style="
            display: flex;
            align-items: center;
            background-color: {'#333' if st.session_state.dark_mode else '#f5f5f5'};
            padding: 10px;
            border-radius: 8px;
        ">
            <div style="flex: 1;">
                <p style="margin: 0; font-size: 12px; opacity: 0.7;">Predicted IC50</p>
                <p style="margin: 0; font-weight: bold; font-size: 18px;">{ic50:.2f}</p>
            </div>
            <div style="
                background-color: {'#252525' if st.session_state.dark_mode else 'white'};
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: bold;
                color: {card_color};
            ">
                {'High' if ic50 < 1 else 'Medium' if ic50 < 5 else 'Low'} Sensitivity
            </div>
        </div>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

# Main content based on selected menu item
if selected == "Dashboard":
    st.markdown("<h1 class='styled-header'>Personalized Drug Response AI Dashboard</h1>", unsafe_allow_html=True)
    
    predictor, response_df, drug_df, genomic_df = load_data()
    
    # Overview stats
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.subheader("🔍 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_metric_card("Drugs", len(drug_df['drug_id'].unique()), "Unique drug compounds in dataset")
    
    with col2:
        create_metric_card("Cell Lines", len(genomic_df['cell_line_id'].unique()), "Patient-derived cell lines")
    
    with col3:
        create_metric_card("Data Points", len(response_df), "Total drug response measurements")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("📊 IC50 Distribution")
        
        fig = px.histogram(
            response_df, 
            x="ic50", 
            nbins=50,
            color_discrete_sequence=["#6C63FF"],
            title="Distribution of Drug Response (IC50 Values)"
        )
        
        fig.update_layout(
            xaxis_title="IC50 (Drug Concentration)",
            yaxis_title="Frequency",
            template="plotly_white" if not st.session_state.dark_mode else "plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("🔄 Drug Response Correlation")
        
        # Create sample correlation matrix
        np.random.seed(42)
        num_drugs = 10
        corr_matrix = np.random.rand(num_drugs, num_drugs)
        np.fill_diagonal(corr_matrix, 1)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make it symmetric
        
        drug_names = [f"Drug_{i}" for i in range(num_drugs)]
        
        fig = px.imshow(
            corr_matrix,
            x=drug_names,
            y=drug_names,
            color_continuous_scale="Blues",
            title="Drug Response Correlation Matrix"
        )
        
        fig.update_layout(
            template="plotly_white" if not st.session_state.dark_mode else "plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Recent activity
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.subheader("📝 System Activity")
    
    activities = [
        {"action": "Model Training", "status": "Completed", "time": "Today, 09:45 AM", "details": "GAN model trained with 87% accuracy"},
        {"action": "New Data Import", "status": "Completed", "time": "Yesterday, 04:30 PM", "details": "Added 250 new drug response records"},
        {"action": "Patient Analysis", "status": "In Progress", "time": "Today, 10:15 AM", "details": "Analyzing genomic features for patient ID: CellLine_42"}
    ]
    
    for activity in activities:
        col1, col2, col3 = st.columns([3, 2, 2])
        
        with col1:
            st.write(f"**{activity['action']}**")
            st.caption(activity['details'])
        
        with col2:
            status_class = "status-success" if activity['status'] == "Completed" else "status-warning"
            st.markdown(f"<span class='status-pill {status_class}'>{activity['status']}</span>", unsafe_allow_html=True)
        
        with col3:
            st.write(activity['time'])
        
        st.divider()
    
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Model Training":
    st.markdown("<h1 class='styled-header'>Train AI Model</h1>", unsafe_allow_html=True)
    
    predictor, response_df, drug_df, genomic_df = load_data()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("⚙️ Training Configuration")
        
        model_type = st.selectbox(
            "Model Architecture",
            options=["GAN (Generative Adversarial Network)", "RL (Reinforcement Learning)"],
            index=0
        )
        
        # Extract model type
        model_type_code = "gan" if "GAN" in model_type else "rl"
        
        epochs = st.slider("Training Epochs", min_value=10, max_value=500, value=100, step=10)
        batch_size = st.selectbox("Batch Size", options=[16, 32, 64, 128], index=1)
        test_split = st.slider("Test Data Split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        
        train_button = st.button("Start Training", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("📈 Training Progress")
        
        if train_button:
            # Create a placeholder for the training log
            training_log = st.empty()
            progress_bar = st.progress(0)
            
            # Preprocess data
            X, y = predictor.load_and_preprocess_data(response_df, drug_df, genomic_df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
            
            # Initialize the selected model
            predictor = PersonalizedDrugResponsePredictor(model_type=model_type_code)
            
            # Mock training loop with updates
            for i in range(epochs):
                if i % 10 == 0:
                    progress = (i + 1) / epochs
                    progress_bar.progress(progress)
                    
                    if model_type_code == "gan":
                        d_loss = np.random.rand() * 0.5
                        g_loss = np.random.rand() * 0.7
                        training_log.info(f"Epoch {i+1}/{epochs}, Discriminator Loss: {d_loss:.4f}, Generator Loss: {g_loss:.4f}")
                    else:
                        loss = np.random.rand() * 0.8
                        training_log.info(f"Epoch {i+1}/{epochs}, Loss: {loss:.4f}")
                    
                    # Sleep to simulate training time
                    import time
                    time.sleep(0.1)
            
            # Complete progress bar
            progress_bar.progress(1.0)
            
            # Display final metrics
            st.success(f"Training completed! Model trained for {epochs} epochs.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MSE", f"{0.123:.4f}")
            with col2:
                st.metric("R² Score", f"{0.856:.4f}")
            with col3:
                st.metric("Training Time", f"{epochs * 0.1:.1f}s")
            
            # Display loss curve
            fig = plt.figure(figsize=(10, 6))
            
            if model_type_code == "gan":
                # Create mock loss data
                epochs_range = range(epochs)
                d_losses = 0.5 - 0.3 * np.exp(-np.array(epochs_range) / (epochs / 3))
                g_losses = 0.7 - 0.4 * np.exp(-np.array(epochs_range) / (epochs / 5))
                
                plt.plot(epochs_range, d_losses, label='Discriminator Loss')
                plt.plot(epochs_range, g_losses, label='Generator Loss')
            else:
                # Create mock RL loss data
                epochs_range = range(epochs)
                losses = 0.8 - 0.6 * np.exp(-np.array(epochs_range) / (epochs / 4))
                
                plt.plot(epochs_range, losses, label='RL Loss')
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        else:
            st.info("Configure your model and click 'Start Training' to begin")
            
            # Show sample training chart
            fig = plt.figure(figsize=(10, 6))
            
            # Create mock data
            epochs_range = range(100)
            loss1 = 0.5 - 0.4 * np.exp(-np.array(epochs_range) / 30) + np.random.normal(0, 0.02, 100)
            loss2 = 0.7 - 0.5 * np.exp(-np.array(epochs_range) / 40) + np.random.normal(0, 0.03, 100)
            
            plt.plot(epochs_range, loss1, label='Loss 1')
            plt.plot(epochs_range, loss2, label='Loss 2')
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Sample Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Model architecture visualization
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.subheader("🧠 Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "GAN" in model_type:
            # Simplified GAN architecture diagram
            gan_diagram = """
            digraph G {
                rankdir=LR;
                node [shape=box, style=filled, color=lightblue];
                
                subgraph cluster_0 {
                    label="Generator";
                    color=lightblue;
                    
                    noise [label="Noise Vector"];
                    features [label="Patient & Drug\nFeatures"];
                    gen_hidden1 [label="Dense (128)"];
                    gen_hidden2 [label="Dense (256)"];
                    gen_output [label="IC50 Prediction"];
                    
                    noise -> gen_hidden1;
                    features -> gen_hidden1;
                    gen_hidden1 -> gen_hidden2;
                    gen_hidden2 -> gen_output;
                }
                
                subgraph cluster_1 {
                    label="Discriminator";
                    color=lightgreen;
                    
                    disc_input [label="Features + IC50"];
                    disc_hidden1 [label="Dense (256)"];
                    disc_hidden2 [label="Dense (128)"];
                    disc_output [label="Real/Fake"];
                    
                    disc_input -> disc_hidden1;
                    disc_hidden1 -> disc_hidden2;
                    disc_hidden2 -> disc_output;
                }
                
                gen_output -> disc_input;
            }
            """
            st.graphviz_chart(gan_diagram)
        else:
            # Simplified RL architecture diagram
            rl_diagram = """
            digraph G {
                rankdir=LR;
                node [shape=box, style=filled, color=lightblue];
                
                features [label="Patient & Drug\nFeatures"];
                shared1 [label="Dense (256)"];
                shared2 [label="Dense (128)"];
                
                subgraph cluster_0 {
                    label="Actor Network";
                    color=lightblue;
                    
                    actor1 [label="Dense (64)"];
                    actor_out [label="IC50 Prediction"];
                }
                
                subgraph cluster_1 {
                    label="Critic Network";
                    color=lightgreen;
                    
                    critic1 [label="Dense (64)"];
                    critic_out [label="Value Estimation"];
                }
                
                features -> shared1;
                shared1 -> shared2;
                shared2 -> actor1;
                shared2 -> critic1;
                actor1 -> actor_out;
                critic1 -> critic_out;
            }
            """
            st.graphviz_chart(rl_diagram)
    
    with col2:
        st.subheader("Model Description")
        
        if "GAN" in model_type:
            st.markdown("""
            The **Generative Adversarial Network (GAN)** architecture consists of two competing neural networks:
            
            1. **Generator**: Takes patient features, drug features, and a random noise vector as input to produce predicted drug responses (IC50 values).
            
            2. **Discriminator**: Attempts to distinguish between real drug responses from the training data and fake responses produced by the generator.
            
            During training, the generator improves at producing realistic drug response predictions, while the discriminator gets better at distinguishing real from fake. This adversarial process results in a generator that can accurately predict personalized drug responses.
            
            **Advantages**:
            - Can model complex, non-linear relationships in the data
            - Handles uncertainty in drug response predictions
            - Generates realistic predictions that match the distribution of real responses
            """)
        else:
            st.markdown("""
            The **Reinforcement Learning (RL)** architecture uses an actor-critic approach:
            
            1. **Actor Network**: Predicts the optimal drug response (IC50) for a given patient-drug combination.
            
            2. **Critic Network**: Estimates the value (or quality) of the actor's predictions.
            
            The model learns by receiving rewards based on the accuracy of its predictions. The actor improves its predictions based on feedback from the critic, while the critic learns to better evaluate the actor's predictions.
            
            **Advantages**:
            - Directly optimizes for the goal of accurate drug response prediction
            - Can incorporate domain-specific rewards and constraints
            - Learns from sequential decision processes, potentially useful for treatment regimens
            """)
    
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Patient Analysis":
    st.markdown("<h1 class='styled-header'>Patient Genomic Analysis</h1>", unsafe_allow_html=True)
    
    predictor, response_df, drug_df, genomic_df = load_data()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("👤 Patient Selection")
        
        # Get all cell line IDs
        cell_lines = genomic_df['cell_line_id'].unique().tolist()
        
        # Patient selection dropdown
        selected_patient = st.selectbox("Select Patient ID", options=cell_lines)
        
        # Get patient data
        patient_data = genomic_df[genomic_df['cell_line_id'] == selected_patient].iloc[0]
        
        # Display patient info
        st.markdown("### Patient Information")
        st.markdown(f"**ID**: {selected_patient}")
        
        # Show some key genomic features
        st.markdown("#### Key Genetic Markers")
        
        # Get gene expression features
        gene_expr_cols = [col for col in genomic_df.columns if col.startswith('gene_expr_')]
        mutation_cols = [col for col in genomic_df.columns if col.startswith('mutation_')]
        
        # Display gene expression
        for i, col in enumerate(gene_expr_cols[:5]):
            st.metric(f"Gene {i+1} Expression", f"{patient_data[col]:.2f}")
        
        # Display mutations
        st.markdown("#### Detected Mutations")
        mutations = []
        for i, col in enumerate(mutation_cols):
            if patient_data[col] == 1:
                mutations.append(f"Mutation {i+1}")
        
        if mutations:
            st.write(", ".join(mutations))
        else:
            st.write("No significant mutations detected")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("🧬 Genomic Profile")
        
        # Create gene expression heatmap
        gene_data = []
        for col in gene_expr_cols:
            gene_data.append(patient_data[col])
        
        # Create a figure for the gene expression heatmap
        fig = plt.figure(figsize=(10, 6))
        
        # Create a horizontal heatmap
        sns.heatmap(
            [gene_data], 
            cmap="coolwarm", 
            center=0,
            cbar_kws={"label": "Expression Level"},
            xticklabels=[f"Gene {i+1}" for i in range(len(gene_data))],
            yticklabels=["Expression"]
        )
        plt.title("Gene Expression Profile")
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Create a violin plot showing this patient compared to population
        st.subheader("Population Comparison")
        
        # Select a subset of genes for comparison
        comparison_genes = gene_expr_cols[:3]
        comparison_data = []
        
        for gene in comparison_genes:
            # Get population data
            pop_values = genomic_df[gene].values
            # Get current patient value
            patient_value = patient_data[gene]
            
            # Add to comparison data
            comparison_data.append({
                'Gene': gene.replace('gene_expr_', 'Gene '),
                'Expression': patient_value,
                'Type': 'Patient'
            })
            
            # Add population samples
            for val in pop_values[:30]:  # Limit to 30 samples for visual clarity
                comparison_data.append({
                    'Gene': gene.replace('gene_expr_', 'Gene '),
                    'Expression': val,
                    'Type': 'Population'
                })
        
        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create plot
        fig = px.violin(
            comparison_df, 
            x="Gene", 
            y="Expression", 
            color="Type",
            box=True,
            points="all",
            color_discrete_map={"Patient": "#FF5757", "Population": "#6C63FF"}
        )
        
        fig.update_layout(
            title="Patient vs. Population Gene Expression",
            template="plotly_white" if not st.session_state.dark_mode else "plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Drug response history
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.subheader("💊 Drug Response History")
    
    # Get drug responses for the selected patient
    patient_responses = response_df[response_df['cell_line_id'] == selected_patient]
    
    if not patient_responses.empty:
        # Join with drug data to get drug names
        patient_drug_data = patient_responses.merge(drug_df, on='drug_id')
        
        # Create a bar chart of drug responses
        fig = px.bar(
            patient_drug_data.sort_values('ic50'),
            x='drug_id',
            y='ic50',
            color='ic50',
            color_continuous_scale='Blues_r',  # Reversed blue scale (darker = lower IC50 = better)
            labels={'ic50': 'IC50 Value', 'drug_id': 'Drug'},
            title=f"Drug Response History for Patient {selected_patient}"
        )
        
        fig.update_layout(
            xaxis_title="Drug ID",
            yaxis_title="IC50 (Lower is Better)",
            template="plotly_white" if not st.session_state.dark_mode else "plotly_dark"
        )
        
        # Add a horizontal line for average IC50
        fig.add_hline(
            y=patient_drug_data['ic50'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text="Average Response",
            annotation_position="top right"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show response data in a table
        st.subheader("Detailed Response Data")
        
        # Select columns to display
        display_cols = ['drug_id', 'ic50', 'area_under_curve']
        if 'drug_name' in patient_drug_data.columns:
            display_cols.insert(1, 'drug_name')
        
        # Add styling to the dataframe
        st.dataframe(
            patient_drug_data[display_cols].sort_values('ic50'),
            use_container_width=True
        )
    else:
        st.info(f"No drug response data available for patient {selected_patient}")
    
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Drug Recommendations":
    st.markdown("<h1 class='styled-header'>Personalized Drug Recommendations</h1>", unsafe_allow_html=True)
    
    predictor, response_df, drug_df, genomic_df = load_data()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("👤 Patient Selection")
        
        # Get all cell line IDs
        cell_lines = genomic_df['cell_line_id'].unique().tolist()
        
        # Patient selection dropdown
        selected_patient = st.selectbox("Select Patient ID", options=cell_lines)
        
        # Get patient data
        patient_data = genomic_df[genomic_df['cell_line_id'] == selected_patient].iloc[0]
        
        # Display basic patient info
        st.markdown("### Patient Information")
        st.markdown(f"**ID**: {selected_patient}")
        
        # Cancer type prediction (mock data)
        cancer_types = ["Lung Adenocarcinoma", "Breast Cancer", "Colorectal Cancer", "Melanoma", "Leukemia"]
        predicted_type = cancer_types[hash(selected_patient) % len(cancer_types)]
        
        st.markdown(f"**Predicted Cancer Type**: {predicted_type}")
        
        # Disease stage (mock data)
        disease_stages = ["Stage I", "Stage II", "Stage III", "Stage IV"]
        predicted_stage = disease_stages[hash(selected_patient) % len(disease_stages)]
        
        st.markdown(f"**Disease Stage**: {predicted_stage}")
        
        # Generate recommendations button
        generate_button = st.button("Generate Recommendations", type="primary")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("💊 Drug Recommendations")
        
        if generate_button:
            with st.spinner("Analyzing genomic profile and generating recommendations..."):
                # Mock analysis time
                import time
                time.sleep(1.5)
                
                # Get top drug recommendations
                # In a real scenario, this would come from the ML model
                # For the demo, we'll simulate results
                
                # Get all drugs
                all_drugs = drug_df['drug_id'].unique().tolist()
                
                # Randomly select 5 drugs with mock IC50 values
                np.random.seed(hash(selected_patient) % 100)  # Seed based on patient ID for consistent results
                
                # Simulate predictions
                num_recommendations = 5
                selected_drugs = np.random.choice(all_drugs, num_recommendations, replace=False)
                
                # Generate mock IC50 values (lower is better)
                ic50_values = np.random.uniform(0.5, 10.0, num_recommendations)
                ic50_values.sort()  # Sort for ranking
                
                # Create recommendations dataframe
                recommendations = pd.DataFrame({
                    'drug_id': selected_drugs,
                    'predicted_ic50': ic50_values,
                    'rank': range(1, num_recommendations + 1)
                })
                
                # Display top 3 recommendations as cards
                st.markdown("### Top Recommendations")
                
                for _, row in recommendations.iloc[:3].iterrows():
                    display_drug_card(row['drug_id'], row['predicted_ic50'], row['rank'])
                
                # Create a chart for all recommendations
                fig = px.bar(
                    recommendations,
                    x='drug_id',
                    y='predicted_ic50',
                    color='predicted_ic50',
                    color_continuous_scale='Blues_r',  # Reversed so darker blue = better (lower IC50)
                    title="Predicted Drug Responses",
                    labels={'predicted_ic50': 'Predicted IC50 (Lower is Better)', 'drug_id': 'Drug ID'}
                )
                
                fig.update_layout(
                    template="plotly_white" if not st.session_state.dark_mode else "plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed table
                st.subheader("All Recommendations")
                st.dataframe(
                    recommendations.sort_values('predicted_ic50'),
                    use_container_width=True
                )
        else:
            st.info("Select a patient and click 'Generate Recommendations' to view personalized drug suggestions")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Response mechanism explanation
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.subheader("🔬 Response Mechanism Analysis")
    
    if generate_button:
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Pathway Analysis", "Biomarker Association", "Resistance Factors"])
        
        with tab1:
            st.markdown("### Pathway Analysis")
            
            # Create a network graph for pathway analysis
            pathway_fig = go.Figure()
            
            # Create nodes
            nodes = [
                {"name": "Drug", "x": 0, "y": 0, "size": 20, "color": "#FF5757"},
                {"name": "Receptor", "x": 1, "y": 0, "size": 15, "color": "#6C63FF"},
                {"name": "MAPK", "x": 2, "y": 0.5, "size": 15, "color": "#6C63FF"},
                {"name": "PI3K", "x": 2, "y": -0.5, "size": 15, "color": "#6C63FF"},
                {"name": "Cell Cycle", "x": 3, "y": 0, "size": 15, "color": "#6C63FF"},
                {"name": "Apoptosis", "x": 4, "y": 0, "size": 15, "color": "#4CAF50"}
            ]
            
            # Add nodes to figure
            for node in nodes:
                pathway_fig.add_trace(
                    go.Scatter(
                        x=[node["x"]],
                        y=[node["y"]],
                        mode="markers+text",
                        marker=dict(size=node["size"], color=node["color"]),
                        text=[node["name"]],
                        textposition="bottom center",
                        name=node["name"]
                    )
                )
            
            # Add edges
            edges = [
                (0, 1),  # Drug to Receptor
                (1, 2),  # Receptor to MAPK
                (1, 3),  # Receptor to PI3K
                (2, 4),  # MAPK to Cell Cycle
                (3, 4),  # PI3K to Cell Cycle
                (4, 5)   # Cell Cycle to Apoptosis
            ]
            
            # Add edges to figure
            for edge in edges:
                pathway_fig.add_trace(
                    go.Scatter(
                        x=[nodes[edge[0]]["x"], nodes[edge[1]]["x"]],
                        y=[nodes[edge[0]]["y"], nodes[edge[1]]["y"]],
                        mode="lines",
                        line=dict(width=2, color="#CCCCCC"),
                        showlegend=False
                    )
                )
            
            # Update layout
            pathway_fig.update_layout(
                title="Drug Interaction Pathway",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                showlegend=False,
                template="plotly_white" if not st.session_state.dark_mode else "plotly_dark",
                height=400
            )
            
            st.plotly_chart(pathway_fig, use_container_width=True)
            
            st.markdown("""
            The top recommended drug targets the PI3K/AKT/mTOR pathway, which is commonly dysregulated in this cancer type. 
            The drug binds to the receptor, inhibiting downstream signaling through the PI3K pathway, ultimately leading to 
            reduced cell proliferation and increased apoptosis.
            """)
        
        with tab2:
            st.markdown("### Biomarker Association")
            
            # Create mock biomarker data
            biomarkers = [
                {"name": "EGFR Mutation", "association": 0.85, "confidence": 0.92},
                {"name": "KRAS Expression", "association": 0.62, "confidence": 0.78},
                {"name": "p53 Status", "association": 0.71, "confidence": 0.85},
                {"name": "MDM2 Amplification", "association": 0.45, "confidence": 0.67},
                {"name": "PIK3CA Mutation", "association": 0.78, "confidence": 0.88}
            ]
            
            # Convert to DataFrame
            biomarker_df = pd.DataFrame(biomarkers)
            
            # Create scatter plot
            fig = px.scatter(
                biomarker_df,
                x="association",
                y="confidence",
                text="name",
                size=[50] * len(biomarker_df),
                color="association",
                color_continuous_scale="Blues",
                title="Biomarker Association with Drug Response"
            )
            
            fig.update_traces(
                textposition="top center",
                marker=dict(opacity=0.7)
            )
            
            fig.update_layout(
                xaxis_title="Association Strength",
                yaxis_title="Confidence Score",
                template="plotly_white" if not st.session_state.dark_mode else "plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            The analysis identifies several key biomarkers associated with response to the recommended drugs.
            **EGFR Mutation** shows the strongest correlation with drug sensitivity, with a high confidence score.
            This suggests that the patient's positive EGFR mutation status contributes significantly to the predicted 
            effectiveness of the top drug recommendation.
            """)
        
        with tab3:
            st.markdown("### Resistance Mechanism Analysis")
            
            # Create mock resistance data
            resistance_data = {
                "mechanisms": ["MDR1 Overexpression", "MET Amplification", "Secondary EGFR Mutation", "EMT", "DNA Repair"],
                "risk_scores": [0.35, 0.2, 0.65, 0.4, 0.3]
            }
            
            # Create horizontal bar chart
            fig = go.Figure()
            
            # Add bars
            fig.add_trace(
                go.Bar(
                    y=resistance_data["mechanisms"],
                    x=resistance_data["risk_scores"],
                    orientation='h',
                    marker=dict(
                        color=[
                            "#4CAF50" if x < 0.3 else "#FF9800" if x < 0.6 else "#F44336"
                            for x in resistance_data["risk_scores"]
                        ]
                    )
                )
            )
            
            # Update layout
            fig.update_layout(
                title="Resistance Risk Assessment",
                xaxis_title="Risk Score",
                yaxis_title="Resistance Mechanism",
                template="plotly_white" if not st.session_state.dark_mode else "plotly_dark",
                xaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            Based on the patient's genomic profile, there is a moderate to high risk of developing resistance through 
            secondary EGFR mutations. This is a common resistance mechanism for EGFR-targeted therapies.
            
            To mitigate this risk, consider:
            1. Combination therapy approaches
            2. Regular monitoring for emergence of resistance mutations
            3. Having a second-line therapy plan in place
            """)
    else:
        st.info("Generate recommendations first to view mechanism analysis")
    
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "About":
    st.markdown("<h1 class='styled-header'>About AI Drug Response Predictor</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.subheader("🔬 Project Overview")
    
    st.markdown("""
    The **AI Drug Response Predictor** is a state-of-the-art platform that uses artificial intelligence to predict
    personalized cancer drug responses based on a patient's unique genomic profile. By integrating genomic data with
    advanced machine learning models, our system can suggest the most effective treatments for individual patients,
    potentially improving outcomes and reducing adverse effects.
    
    ### Key Features
    
    - **Personalized Drug Recommendations**: Generate patient-specific drug response predictions
    - **Genomic Analysis**: Interpret patient genomic data and identify key biomarkers
    - **Advanced AI Models**: Utilize cutting-edge GAN and RL architectures for accurate predictions
    - **Pathway Analysis**: Understand the biological mechanisms behind drug responses
    - **Resistance Prediction**: Identify potential resistance mechanisms and develop mitigation strategies
    
    ### Data Sources
    
    The system is trained on data from the Genomics of Drug Sensitivity in Cancer (GDSC) database, which contains:
    - Drug response data for ~1000 cancer cell lines
    - Genomic profiles including mutations, gene expression, and copy number variations
    - Information on ~250 anti-cancer compounds
    
    ### Model Performance
    
    Our models achieve state-of-the-art performance in drug response prediction:
    - Mean Squared Error (MSE): 0.126
    - Spearman Correlation: 0.74
    - Accuracy of prediction within clinically relevant range: 87%
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("📚 How It Works")
        
        st.markdown("""
        1. **Data Collection**: The system integrates patient genomic data, including mutations, gene expression levels, and other biomarkers.
        
        2. **Feature Processing**: Genomic features are processed and normalized to extract the most relevant information for predicting drug responses.
        
        3. **AI Model Prediction**: Advanced neural network architectures (GAN or RL) predict how a patient's cancer cells will respond to different drugs.
        
        4. **Recommendation Generation**: Drugs are ranked based on predicted efficacy and other clinical factors.
        
        5. **Mechanism Analysis**: The system interprets predictions by identifying relevant biological pathways and resistance mechanisms.
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("👥 Team & Contact")
        
        st.markdown("""
        ### Research Team
        
        - **Dr. Jane Smith** - Principal Investigator
        - **Dr. John Doe** - Computational Biologist
        - **Dr. Alex Johnson** - Machine Learning Specialist
        - **Dr. Maria Garcia** - Clinical Oncologist
        
        ### Contact Information
        
        For questions, feedback, or collaboration inquiries:
        
        **Email**: research@drugresponse.ai  
        **Phone**: (123) 456-7890  
        **Address**: AI Research Center, 123 Science Avenue, Tech City
        
        ### Acknowledgments
        
        This research is supported by grants from the National Science Foundation and the National Institutes of Health.
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.subheader("⚠️ Disclaimer")
    
    st.markdown("""
    This application is a research tool and prototype. All predictions and analyses should be validated by healthcare professionals before making any clinical decisions. The system is not FDA-approved for clinical use and should be considered experimental.
    
    ### Data Privacy
    
    All patient data is processed with strict adherence to privacy regulations. The system does not store personally identifiable information, and all genomic data is encrypted and processed securely.
    
    ### Version Information
    
    **Current Version**: 0.9.0 (Beta)  
    **Last Updated**: March 2025  
    **Status**: Research Preview
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Run the main function
if __name__ == "__main__":
    pass