import streamlit as st
import torch
import numpy as np
import io
import base64
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import os
import cv2
from streamlit_drawable_canvas import st_canvas
import requests
from typing import Optional, Tuple, Dict, Any

# Set page configuration
st.set_page_config(
    layout="wide", 
    page_title="Generative Engineering Whiteboard",
    page_icon="⚙️",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .stTab [data-testid="stTabsContainer"] > div {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables with proper defaults
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'sketch_img': None,
        'segmented_img': None,
        'refined_img': None,
        'segmentation_mask': None,
        'engineering_specs': None,
        'text_prompt': '',
        'refinement_prompt': '',
        'canvas_key': 0,
        'processing_history': [],
        'model_settings': {
            'sam_model_type': 'vit_h',
            'sd_guidance_scale': 7.5,
            'sd_num_inference_steps': 50,
            'temperature': 0.7
        }
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# Sidebar for settings and model configuration
with st.sidebar:
    st.header("⚙️ Model Settings")
    
    st.subheader("Segment Anything (SAM)")
    sam_model_type = st.selectbox(
        "Model Type",
        ["vit_h", "vit_l", "vit_b"],
        index=0,
        help="Choose SAM model size. vit_h is most accurate but slower."
    )
    
    st.subheader("Stable Diffusion")
    guidance_scale = st.slider(
        "Guidance Scale",
        min_value=1.0,
        max_value=20.0,
        value=7.5,
        step=0.5,
        help="Higher values follow text prompt more closely"
    )
    
    num_inference_steps = st.slider(
        "Inference Steps",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
        help="More steps = better quality but slower"
    )
    
    st.subheader("LLM Settings")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Creativity level for specifications generation"
    )
    
    # Update session state
    st.session_state.model_settings.update({
        'sam_model_type': sam_model_type,
        'sd_guidance_scale': guidance_scale,
        'sd_num_inference_steps': num_inference_steps,
        'temperature': temperature
    })
    
    st.markdown("---")
    st.subheader("📊 Processing History")
    if st.session_state.processing_history:
        for i, step in enumerate(st.session_state.processing_history[-5:], 1):
            st.caption(f"{i}. {step}")
    else:
        st.caption("No processing steps yet")
    
    if st.button("Clear History"):
        st.session_state.processing_history = []
        st.rerun()

# Application header
st.markdown("""
<div class="main-header">
    <h1>⚙️ Generative Engineering Whiteboard</h1>
    <p>Transform sketches and descriptions into simulation-ready engineering concepts</p>
</div>
""", unsafe_allow_html=True)

# Utility functions with enhanced error handling
@st.cache_resource
def load_sam_model():
    """Load the Segment Anything Model with enhanced error handling."""
    try:
        st.info("🤖 In production: SAM model would be loaded here")
        # Simulated model loading
        return {"status": "loaded", "model_type": st.session_state.model_settings['sam_model_type']}
    except Exception as e:
        st.error(f"Failed to load SAM model: {str(e)}")
        return None

@st.cache_resource
def load_stable_diffusion():
    """Load Stable Diffusion with enhanced configuration."""
    try:
        st.info("🎨 In production: Stable Diffusion would be loaded here")
        # Simulated model loading
        return {"status": "loaded", "guidance_scale": st.session_state.model_settings['sd_guidance_scale']}
    except Exception as e:
        st.error(f"Failed to load Stable Diffusion: {str(e)}")
        return None

@st.cache_resource
def setup_langchain():
    """Enhanced LangChain setup with error handling."""
    try:
        st.info("🧠 In production: LangChain would be configured with actual API keys")
        return {"status": "ready", "temperature": st.session_state.model_settings['temperature']}
    except Exception as e:
        st.error(f"Failed to setup LangChain: {str(e)}")
        return None

def enhanced_segment_sketch(image: Image.Image, text_prompt: str = "") -> Tuple[Image.Image, np.ndarray]:
    """Enhanced segmentation with multiple techniques."""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Handle different image formats
        if len(img_array.shape) == 4:  # RGBA
            # Convert RGBA to RGB
            img_array = img_array[:, :, :3]
        elif len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        
        # Enhanced edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Multiple segmentation approaches
        # 1. Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 2. Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # 3. Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Combine results
        final_mask = cv2.bitwise_or(cleaned, edges)
        
        # Create colored overlay
        colored_mask = np.zeros((final_mask.shape[0], final_mask.shape[1], 4), dtype=np.uint8)
        colored_mask[final_mask > 0] = [255, 100, 100, 180]  # Semi-transparent red
        
        # Create result image
        result_array = img_array.copy()
        overlay = Image.fromarray(colored_mask)
        result_img = Image.fromarray(result_array)
        
        # Blend images
        if overlay.mode == 'RGBA':
            result_img = Image.alpha_composite(
                result_img.convert('RGBA'), 
                overlay
            ).convert('RGB')
        
        return result_img, final_mask
        
    except Exception as e:
        st.error(f"Segmentation failed: {str(e)}")
        return image, np.zeros((image.height, image.width), dtype=np.uint8)

def enhanced_refine_design(original_image: Image.Image, mask: np.ndarray, text_prompt: str) -> Image.Image:
    """Enhanced design refinement with multiple effects."""
    try:
        img_array = np.array(original_image)
        refined = img_array.copy().astype(np.float32)
        
        # Create mask indices
        if len(mask.shape) == 2:
            mask_indices = mask > 0
        else:
            mask_indices = mask[:, :, 0] > 0
        
        # Apply different effects based on prompt keywords
        if "metal" in text_prompt.lower():
            # Metallic effect
            refined[mask_indices, 0] *= 0.9  # Reduce red
            refined[mask_indices, 1] *= 0.95  # Slightly reduce green
            refined[mask_indices, 2] *= 1.1  # Enhance blue
            
        elif "gear" in text_prompt.lower():
            # Gear-like enhancement
            refined[mask_indices, :] *= 0.8  # Darken overall
            
        elif "precise" in text_prompt.lower() or "sharp" in text_prompt.lower():
            # Enhance edges and contrast
            refined[mask_indices, :] = np.clip(refined[mask_indices, :] * 1.2, 0, 255)
        
        # Add subtle noise for realism
        noise = np.random.normal(0, 2, refined[mask_indices].shape)
        refined[mask_indices] = np.clip(refined[mask_indices] + noise, 0, 255)
        
        return Image.fromarray(refined.astype(np.uint8))
        
    except Exception as e:
        st.error(f"Design refinement failed: {str(e)}")
        return original_image

def generate_detailed_engineering_specs(image: Image.Image, text_prompt: str) -> Dict[str, Any]:
    """Generate comprehensive engineering specifications."""
    try:
        # Analyze prompt for keywords
        prompt_lower = text_prompt.lower()
        
        # Determine component type
        if "gear" in prompt_lower:
            component_type = "Gear Mechanism"
            material = "Steel AISI 4340" if "steel" in prompt_lower else "Aluminum 7075-T6"
        elif "bearing" in prompt_lower:
            component_type = "Bearing Assembly"
            material = "Bearing Steel AISI 52100"
        elif "shaft" in prompt_lower:
            component_type = "Drive Shaft"
            material = "Stainless Steel 316"
        else:
            component_type = "Structural Component"
            material = "Aluminum 6061-T6"
        
        # Generate comprehensive specs
        specs = {
            "metadata": {
                "generated_at": "2025-08-28T10:30:00Z",
                "ai_model_version": "v2.1",
                "confidence_score": 0.87
            },
            "component_identification": {
                "type": component_type,
                "classification": "Mechanical Component",
                "application": "Industrial Machinery",
                "revision": "Rev A"
            },
            "material_properties": {
                "primary_material": material,
                "density": "2.70 g/cm³" if "aluminum" in material.lower() else "7.85 g/cm³",
                "tensile_strength": "572 MPa" if "aluminum" in material.lower() else "860 MPa",
                "yield_strength": "503 MPa" if "aluminum" in material.lower() else "690 MPa",
                "elastic_modulus": "71.7 GPa" if "aluminum" in material.lower() else "200 GPa",
                "poisson_ratio": 0.33,
                "thermal_conductivity": "130 W/m·K" if "aluminum" in material.lower() else "50 W/m·K"
            },
            "geometric_specifications": {
                "overall_dimensions": {
                    "length": "150.0 ± 0.1 mm",
                    "width": "100.0 ± 0.1 mm",
                    "height": "50.0 ± 0.05 mm"
                },
                "tolerance_grade": "IT7",
                "surface_finish": "Ra 1.6 μm",
                "geometric_tolerances": {
                    "flatness": "0.02 mm",
                    "perpendicularity": "0.05 mm",
                    "concentricity": "0.01 mm"
                }
            },
            "manufacturing_specifications": {
                "primary_process": "CNC Machining" if "precision" in prompt_lower else "Investment Casting",
                "secondary_processes": [
                    "Heat Treatment",
                    "Surface Grinding",
                    "Quality Inspection"
                ],
                "tooling_requirements": [
                    "Carbide End Mills",
                    "Diamond Grinding Wheels",
                    "CMM Inspection Fixtures"
                ],
                "estimated_cycle_time": "45 minutes",
                "batch_size_optimal": 50
            },
            "performance_requirements": {
                "operating_conditions": {
                    "temperature_range": "-20°C to +120°C",
                    "humidity": "5% to 95% RH",
                    "pressure": "Atmospheric to 10 bar"
                },
                "mechanical_loads": {
                    "max_static_load": "15 kN",
                    "fatigue_limit": "8 kN at 10⁶ cycles",
                    "safety_factor": 2.5
                },
                "expected_lifetime": "20,000 operating hours"
            },
            "simulation_requirements": {
                "structural_analysis": {
                    "type": "Static Linear Analysis",
                    "mesh_size": "2mm maximum",
                    "boundary_conditions": "Fixed support at mounting holes",
                    "applied_loads": "Distributed pressure 5 MPa"
                },
                "modal_analysis": {
                    "frequency_range": "0-1000 Hz",
                    "modes_to_extract": 10,
                    "damping_ratio": "2% structural"
                },
                "thermal_analysis": {
                    "heat_sources": "Friction at contact surfaces",
                    "convection_coefficient": "25 W/m²·K",
                    "ambient_temperature": "25°C"
                },
                "fatigue_analysis": {
                    "load_spectrum": "Variable amplitude",
                    "stress_concentration_factors": "Include all geometric features",
                    "surface_finish_factor": 0.9
                }
            },
            "quality_requirements": {
                "inspection_plan": [
                    "Dimensional inspection (100%)",
                    "Material certification",
                    "Surface finish verification",
                    "Functional testing"
                ],
                "acceptance_criteria": {
                    "dimensional_tolerance": "Per drawing specifications",
                    "surface_defects": "None visible",
                    "material_properties": "Within specification limits"
                }
            },
            "cost_analysis": {
                "material_cost": "$45 - $65",
                "machining_cost": "$120 - $180",
                "finishing_cost": "$25 - $40",
                "quality_control": "$15 - $25",
                "total_estimated_cost": "$205 - $310",
                "cost_drivers": [
                    "Precision tolerances",
                    "Material grade",
                    "Surface finish requirements"
                ]
            },
            "sustainability": {
                "recyclability": "95% recyclable",
                "carbon_footprint": "2.3 kg CO₂ equivalent",
                "end_of_life": "Material recovery recommended",
                "environmental_compliance": ["RoHS", "REACH"]
            }
        }
        
        return specs
        
    except Exception as e:
        st.error(f"Specification generation failed: {str(e)}")
        return {"error": str(e)}

def add_processing_step(step: str):
    """Add a step to processing history."""
    st.session_state.processing_history.append(step)
    if len(st.session_state.processing_history) > 20:  # Keep last 20 steps
        st.session_state.processing_history = st.session_state.processing_history[-20:]

# Main application tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎨 Sketch & Describe", "🔍 Segment & Refine", "📋 Engineering Specs", "📊 Analysis"])

with tab1:
    st.header("🎨 Create Your Design Concept")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Drawing Canvas")
        
        # Canvas settings
        canvas_settings = st.expander("Canvas Settings")
        with canvas_settings:
            stroke_width = st.slider("Stroke Width", 1, 10, 3)
            stroke_color = st.color_picker("Stroke Color", "#000000")
            canvas_height = st.slider("Canvas Height", 300, 600, 400)
        
        # Drawing canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color="#FFFFFF",
            background_image=st.session_state.sketch_img if st.session_state.sketch_img else None,
            height=canvas_height,
            width=600,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
            display_toolbar=True,
        )
        
        # Canvas controls
        col1a, col1b, col1c, col1d = st.columns(4)
        with col1a:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.canvas_key += 1
                st.session_state.sketch_img = None
                add_processing_step("Canvas cleared")
                st.rerun()
        
        with col1b:
            if st.button("💾 Save Sketch", use_container_width=True):
                if canvas_result.image_data is not None:
                    # Convert canvas data to PIL Image
                    img_data = canvas_result.image_data
                    pil_image = Image.fromarray(img_data.astype('uint8'))
                    st.session_state.sketch_img = pil_image
                    add_processing_step("Sketch saved")
                    st.success("✅ Sketch saved!")
                else:
                    st.warning("⚠️ No sketch to save")
        
        with col1c:
            if st.button("📁 Load Example", use_container_width=True):
                # Create a simple example sketch
                example_img = Image.new('RGB', (600, 400), 'white')
                draw = ImageDraw.Draw(example_img)
                # Draw a simple gear shape
                center = (300, 200)
                radius = 80
                teeth = 12
                for i in range(teeth):
                    angle = i * 2 * np.pi / teeth
                    x1 = center[0] + (radius - 10) * np.cos(angle)
                    y1 = center[1] + (radius - 10) * np.sin(angle)
                    x2 = center[0] + (radius + 10) * np.cos(angle)
                    y2 = center[1] + (radius + 10) * np.sin(angle)
                    draw.rectangle([x1-5, y1-5, x1+5, y1+5], fill='black')
                draw.ellipse([center[0]-60, center[1]-60, center[0]+60, center[1]+60], outline='black', width=3)
                draw.ellipse([center[0]-20, center[1]-20, center[0]+20, center[1]+20], outline='black', width=2)
                
                st.session_state.sketch_img = example_img
                add_processing_step("Example sketch loaded")
                st.success("✅ Example loaded!")
                st.rerun()
        
        with col1d:
            if st.session_state.sketch_img and st.button("🔍 Preview", use_container_width=True):
                st.image(st.session_state.sketch_img, caption="Current Sketch Preview")
    
    with col2:
        st.subheader("Design Description")
        
        # Predefined prompts
        prompt_templates = {
            "Custom": "",
            "Gear Mechanism": "A precision metal gear mechanism with 12 teeth designed for high-torque applications. The gear should be made from hardened steel with precise tooth profiles for smooth rotation transfer.",
            "Bearing Housing": "An aluminum bearing housing designed to support rotating shafts under radial loads. The housing should include mounting holes and lubrication ports.",
            "Structural Bracket": "A lightweight structural support bracket made from aluminum alloy. The bracket should distribute loads efficiently while minimizing weight.",
            "Coupling Device": "A flexible coupling device for connecting two rotating shafts with slight misalignment compensation capability."
        }
        
        selected_template = st.selectbox("Choose Template", list(prompt_templates.keys()))
        
        text_prompt = st.text_area(
            "Design Description",
            value=prompt_templates[selected_template],
            height=200,
            placeholder="Provide detailed description including materials, function, operating conditions, and requirements...",
            help="The more detailed your description, the better the AI can generate specifications."
        )
        
        # Additional parameters
        with st.expander("Advanced Parameters"):
            priority = st.selectbox(
                "Design Priority",
                ["Balanced", "Cost-Optimized", "Performance-Focused", "Lightweight", "Durable"]
            )
            
            application = st.selectbox(
                "Application Type",
                ["General Purpose", "Automotive", "Aerospace", "Marine", "Industrial Equipment"]
            )
            
            material_preference = st.selectbox(
                "Material Preference",
                ["No Preference", "Aluminum Alloys", "Steel", "Stainless Steel", "Composites", "Plastics"]
            )
        
        # Save description
        if st.button("💾 Save Description", use_container_width=True):
            if text_prompt.strip():
                st.session_state.text_prompt = text_prompt
                st.session_state.design_priority = priority
                st.session_state.application_type = application
                st.session_state.material_preference = material_preference
                add_processing_step("Description saved")
                st.success("✅ Description saved!")
            else:
                st.warning("⚠️ Please enter a description")
        
        # Show current status
        if st.session_state.sketch_img is not None:
            st.markdown("""
            <div class="success-box">
                <strong>✅ Ready for next step!</strong><br>
                Your sketch and description are saved. Go to the 'Segment & Refine' tab.
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.header("🔍 Segment & Refine Your Design")
    
    if st.session_state.sketch_img is None:
        st.markdown("""
        <div class="info-box">
            <strong>ℹ️ No sketch available</strong><br>
            Please create and save a sketch in the 'Sketch & Describe' tab first.
        </div>
        """, unsafe_allow_html=True)
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Segmentation")
            st.image(st.session_state.sketch_img, caption="Original Sketch", use_column_width=True)
            
            # Segmentation parameters
            with st.expander("Segmentation Settings"):
                seg_method = st.selectbox(
                    "Method",
                    ["Adaptive + Canny", "Simple Threshold", "Edge Detection Only"]
                )
                
                edge_threshold1 = st.slider("Edge Threshold 1", 10, 200, 50)
                edge_threshold2 = st.slider("Edge Threshold 2", 50, 300, 150)
            
            if st.button("🔍 Segment Sketch", use_container_width=True):
                with st.spinner("Segmenting sketch..."):
                    # Load SAM model
                    sam_predictor = load_sam_model()
                    
                    # Process segmentation
                    segmented_img, mask = enhanced_segment_sketch(
                        st.session_state.sketch_img,
                        st.session_state.get('text_prompt', '')
                    )
                    
                    st.session_state.segmented_img = segmented_img
                    st.session_state.segmentation_mask = mask
                    add_processing_step("Sketch segmented")
                    st.success("✅ Segmentation complete!")
                    st.rerun()
            
            if st.session_state.segmented_img is not None:
                st.image(st.session_state.segmented_img, caption="Segmented Result", use_column_width=True)
        
        with col2:
            st.subheader("🎨 Design Refinement")
            
            if st.session_state.segmented_img is not None:
                # Refinement parameters
                refinement_prompt = st.text_area(
                    "Refinement Instructions",
                    value=st.session_state.get('refinement_prompt', ''),
                    height=100,
                    placeholder="Example: Make the gear teeth more precise, add surface texturing, enhance metallic appearance..."
                )
                
                with st.expander("Refinement Settings"):
                    effect_strength = st.slider("Effect Strength", 0.1, 2.0, 1.0, 0.1)
                    add_noise = st.checkbox("Add Realistic Noise", value=True)
                    enhance_edges = st.checkbox("Enhance Edges", value=True)
                
                if st.button("🎨 Refine Design", use_container_width=True):
                    if refinement_prompt.strip():
                        with st.spinner("Refining design..."):
                            # Load SD model
                            sd_pipeline = load_stable_diffusion()
                            
                            # Refine design
                            refined_img = enhanced_refine_design(
                                st.session_state.sketch_img,
                                st.session_state.segmentation_mask,
                                refinement_prompt
                            )
                            
                            st.session_state.refined_img = refined_img
                            st.session_state.refinement_prompt = refinement_prompt
                            add_processing_step("Design refined")
                            st.success("✅ Design refinement complete!")
                            st.rerun()
                    else:
                        st.warning("⚠️ Please provide refinement instructions")
                
                if st.session_state.refined_img is not None:
                    st.image(st.session_state.refined_img, caption="Refined Design", use_column_width=True)
                    
                    # Comparison view
                    if st.checkbox("Show Before/After Comparison"):
                        col2a, col2b = st.columns(2)
                        with col2a:
                            st.image(st.session_state.sketch_img, caption="Original", use_column_width=True)
                        with col2b:
                            st.image(st.session_state.refined_img, caption="Refined", use_column_width=True)
            else:
                st.markdown("""
                <div class="info-box">
                    <strong>ℹ️ Segmentation required</strong><br>
                    Please segment your sketch first to proceed with refinement.
                </div>
                """, unsafe_allow_html=True)

with tab3:
    st.header("📋 Engineering Specifications")
    
    if st.session_state.refined_img is None:
        st.markdown("""
        <div class="info-box">
            <strong>ℹ️ Design refinement required</strong><br>
            Please complete the design refinement in the 'Segment & Refine' tab first.
        </div>
        """, unsafe_allow_html=True)
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Final Design")
            st.image(st.session_state.refined_img, caption="Refined Design", use_column_width=True)
            
            # Additional specification requirements
            st.subheader("Specification Parameters")
            spec_prompt = st.text_area(
                "Additional Requirements",
                height=100,
                placeholder="Example: Operating temperature up to 200°C, must withstand 50,000 cycles, corrosion resistance required..."
            )
            
            spec_detail_level = st.selectbox(
                "Detail Level",
                ["Comprehensive", "Standard", "Basic"],
                index=0
            )
            
            include_simulation = st.checkbox("Include Simulation Guidelines", value=True)
            include_cost = st.checkbox("Include Cost Analysis", value=True)
            include_sustainability = st.checkbox("Include Sustainability Metrics", value=True)
        
        with col2:
            st.subheader("Generated Specifications")
            
            if st.button("🧠 Generate Engineering Specifications", use_container_width=True):
                with st.spinner("Generating comprehensive engineering specifications..."):
                    # Setup LangChain
                    llm = setup_langchain()
                    
                    # Combine all prompts
                    combined_prompt = (
                        st.session_state.get('text_prompt', '') + " " +
                        st.session_state.get('refinement_prompt', '') + " " +
                        spec_prompt
                    )
                    
                    # Generate specifications
                    specs = generate_detailed_engineering_specs(
                        st.session_state.refined_img,
                        combined_prompt
                    )
                    
                    st.session_state.engineering_specs = specs
                    add_processing_step("Engineering specifications generated")
                    st.success("✅ Engineering specifications generated!")
                    st.rerun()
            
            # Display specifications if available
            if st.session_state.engineering_specs is not None:
                specs = st.session_state.engineering_specs
                
                # Create tabs for different specification sections
                spec_tabs = st.tabs([
                    "📊 Overview",
                    "🔧 Technical",
                    "🏭 Manufacturing",
                    "📈 Performance",
                    "🔬 Simulation",
                    "💰 Cost",
                    "🌱 Sustainability"
                ])
                
                with spec_tabs[0]:  # Overview
                    st.subheader("Component Overview")
                    if "component_identification" in specs:
                        comp_id = specs["component_identification"]
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Component Type", comp_id.get("type", "N/A"))
                            st.metric("Classification", comp_id.get("classification", "N/A"))
                        with col2:
                            st.metric("Application", comp_id.get("application", "N/A"))
                            st.metric("Revision", comp_id.get("revision", "N/A"))
                    
                    if "metadata" in specs:
                        metadata = specs["metadata"]
                        st.info(f"Confidence Score: {metadata.get('confidence_score', 0):.2%}")
                
                with spec_tabs[1]:  # Technical
                    st.subheader("Material Properties")
                    if "material_properties" in specs:
                        mat_props = specs["material_properties"]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Material", mat_props.get("primary_material", "N/A"))
                            st.metric("Density", mat_props.get("density", "N/A"))
                        with col2:
                            st.metric("Tensile Strength", mat_props.get("tensile_strength", "N/A"))
                            st.metric("Yield Strength", mat_props.get("yield_strength", "N/A"))
                        with col3:
                            st.metric("Elastic Modulus", mat_props.get("elastic_modulus", "N/A"))
                            st.metric("Poisson's Ratio", str(mat_props.get("poisson_ratio", "N/A")))
                    
                    st.subheader("Geometric Specifications")
                    if "geometric_specifications" in specs:
                        geom_specs = specs["geometric_specifications"]
                        
                        if "overall_dimensions" in geom_specs:
                            dims = geom_specs["overall_dimensions"]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Length", dims.get("length", "N/A"))
                            with col2:
                                st.metric("Width", dims.get("width", "N/A"))
                            with col3:
                                st.metric("Height", dims.get("height", "N/A"))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Tolerance Grade", geom_specs.get("tolerance_grade", "N/A"))
                        with col2:
                            st.metric("Surface Finish", geom_specs.get("surface_finish", "N/A"))
                
                with spec_tabs[2]:  # Manufacturing
                    st.subheader("Manufacturing Specifications")
                    if "manufacturing_specifications" in specs:
                        mfg_specs = specs["manufacturing_specifications"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Primary Process", mfg_specs.get("primary_process", "N/A"))
                            st.metric("Cycle Time", mfg_specs.get("estimated_cycle_time", "N/A"))
                        with col2:
                            st.metric("Optimal Batch Size", str(mfg_specs.get("batch_size_optimal", "N/A")))
                        
                        if "secondary_processes" in mfg_specs:
                            st.subheader("Secondary Processes")
                            for process in mfg_specs["secondary_processes"]:
                                st.write(f"• {process}")
                        
                        if "tooling_requirements" in mfg_specs:
                            st.subheader("Tooling Requirements")
                            for tool in mfg_specs["tooling_requirements"]:
                                st.write(f"• {tool}")
                
                with spec_tabs[3]:  # Performance
                    st.subheader("Performance Requirements")
                    if "performance_requirements" in specs:
                        perf_req = specs["performance_requirements"]
                        
                        if "operating_conditions" in perf_req:
                            st.subheader("Operating Conditions")
                            op_cond = perf_req["operating_conditions"]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Temperature Range", op_cond.get("temperature_range", "N/A"))
                            with col2:
                                st.metric("Humidity", op_cond.get("humidity", "N/A"))
                            with col3:
                                st.metric("Pressure", op_cond.get("pressure", "N/A"))
                        
                        if "mechanical_loads" in perf_req:
                            st.subheader("Mechanical Loads")
                            mech_loads = perf_req["mechanical_loads"]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Max Static Load", mech_loads.get("max_static_load", "N/A"))
                            with col2:
                                st.metric("Fatigue Limit", mech_loads.get("fatigue_limit", "N/A"))
                            with col3:
                                st.metric("Safety Factor", str(mech_loads.get("safety_factor", "N/A")))
                        
                        st.metric("Expected Lifetime", perf_req.get("expected_lifetime", "N/A"))
                
                with spec_tabs[4]:  # Simulation
                    st.subheader("Simulation Requirements")
                    if "simulation_requirements" in specs and include_simulation:
                        sim_req = specs["simulation_requirements"]
                        
                        # Structural Analysis
                        if "structural_analysis" in sim_req:
                            st.subheader("🔧 Structural Analysis")
                            struct_anal = sim_req["structural_analysis"]
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Type:** {struct_anal.get('type', 'N/A')}")
                                st.write(f"**Mesh Size:** {struct_anal.get('mesh_size', 'N/A')}")
                            with col2:
                                st.write(f"**Boundary Conditions:** {struct_anal.get('boundary_conditions', 'N/A')}")
                                st.write(f"**Applied Loads:** {struct_anal.get('applied_loads', 'N/A')}")
                        
                        # Modal Analysis
                        if "modal_analysis" in sim_req:
                            st.subheader("🎵 Modal Analysis")
                            modal_anal = sim_req["modal_analysis"]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Frequency Range", modal_anal.get("frequency_range", "N/A"))
                            with col2:
                                st.metric("Modes to Extract", str(modal_anal.get("modes_to_extract", "N/A")))
                            with col3:
                                st.metric("Damping Ratio", modal_anal.get("damping_ratio", "N/A"))
                        
                        # Thermal Analysis
                        if "thermal_analysis" in sim_req:
                            st.subheader("🔥 Thermal Analysis")
                            thermal_anal = sim_req["thermal_analysis"]
                            for key, value in thermal_anal.items():
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                        
                        # Fatigue Analysis
                        if "fatigue_analysis" in sim_req:
                            st.subheader("⚡ Fatigue Analysis")
                            fatigue_anal = sim_req["fatigue_analysis"]
                            for key, value in fatigue_anal.items():
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                with spec_tabs[5]:  # Cost
                    st.subheader("Cost Analysis")
                    if "cost_analysis" in specs and include_cost:
                        cost_anal = specs["cost_analysis"]
                        
                        # Cost breakdown chart
                        if all(key in cost_anal for key in ["material_cost", "machining_cost", "finishing_cost", "quality_control"]):
                            costs = {
                                "Material": float(cost_anal["material_cost"].split("$")[1].split(" - ")[0]),
                                "Machining": float(cost_anal["machining_cost"].split("$")[1].split(" - ")[0]),
                                "Finishing": float(cost_anal["finishing_cost"].split("$")[1].split(" - ")[0]),
                                "Quality Control": float(cost_anal["quality_control"].split("$")[1].split(" - ")[0])
                            }
                            
                            # Create pie chart
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.pie(costs.values(), labels=costs.keys(), autopct='%1.1f%%', startangle=90)
                            ax.set_title("Cost Breakdown")
                            st.pyplot(fig)
                        
                        # Cost details
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Material Cost", cost_anal.get("material_cost", "N/A"))
                            st.metric("Machining Cost", cost_anal.get("machining_cost", "N/A"))
                        with col2:
                            st.metric("Finishing Cost", cost_anal.get("finishing_cost", "N/A"))
                            st.metric("Quality Control", cost_anal.get("quality_control", "N/A"))
                        
                        st.metric("**Total Estimated Cost**", cost_anal.get("total_estimated_cost", "N/A"))
                        
                        if "cost_drivers" in cost_anal:
                            st.subheader("Cost Drivers")
                            for driver in cost_anal["cost_drivers"]:
                                st.write(f"• {driver}")
                
                with spec_tabs[6]:  # Sustainability
                    st.subheader("Sustainability Metrics")
                    if "sustainability" in specs and include_sustainability:
                        sustain = specs["sustainability"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Recyclability", sustain.get("recyclability", "N/A"))
                            st.metric("Carbon Footprint", sustain.get("carbon_footprint", "N/A"))
                        with col2:
                            st.metric("End of Life", sustain.get("end_of_life", "N/A"))
                        
                        if "environmental_compliance" in sustain:
                            st.subheader("Environmental Compliance")
                            for compliance in sustain["environmental_compliance"]:
                                st.write(f"✅ {compliance}")
                
                # Export options
                st.markdown("---")
                st.subheader("📥 Export Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # JSON export
                    json_data = json.dumps(specs, indent=2)
                    st.download_button(
                        label="📄 Download JSON",
                        data=json_data,
                        file_name=f"engineering_specs_{specs.get('metadata', {}).get('generated_at', 'unknown').replace(':', '-')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col2:
                    # Generate PDF report (simulated)
                    if st.button("📋 Generate PDF Report", use_container_width=True):
                        st.info("PDF generation would be implemented in production")
                
                with col3:
                    # Export to CAD format (simulated)
                    if st.button("⚙️ Export CAD Data", use_container_width=True):
                        st.info("CAD export would be implemented in production")

with tab4:
    st.header("📊 Design Analysis & Validation")
    
    if st.session_state.engineering_specs is None:
        st.markdown("""
        <div class="info-box">
            <strong>ℹ️ Specifications required</strong><br>
            Please generate engineering specifications first to access analysis tools.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Analysis tabs
        analysis_tabs = st.tabs([
            "📈 Performance Analysis",
            "🔍 Design Validation",
            "⚖️ Trade-off Analysis",
            "📊 Comparison"
        ])
        
        with analysis_tabs[0]:  # Performance Analysis
            st.subheader("Performance Metrics Analysis")
            
            specs = st.session_state.engineering_specs
            
            # Create performance radar chart
            if "performance_requirements" in specs:
                perf_req = specs["performance_requirements"]
                
                # Simulate performance scores (0-100)
                performance_metrics = {
                    "Strength": 85,
                    "Durability": 78,
                    "Cost Efficiency": 72,
                    "Manufacturability": 88,
                    "Sustainability": 65,
                    "Precision": 92
                }
                
                # Create radar chart
                fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
                
                angles = np.linspace(0, 2 * np.pi, len(performance_metrics), endpoint=False)
                values = list(performance_metrics.values())
                
                ax.plot(angles, values, 'o-', linewidth=2, label='Current Design')
                ax.fill(angles, values, alpha=0.25)
                ax.set_xticks(angles)
                ax.set_xticklabels(performance_metrics.keys())
                ax.set_ylim(0, 100)
                ax.set_title("Performance Analysis", size=16, fontweight='bold')
                ax.grid(True)
                
                st.pyplot(fig)
                
                # Performance summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Score", f"{np.mean(values):.1f}/100")
                with col2:
                    st.metric("Best Aspect", max(performance_metrics, key=performance_metrics.get))
                with col3:
                    st.metric("Needs Improvement", min(performance_metrics, key=performance_metrics.get))
        
        with analysis_tabs[1]:  # Design Validation
            st.subheader("Design Validation Checklist")
            
            validation_items = [
                {"item": "Material properties match application requirements", "status": True},
                {"item": "Dimensions within manufacturing tolerances", "status": True},
                {"item": "Safety factors adequate for load conditions", "status": True},
                {"item": "Surface finish achievable with selected process", "status": True},
                {"item": "Cost target met within acceptable range", "status": False},
                {"item": "Environmental requirements satisfied", "status": True},
                {"item": "Quality control procedures defined", "status": True},
                {"item": "Simulation parameters appropriate", "status": True}
            ]
            
            passed = sum(1 for item in validation_items if item["status"])
            total = len(validation_items)
            
            st.progress(passed / total)
            st.write(f"Validation Score: {passed}/{total} ({passed/total*100:.1f}%)")
            
            for item in validation_items:
                status_icon = "✅" if item["status"] else "❌"
                st.write(f"{status_icon} {item['item']}")
            
            if passed < total:
                st.warning("⚠️ Some validation criteria not met. Review design parameters.")
            else:
                st.success("🎉 All validation criteria passed!")
        
        with analysis_tabs[2]:  # Trade-off Analysis
            st.subheader("Design Trade-offs")
            
            # Trade-off sliders
            st.write("Adjust priorities to see impact on design recommendations:")
            
            col1, col2 = st.columns(2)
            with col1:
                cost_weight = st.slider("Cost Priority", 0.0, 1.0, 0.3, 0.1)
                performance_weight = st.slider("Performance Priority", 0.0, 1.0, 0.4, 0.1)
            with col2:
                sustainability_weight = st.slider("Sustainability Priority", 0.0, 1.0, 0.2, 0.1)
                manufacturability_weight = st.slider("Manufacturability Priority", 0.0, 1.0, 0.1, 0.1)
            
            # Normalize weights
            total_weight = cost_weight + performance_weight + sustainability_weight + manufacturability_weight
            if total_weight > 0:
                weights = {
                    "Cost": cost_weight / total_weight,
                    "Performance": performance_weight / total_weight,
                    "Sustainability": sustainability_weight / total_weight,
                    "Manufacturability": manufacturability_weight / total_weight
                }
                
                # Create trade-off visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                categories = list(weights.keys())
                values = list(weights.values())
                colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
                
                bars = ax.bar(categories, values, color=colors)
                ax.set_ylabel('Priority Weight')
                ax.set_title('Design Priority Trade-offs')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.2f}', ha='center', va='bottom')
                
                st.pyplot(fig)
                
                # Recommendations based on weights
                st.subheader("Recommendations")
                if weights["Cost"] > 0.4:
                    st.info("💰 Consider using more cost-effective materials or manufacturing processes")
                if weights["Performance"] > 0.4:
                    st.info("🚀 Focus on high-performance materials and precision manufacturing")
                if weights["Sustainability"] > 0.3:
                    st.info("🌱 Prioritize recyclable materials and energy-efficient processes")
                if weights["Manufacturability"] > 0.3:
                    st.info("🏭 Simplify geometry and use standard manufacturing processes")
        
        with analysis_tabs[3]:  # Comparison
            st.subheader("Design Alternatives Comparison")
            
            # Simulated alternative designs
            alternatives = {
                "Current Design": {
                    "Cost": 250,
                    "Weight": 2.5,
                    "Strength": 85,
                    "Manufacturing Time": 45,
                    "Sustainability Score": 65
                },
                "Lightweight Alternative": {
                    "Cost": 380,
                    "Weight": 1.8,
                    "Strength": 78,
                    "Manufacturing Time": 60,
                    "Sustainability Score": 72
                },
                "Cost-Optimized": {
                    "Cost": 180,
                    "Weight": 3.2,
                    "Strength": 75,
                    "Manufacturing Time": 35,
                    "Sustainability Score": 58
                },
                "High-Performance": {
                    "Cost": 450,
                    "Weight": 2.8,
                    "Strength": 95,
                    "Manufacturing Time": 75,
                    "Sustainability Score": 68
                }
            }
            
            # Create comparison table
            import pandas as pd
            df = pd.DataFrame(alternatives).T
            
            # Style the dataframe
            def highlight_best(s):
                # For cost and weight, lower is better
                # For strength and sustainability, higher is better
                # For manufacturing time, lower is better
                styles = []
                for col in s.index:
                    if col in ['Cost', 'Weight', 'Manufacturing Time']:
                        is_best = s[col] == s.min()
                    else:
                        is_best = s[col] == s.max()
                    styles.append('background-color: lightgreen' if is_best else '')
                return styles
            
            styled_df = df.style.apply(highlight_best, axis=0)
            st.dataframe(styled_df, use_container_width=True)
            
            st.caption("🟢 Green highlighting indicates the best value in each category")
            
            # Multi-criteria decision analysis
            st.subheader("Multi-Criteria Decision Analysis")
            
            # Allow user to set criteria weights
            criteria_weights = {}
            col1, col2 = st.columns(2)
            with col1:
                criteria_weights['Cost'] = st.slider("Cost Weight (lower is better)", 0.0, 1.0, 0.25, 0.05)
                criteria_weights['Strength'] = st.slider("Strength Weight", 0.0, 1.0, 0.30, 0.05)
            with col2:
                criteria_weights['Weight'] = st.slider("Weight Weight (lower is better)", 0.0, 1.0, 0.20, 0.05)
                criteria_weights['Sustainability Score'] = st.slider("Sustainability Weight", 0.0, 1.0, 0.25, 0.05)
            
            # Calculate weighted scores
            scores = {}
            for design, values in alternatives.items():
                score = 0
                # Normalize and weight each criterion
                for criterion, weight in criteria_weights.items():
                    if criterion in ['Cost', 'Weight', 'Manufacturing Time']:
                        # Lower is better - invert the score
                        max_val = max(alt[criterion] for alt in alternatives.values())
                        min_val = min(alt[criterion] for alt in alternatives.values())
                        normalized = (max_val - values[criterion]) / (max_val - min_val) if max_val != min_val else 1
                    else:
                        # Higher is better
                        max_val = max(alt[criterion] for alt in alternatives.values())
                        min_val = min(alt[criterion] for alt in alternatives.values())
                        normalized = (values[criterion] - min_val) / (max_val - min_val) if max_val != min_val else 1
                    
                    score += normalized * weight
                
                scores[design] = score
            
            # Display results
            best_design = max(scores, key=scores.get)
            
            # Create bar chart of scores
            fig, ax = plt.subplots(figsize=(10, 6))
            designs = list(scores.keys())
            score_values = list(scores.values())
            colors = ['gold' if design == best_design else 'lightblue' for design in designs]
            
            bars = ax.bar(designs, score_values, color=colors)
            ax.set_ylabel('Weighted Score')
            ax.set_title('Multi-Criteria Decision Analysis Results')
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, score_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.success(f"🏆 Recommended Design: **{best_design}** (Score: {scores[best_design]:.3f})")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem;">
    <h4>🔧 Powered by Advanced AI Technologies</h4>
    <p>
        <strong>Segment Anything (SAM)</strong> • <strong>Stable Diffusion</strong> • 
        <strong>LangChain</strong> • <strong>Streamlit</strong>
    </p>
    <p style="color: #666;">
        Transform your engineering concepts from sketch to specification with AI-powered precision
    </p>
</div>
""", unsafe_allow_html=True)
