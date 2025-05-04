# Smart Grocery Planner
# An application that suggests meals based on pantry photos
# Uses YOLOv8 for object detection, Mistral for recipe generation, OpenCV for image processing, and Streamlit for UI

import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import io
import json
import os
import requests
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import tempfile
import re

# Set page configuration
st.set_page_config(
    page_title="Smart Grocery Planner",
    page_icon="🥗",
    layout="wide",
)

# App title and description
st.title("🥗 Smart Grocery Planner")
st.markdown("""
Take a photo of your pantry items, and get personalized meal suggestions based on what you have!
Upload an image of your pantry, fridge, or counter with food items visible.
""")

# Global variables
MODEL_PATH = "yolov8n.pt"  # Default model path
CONFIDENCE_THRESHOLD = 0.25  # Default confidence threshold
MAX_RECIPES = 3  # Maximum number of recipes to generate

# Initialize session state for storing data between reruns
if 'detected_items' not in st.session_state:
    st.session_state.detected_items = []
if 'recipes' not in st.session_state:
    st.session_state.recipes = []
if 'missing_ingredients' not in st.session_state:
    st.session_state.missing_ingredients = []
if 'shopping_list' not in st.session_state:
    st.session_state.shopping_list = []

# Function to download YOLOv8 model if not present
@st.cache_resource
def load_model():
    """Load YOLOv8 model with caching to avoid reloading on each rerun"""
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to process the uploaded image and detect food items
def detect_food_items(image, confidence_threshold=0.25):
    """Detect food items in the uploaded image using YOLOv8"""
    model = load_model()
    if model is None:
        return [], None
    
    # Convert image to format acceptable by YOLOv8
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = np.array(image)
    
    # Run inference
    results = model(img, conf=confidence_threshold)
    
    # Process results
    detected_items = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            label = result.names[cls_id]
            
            # Only consider food items (assuming model is trained on COCO or similar)
            # This filtering might need adjustment based on the specific model used
            food_categories = [
                'apple', 'orange', 'banana', 'carrot', 'broccoli', 'tomato', 'potato',
                'onion', 'pepper', 'lettuce', 'cucumber', 'celery', 'bread', 'sandwich',
                'pizza', 'donut', 'cake', 'egg', 'cheese', 'milk', 'bottle', 'wine glass',
                'cup', 'bowl', 'spoon', 'fork', 'knife', 'rice', 'pasta', 'meat', 'fish',
                'chicken', 'beef', 'pork', 'lamb', 'fruit', 'vegetable'
            ]
            
            # Use fuzzy matching for food categories
            if any(food_word in label.lower() for food_word in food_categories) or any(label.lower() in food_word for food_word in food_categories):
                detected_items.append({
                    'label': label,
                    'confidence': conf,
                    'box': box.xyxy.tolist()[0]  # Convert tensor to list for JSON serialization
                })
    
    # Visualize results
    annotated_img = results[0].plot()
    
    # Remove duplicates while preserving order
    unique_items = []
    unique_labels = set()
    for item in detected_items:
        if item['label'] not in unique_labels:
            unique_labels.add(item['label'])
            unique_items.append(item)
    
    return unique_items, annotated_img

# Function to generate recipe suggestions based on detected items
def generate_recipes(ingredients: List[str], dietary_preferences: str = "", max_recipes: int = 3) -> List[Dict]:
    """Generate recipe suggestions based on detected ingredients using Mistral API"""
    # In a production app, you would call the Mistral API here
    # For demonstration, we'll use a simplified approach
    
    # Convert ingredients to comma-separated string
    ingredient_list = ", ".join(ingredients)
    
    # Create prompt for recipe generation
    prompt = f"""
    Generate {max_recipes} recipe suggestions using some or all of these ingredients: {ingredient_list}.
    {dietary_preferences}
    
    For each recipe, provide:
    1. Recipe name
    2. List of required ingredients (marking which ones are already available and which need to be purchased)
    3. Brief cooking instructions
    4. Preparation time
    5. Difficulty level (Easy, Medium, Hard)
    
    Format each recipe as JSON with these fields: name, ingredients (list of dicts with 'name' and 'available' fields), instructions, prep_time, difficulty.
    """
    
    # Simulate API response with mock data
    # In a real application, replace this with actual API call
    recipes = []
    
    # Mock recipes based on ingredients
    if "apple" in ingredients or "banana" in ingredients or "fruit" in ingredients:
        recipes.append({
            "name": "Fruit Salad with Honey Yogurt",
            "ingredients": [
                {"name": "apple", "available": "apple" in ingredients},
                {"name": "banana", "available": "banana" in ingredients},
                {"name": "orange", "available": "orange" in ingredients},
                {"name": "honey", "available": "honey" in ingredients},
                {"name": "yogurt", "available": "yogurt" in ingredients},
                {"name": "mint leaves", "available": "mint" in ingredients}
            ],
            "instructions": "1. Wash and dice all fruits. 2. Mix in a bowl. 3. Combine honey and yogurt. 4. Pour over fruits and garnish with mint.",
            "prep_time": "10 minutes",
            "difficulty": "Easy"
        })
    
    if "bread" in ingredients or "cheese" in ingredients:
        recipes.append({
            "name": "Grilled Cheese Sandwich",
            "ingredients": [
                {"name": "bread", "available": "bread" in ingredients},
                {"name": "cheese", "available": "cheese" in ingredients},
                {"name": "butter", "available": "butter" in ingredients},
                {"name": "ham (optional)", "available": "ham" in ingredients}
            ],
            "instructions": "1. Butter the outside of two bread slices. 2. Place cheese (and ham if available) between slices. 3. Grill on pan until golden brown on both sides.",
            "prep_time": "5 minutes",
            "difficulty": "Easy"
        })
    
    if "pasta" in ingredients or "tomato" in ingredients:
        recipes.append({
            "name": "Simple Pasta with Tomato Sauce",
            "ingredients": [
                {"name": "pasta", "available": "pasta" in ingredients},
                {"name": "tomato", "available": "tomato" in ingredients},
                {"name": "garlic", "available": "garlic" in ingredients},
                {"name": "olive oil", "available": "oil" in ingredients},
                {"name": "basil", "available": "basil" in ingredients},
                {"name": "parmesan cheese", "available": "cheese" in ingredients}
            ],
            "instructions": "1. Cook pasta according to package. 2. Sauté garlic in olive oil. 3. Add chopped tomatoes and simmer for 10 minutes. 4. Toss with pasta and garnish with basil and cheese.",
            "prep_time": "20 minutes",
            "difficulty": "Easy"
        })
    
    if "chicken" in ingredients or "vegetable" in ingredients:
        recipes.append({
            "name": "Chicken and Vegetable Stir-Fry",
            "ingredients": [
                {"name": "chicken breast", "available": "chicken" in ingredients},
                {"name": "bell pepper", "available": "pepper" in ingredients},
                {"name": "broccoli", "available": "broccoli" in ingredients},
                {"name": "carrot", "available": "carrot" in ingredients},
                {"name": "soy sauce", "available": "sauce" in ingredients},
                {"name": "rice", "available": "rice" in ingredients}
            ],
            "instructions": "1. Slice chicken and vegetables. 2. Stir-fry chicken until cooked. 3. Add vegetables and cook until tender. 4. Season with soy sauce. 5. Serve over rice.",
            "prep_time": "25 minutes",
            "difficulty": "Medium"
        })
    
    # If no specific recipes match or we need more, add generic ones
    while len(recipes) < max_recipes:
        recipes.append({
            "name": f"Creative Dish with {ingredients[0] if ingredients else 'Available Ingredients'}",
            "ingredients": [
                {"name": ingredient, "available": True} for ingredient in ingredients[:3]
            ] + [
                {"name": "salt and pepper", "available": True},
                {"name": "cooking oil", "available": False}
            ],
            "instructions": "Combine ingredients in a creative way based on your cooking preferences.",
            "prep_time": "15-20 minutes",
            "difficulty": "Medium"
        })
    
    return recipes[:max_recipes]

# Function to generate shopping list from missing ingredients
def generate_shopping_list(recipes: List[Dict]) -> List[str]:
    """Generate a shopping list from missing ingredients in selected recipes"""
    shopping_list = set()
    
    for recipe in recipes:
        for ingredient in recipe['ingredients']:
            if not ingredient['available']:
                shopping_list.add(ingredient['name'])
    
    return sorted(list(shopping_list))

# Sidebar for app settings
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider("Detection Confidence", 0.1, 1.0, CONFIDENCE_THRESHOLD, 0.05)
    
    st.header("Dietary Preferences")
    vegetarian = st.checkbox("Vegetarian")
    vegan = st.checkbox("Vegan")
    gluten_free = st.checkbox("Gluten-Free")
    dairy_free = st.checkbox("Dairy-Free")
    
    # Compile dietary preferences
    dietary_prefs = []
    if vegetarian:
        dietary_prefs.append("Vegetarian")
    if vegan:
        dietary_prefs.append("Vegan")
    if gluten_free:
        dietary_prefs.append("Gluten-Free")
    if dairy_free:
        dietary_prefs.append("Dairy-Free")
    
    dietary_preferences_str = "Dietary preferences: " + ", ".join(dietary_prefs) if dietary_prefs else ""

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.header("Upload Pantry Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Camera input option
    camera_input = st.camera_input("Or take a photo")
    
    # Process the uploaded image or camera input
    if uploaded_file is not None or camera_input is not None:
        input_image = Image.open(uploaded_file if uploaded_file is not None else camera_input)
        st.image(input_image, caption="Uploaded Image", use_column_width=True)
        
        # Detect food items
        with st.spinner("Detecting food items..."):
            detected_items, annotated_img = detect_food_items(input_image, confidence_threshold)
            
            if detected_items:
                st.session_state.detected_items = [item['label'] for item in detected_items]
                
                # Display annotated image
                st.image(annotated_img, caption="Detected Items", use_column_width=True)
                
                # Display detected items
                st.write(f"Detected {len(detected_items)} food items:")
                detected_labels = [item['label'] for item in detected_items]
                st.write(", ".join(detected_labels))
            else:
                st.warning("No food items detected. Try adjusting the confidence threshold or upload a clearer image.")
    
    # Manual input for items
    st.header("Manual Input")
    st.markdown("Don't see all your items? Add them manually:")
    
    # Get existing items
    existing_items = st.session_state.detected_items.copy() if hasattr(st.session_state, 'detected_items') else []
    
    # Input for new item
    new_item = st.text_input("Enter an item name")
    if st.button("Add Item") and new_item:
        if new_item not in existing_items:
            existing_items.append(new_item)
            st.session_state.detected_items = existing_items
            st.success(f"Added {new_item} to your pantry items!")
    
    # Display and allow editing of all items
    if existing_items:
        st.subheader("Your Pantry Items")
        items_to_remove = []
        for i, item in enumerate(existing_items):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{i+1}. {item}")
            with col2:
                if st.button("Remove", key=f"remove_{i}"):
                    items_to_remove.append(i)
        
        # Remove items marked for removal
        if items_to_remove:
            for index in sorted(items_to_remove, reverse=True):
                del existing_items[index]
            st.session_state.detected_items = existing_items
            st.experimental_rerun()

with col2:
    st.header("Recipe Suggestions")
    
    # Generate recipes button
    if st.button("Generate Recipe Suggestions") and st.session_state.detected_items:
        with st.spinner("Generating recipe suggestions..."):
            recipes = generate_recipes(
                st.session_state.detected_items,
                dietary_preferences_str,
                MAX_RECIPES
            )
            st.session_state.recipes = recipes
    
    # Display recipes
    if st.session_state.recipes:
        for i, recipe in enumerate(st.session_state.recipes):
            with st.expander(f"{i+1}. {recipe['name']}", expanded=True):
                st.subheader("Ingredients")
                
                # Separate available and missing ingredients
                available = []
                missing = []
                
                for ingredient in recipe['ingredients']:
                    if ingredient['available']:
                        available.append(f"✓ {ingredient['name']}")
                    else:
                        missing.append(f"✗ {ingredient['name']}")
                
                # Display ingredients
                if available:
                    st.markdown("**Available:**")
                    for item in available:
                        st.markdown(item)
                
                if missing:
                    st.markdown("**Need to Buy:**")
                    for item in missing:
                        st.markdown(item)
                
                st.subheader("Instructions")
                st.write(recipe['instructions'])
                
                st.markdown(f"**Preparation Time:** {recipe['prep_time']}")
                st.markdown(f"**Difficulty:** {recipe['difficulty']}")
                
                # Add to shopping list button
                if st.button(f"Add missing ingredients to shopping list", key=f"add_shopping_{i}"):
                    missing_ingredients = [ingredient['name'] for ingredient in recipe['ingredients'] if not ingredient['available']]
                    for item in missing_ingredients:
                        if item not in st.session_state.shopping_list:
                            st.session_state.shopping_list.append(item)
    
    # Shopping list
    st.header("Shopping List")
    if st.session_state.shopping_list:
        for i, item in enumerate(st.session_state.shopping_list):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{i+1}. {item}")
            with col2:
                if st.button("Remove", key=f"remove_shopping_{i}"):
                    st.session_state.shopping_list.pop(i)
                    st.experimental_rerun()
        
        # Export shopping list
        if st.button("Export Shopping List"):
            shopping_list_text = "\n".join([f"- {item}" for item in st.session_state.shopping_list])
            st.download_button(
                label="Download as text",
                data=shopping_list_text,
                file_name="shopping_list.txt",
                mime="text/plain"
            )
    else:
        st.write("Your shopping list is empty.")

# Setup instructions in the sidebar
with st.sidebar:
    st.header("Setup Instructions")
    with st.expander("How to set up"):
        st.markdown("""
        ### Setup Steps:
        1. Install required packages:
        ```
        pip install streamlit opencv-python pillow ultralytics torch
        ```
        
        2. Download YOLOv8 model:
        ```
        # The app will download this automatically
        ```
        
        3. Run the app:
        ```
        streamlit run app.py
        ```
        
        ### Usage:
        1. Upload an image of your pantry/fridge
        2. Adjust settings if needed
        3. Review detected items and add any missing ones
        4. Generate recipe suggestions
        5. Add missing ingredients to your shopping list
        """)

# Footer
st.markdown("---")
st.markdown("Smart Grocery Planner | Built with YOLOv8, OpenCV, and Streamlit")
