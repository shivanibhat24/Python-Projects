# DIY Craft Generator - Sample Usage Guide

## Basic Usage

Here's how to use the DIY Craft Generator application:

```python
# Import the class
from diy_craft_generator import DIYCraftGenerator

# Create an instance
generator = DIYCraftGenerator()

# Get a project suggestion based on available materials
materials = ["cardboard boxes", "paint", "glue", "yarn"]
theme = "home decor"
difficulty = "easy"  # options: "easy", "medium", "advanced", or None for any

# Generate project instructions
instructions = generator.suggest_project(materials, theme, difficulty)
print(instructions)
```

## Example Scenarios

### Scenario 1: Kid's Craft with Limited Materials

```python
generator = DIYCraftGenerator()

# A parent with basic supplies looking for a kid-friendly project
materials = ["paper plates", "markers", "scissors", "glue"]
theme = "kids"

instructions = generator.suggest_project(materials, theme)
print(instructions)
```

### Scenario 2: Upcycling Project

```python
generator = DIYCraftGenerator()

# Someone looking to upcycle household items
materials = ["plastic bottles", "scissors", "paint", "soil", "seeds"]
theme = "garden"

instructions = generator.suggest_project(materials, theme)
print(instructions)
```

### Scenario 3: Holiday Decoration Project

```python
generator = DIYCraftGenerator()

# Someone preparing for holiday decorations
materials = ["paper", "glitter", "paint", "ribbon", "cardboard boxes"]
theme = "holiday"

instructions = generator.suggest_project(materials, theme, "medium")
print(instructions)
```

## Advanced Functions

The generator also provides these helpful functions:

- `list_all_materials()` - Shows all materials in the database by category
- `list_all_themes()` - Shows all available themes
- `list_all_projects()` - Shows all project templates by difficulty level
- `generate_project_instructions(project_name, materials, theme)` - Generate instructions for a specific project

## Extending the System

You can easily extend the system by:

1. Adding new materials to the `materials_db` dictionary
2. Creating new project templates in the `project_templates` dictionary
3. Adding new step instructions in the `steps_templates` dictionary
4. Expanding theme details in the `theme_details` dictionary

This makes the system highly customizable as your crafting needs grow!
