import random
import re
import json
from collections import defaultdict

class DIYCraftGenerator:
    """
    A generative AI system that creates DIY craft project instructions
    based on user inputs like available materials and theme preferences.
    """
    
    def __init__(self):
        # Knowledge base of materials categorized by type
        self.materials_db = {
            "paper": ["construction paper", "newspaper", "cardstock", "tissue paper", "origami paper", "magazine pages", "paper bags", "wrapping paper"],
            "fabric": ["felt", "cotton", "denim", "yarn", "embroidery floss", "ribbon", "burlap", "old t-shirts", "buttons", "lace"],
            "recyclables": ["cardboard boxes", "plastic bottles", "bottle caps", "tin cans", "glass jars", "toilet paper rolls", "egg cartons"],
            "nature": ["pinecones", "twigs", "leaves", "stones", "shells", "dried flowers", "acorns", "moss"],
            "craft_supplies": ["glue", "scissors", "paint", "markers", "glitter", "beads", "popsicle sticks", "pipe cleaners", "googly eyes", "clay"],
            "household": ["mason jars", "clothespins", "straws", "coffee filters", "old CDs", "wooden spoons", "wire hangers"]
        }
        
        # Common themes for craft projects
        self.themes = ["seasonal", "holiday", "home decor", "kids", "gifts", "garden", "organization", "wearable", "wall art", "party decoration"]
        
        # Project templates with difficulty levels
        self.project_templates = {
            "easy": [
                {"name": "Decorative Jar", "required_materials": ["glass jars", "paint"], "optional_materials": ["ribbon", "glitter", "beads"]},
                {"name": "Paper Garland", "required_materials": ["paper"], "optional_materials": ["string", "paint", "glitter", "markers"]},
                {"name": "Rock Painting", "required_materials": ["stones", "paint"], "optional_materials": ["markers", "glitter"]},
                {"name": "Paper Plate Craft", "required_materials": ["paper plates"], "optional_materials": ["markers", "paint", "yarn", "glitter"]}
            ],
            "medium": [
                {"name": "Fabric Wall Hanging", "required_materials": ["fabric", "wooden dowel"], "optional_materials": ["beads", "yarn", "ribbon"]},
                {"name": "Paper Mache Bowl", "required_materials": ["newspaper", "flour", "balloon"], "optional_materials": ["paint", "glitter", "tissue paper"]},
                {"name": "Upcycled Bottle Planter", "required_materials": ["plastic bottles", "scissors"], "optional_materials": ["paint", "soil", "seeds", "ribbon"]},
                {"name": "DIY Candle Holder", "required_materials": ["glass jars"], "optional_materials": ["ribbon", "twine", "paint", "glitter", "beads"]}
            ],
            "advanced": [
                {"name": "Woven Wall Art", "required_materials": ["yarn", "wooden dowel"], "optional_materials": ["beads", "fabric scraps", "twigs"]},
                {"name": "Mosaic Picture Frame", "required_materials": ["picture frame", "tiles/broken ceramics", "grout"], "optional_materials": ["paint", "glitter"]},
                {"name": "Fabric Bowl", "required_materials": ["fabric", "rope", "thread"], "optional_materials": ["beads", "embroidery floss", "buttons"]},
                {"name": "Mixed Media Canvas", "required_materials": ["canvas", "paint"], "optional_materials": ["magazine pages", "fabric scraps", "buttons", "yarn"]}
            ]
        }
        
        # Project steps templates
        self.steps_templates = {
            "Decorative Jar": [
                "Clean {material: glass jars} thoroughly and let dry completely.",
                "Apply a base coat of {material: paint} to the jar and let dry.",
                "{if: ribbon} Tie {material: ribbon} around the jar's neck for decoration.",
                "{if: glitter} Apply glue in patterns and sprinkle {material: glitter} over it.",
                "{if: beads} Attach {material: beads} with strong glue to create patterns.",
                "Optional: Seal with clear varnish for durability.",
                "Use as a vase, storage container, or candle holder!"
            ],
            
            "Paper Garland": [
                "Cut {material: paper} into desired shapes (circles, triangles, hearts, etc.).",
                "{if: paint} Decorate the shapes with {material: paint} and let dry.",
                "{if: markers} Draw designs on the shapes using {material: markers}.",
                "{if: glitter} Add {material: glitter} to some shapes for extra sparkle.",
                "Punch holes in each shape at the top.",
                "Thread {material: string} through the holes to connect all shapes.",
                "Hang your garland to decorate your space!"
            ],
            
            "Rock Painting": [
                "Clean {material: stones} and let them dry completely.",
                "Apply a base coat of {material: paint} if desired and let dry.",
                "Paint designs, patterns, or characters on the rocks.",
                "{if: markers} Add details with {material: markers}.",
                "{if: glitter} Apply {material: glitter} to add sparkle to certain areas.",
                "Seal with clear varnish or mod podge for protection.",
                "Use as paperweights, garden decorations, or gifts!"
            ],
            
            "Paper Plate Craft": [
                "Gather {material: paper plates} and plan your design.",
                "Cut the plate if needed for your specific project.",
                "{if: paint} Paint the plate with {material: paint} and let dry.",
                "{if: markers} Draw designs using {material: markers}.",
                "{if: yarn} Attach {material: yarn} for hair or other details.",
                "{if: glitter} Add {material: glitter} for sparkle and decoration.",
                "Display your creation or use it for pretend play!"
            ],
            
            "Fabric Wall Hanging": [
                "Cut {material: wooden dowel} to your desired width.",
                "Cut different lengths of {material: fabric} for hanging.",
                "Fold each piece of fabric in half and loop around the dowel.",
                "{if: beads} Thread {material: beads} onto some fabric pieces for decoration.",
                "{if: yarn} Incorporate {material: yarn} pieces for texture variety.",
                "Trim the bottom to create your desired shape.",
                "Attach string to the ends of the dowel for hanging."
            ],
            
            "Paper Mache Bowl": [
                "Mix paste with equal parts {material: flour} and water.",
                "Inflate a {material: balloon} to your desired bowl size.",
                "Tear {material: newspaper} into strips.",
                "Dip paper strips into paste and layer over the balloon.",
                "Let dry completely (24-48 hours).",
                "Pop and remove the balloon.",
                "{if: paint} Paint the bowl with {material: paint}.",
                "{if: tissue paper} Decorate with {material: tissue paper} using mod podge.",
                "Seal with clear varnish for durability."
            ],
            
            "Upcycled Bottle Planter": [
                "Clean {material: plastic bottles} thoroughly.",
                "Using {material: scissors}, cut the bottle at your desired height.",
                "{if: paint} Paint the bottle with {material: paint} suited for plastic.",
                "Make drainage holes in the bottom.",
                "{if: ribbon} Decorate with {material: ribbon} around the rim.",
                "{if: soil} Fill with {material: soil} and plant your {material: seeds}.",
                "Place in a sunny spot and water regularly."
            ],
            
            "DIY Candle Holder": [
                "Clean {material: glass jars} and remove any labels.",
                "{if: paint} Apply {material: paint} in your desired pattern and let dry.",
                "{if: ribbon} Wrap {material: ribbon} around the rim or middle and secure with glue.",
                "{if: twine} Wrap {material: twine} around the jar for a rustic look.",
                "{if: glitter} Apply glue and {material: glitter} in patterns.",
                "{if: beads} Attach {material: beads} with strong glue for decoration.",
                "Place a tealight or votive candle inside and enjoy the ambiance!"
            ],
            
            "Woven Wall Art": [
                "Secure {material: wooden dowel} as the top support for your weaving.",
                "Tie {material: yarn} strands to the dowel to create the warp threads.",
                "Begin weaving different colors and textures of yarn horizontally.",
                "{if: beads} Incorporate {material: beads} into your weaving for decoration.",
                "{if: fabric scraps} Weave in {material: fabric scraps} for added texture.",
                "{if: twigs} Integrate {material: twigs} for a natural element.",
                "Trim and style the bottom when your weaving is complete.",
                "Attach a string to the dowel ends for hanging."
            ],
            
            "Mosaic Picture Frame": [
                "Prepare your {material: picture frame} by cleaning the surface.",
                "Break {material: tiles/broken ceramics} into small pieces (carefully!).",
                "Arrange your mosaic design on the frame without glue first.",
                "Once satisfied, glue pieces one by one in your design.",
                "Let the glue dry completely.",
                "Mix and apply {material: grout} between all the pieces.",
                "Wipe away excess grout before it dries completely.",
                "{if: paint} Add details with {material: paint} if desired.",
                "Seal with clear varnish for protection."
            ],
            
            "Fabric Bowl": [
                "Cut {material: fabric} into long strips.",
                "Begin coiling {material: rope} into a spiral for the base.",
                "Wrap fabric around the rope as you go.",
                "Use {material: thread} to sew through both fabric and rope to secure.",
                "Continue coiling and sewing to build up the sides of your bowl.",
                "{if: beads} Add {material: beads} to the thread occasionally for decoration.",
                "{if: embroidery floss} Use {material: embroidery floss} for decorative stitches.",
                "{if: buttons} Sew {material: buttons} onto the outside for embellishment.",
                "Shape your bowl as desired as you work."
            ],
            
            "Mixed Media Canvas": [
                "Prime your {material: canvas} if it isn't pre-primed.",
                "Create a background using {material: paint}.",
                "{if: magazine pages} Cut shapes or images from {material: magazine pages} and collage onto the canvas.",
                "{if: fabric scraps} Glue {material: fabric scraps} to create texture and visual interest.",
                "{if: buttons} Attach {material: buttons} for dimensional elements.",
                "{if: yarn} Glue or stitch {material: yarn} to create lines or borders.",
                "Add additional paint layers to blend elements together.",
                "Seal with clear varnish when complete."
            ]
        }
        
        # Theme-specific additional details
        self.theme_details = {
            "seasonal": {
                "spring": ["Add flower elements", "Use pastel colors", "Incorporate seed packets or planting themes"],
                "summer": ["Use bright colors", "Add seashells or beach elements", "Incorporate sun or water themes"],
                "fall": ["Use autumn color palette (orange, red, brown)", "Add leaf impressions or real leaves", "Incorporate harvest elements"],
                "winter": ["Use cool colors with white accents", "Add snowflake designs", "Incorporate cozy elements like miniature scarves"]
            },
            "holiday": {
                "Christmas": ["Use red and green color scheme", "Add glitter for snow effect", "Incorporate stars or tree shapes"],
                "Halloween": ["Use orange and black colors", "Add spooky elements", "Incorporate bat or pumpkin shapes"],
                "Easter": ["Use pastel colors", "Add rabbit or egg designs", "Incorporate flower motifs"],
                "Valentine's": ["Use red and pink colors", "Add heart shapes", "Incorporate lace or doily elements"]
            },
            "home decor": ["Consider your room's color palette", "Create pieces that serve a purpose (storage, display, etc.)", "Think about where the item will be placed"],
            "kids": ["Use bright colors", "Ensure all materials are child-safe", "Consider making the project interactive"],
            "gifts": ["Add personalized elements", "Consider packaging", "Think about the recipient's preferences"],
            "garden": ["Use weather-resistant materials", "Consider hanging vs. standing placement", "Incorporate natural elements"],
            "organization": ["Focus on functionality", "Label components clearly", "Ensure durability for repeated use"],
            "wearable": ["Test for comfort", "Use flexible connections", "Consider size adjustability"],
            "wall art": ["Plan hanging mechanisms", "Consider visual balance", "Think about scale for your wall space"],
            "party decoration": ["Match party color theme", "Create multiples for consistency", "Consider ease of setup/takedown"]
        }

    def flatten_materials_list(self):
        """Returns a flat list of all materials"""
        flat_list = []
        for category in self.materials_db.values():
            flat_list.extend(category)
        return flat_list

    def get_material_category(self, material):
        """Find which category a material belongs to"""
        for category, materials in self.materials_db.items():
            if material in materials:
                return category
        return None

    def filter_projects_by_materials(self, available_materials, difficulty=None):
        """
        Find projects that can be created with the available materials
        """
        suitable_projects = []
        
        # Normalize material names
        available_materials = [m.lower().strip() for m in available_materials]
        
        # Add general category names if specific materials in that category are available
        expanded_materials = available_materials.copy()
        for material in available_materials:
            for category, items in self.materials_db.items():
                if material in items and category not in expanded_materials:
                    expanded_materials.append(category)
        
        # Consider all difficulty levels if none specified
        if difficulty:
            difficulty_levels = [difficulty]
        else:
            difficulty_levels = self.project_templates.keys()
        
        # Check each project template
        for diff_level in difficulty_levels:
            for project in self.project_templates[diff_level]:
                # Check if all required materials are available
                required_met = True
                for req_material in project["required_materials"]:
                    # Handle category names (like "fabric") as well as specific materials
                    if req_material in self.materials_db:
                        # It's a category name
                        category_match = False
                        for material in expanded_materials:
                            if material in self.materials_db[req_material] or material == req_material:
                                category_match = True
                                break
                        if not category_match:
                            required_met = False
                            break
                    else:
                        # It's a specific material
                        if req_material not in expanded_materials:
                            # Check if an equivalent category is available
                            category = self.get_material_category(req_material)
                            if category and category in expanded_materials:
                                continue
                            required_met = False
                            break
                
                if required_met:
                    # Calculate how many optional materials are available
                    optional_available = 0
                    for opt_material in project["optional_materials"]:
                        if opt_material in expanded_materials:
                            optional_available += 1
                        else:
                            # Check if an equivalent category is available
                            category = self.get_material_category(opt_material)
                            if category and category in expanded_materials:
                                optional_available += 1
                    
                    # Add project to suitable list with a score based on optional materials
                    suitable_projects.append({
                        "name": project["name"],
                        "difficulty": diff_level,
                        "score": optional_available,
                        "required_materials": project["required_materials"],
                        "optional_materials": project["optional_materials"]
                    })
        
        # Sort by score (number of optional materials available)
        suitable_projects.sort(key=lambda x: x["score"], reverse=True)
        return suitable_projects

    def generate_project_instructions(self, project_name, available_materials, theme=None):
        """
        Generate detailed instructions for a specific project based on available materials
        """
        if project_name not in self.steps_templates:
            return f"Sorry, I don't have instructions for {project_name}."
        
        # Normalize material names
        available_materials = [m.lower().strip() for m in available_materials]
        
        # Process steps with conditionals and material references
        processed_steps = []
        step_number = 1
        
        for step in self.steps_templates[project_name]:
            # Process conditional steps
            if step.startswith("{if:"):
                condition = re.match(r"\{if: ([^}]+)\}", step).group(1)
                if condition not in available_materials:
                    # Skip this step if the condition material isn't available
                    continue
                # Remove the conditional marker for display
                step = re.sub(r"\{if: [^}]+\}", "", step)
            
            # Process material references
            material_refs = re.findall(r"\{material: ([^}]+)\}", step)
            for material_ref in material_refs:
                # If the specific material is available, use it
                if material_ref in available_materials:
                    step = step.replace(f"{{material: {material_ref}}}", material_ref)
                else:
                    # Try to find an alternative in the same category
                    category = self.get_material_category(material_ref)
                    if category:
                        alternatives = [m for m in self.materials_db[category] if m in available_materials]
                        if alternatives:
                            step = step.replace(f"{{material: {material_ref}}}", alternatives[0])
                            continue
                    
                    # If no alternative, just use the original term without formatting
                    step = step.replace(f"{{material: {material_ref}}}", material_ref)
            
            processed_steps.append(f"{step_number}. {step}")
            step_number += 1
        
        # Add theme-specific tips if a theme is specified
        theme_tips = []
        if theme:
            if theme in self.theme_details:
                if isinstance(self.theme_details[theme], dict):
                    # If it's a dictionary with seasons/holidays
                    theme_tip = f"Theme tips for {theme}:\n"
                    for subtheme, tips in self.theme_details[theme].items():
                        theme_tip += f"  - For {subtheme}: {', '.join(tips)}\n"
                    theme_tips.append(theme_tip)
                else:
                    # It's a list of tips
                    theme_tips.append(f"Theme tips for {theme}:\n  - " + "\n  - ".join(self.theme_details[theme]))
        
        # Combine all parts
        result = [
            f"# {project_name} Instructions",
            "",
            "## Materials You'll Need:",
            "- " + "\n- ".join([m for m in available_materials if m in self.flatten_materials_list()]),
            "",
            "## Step-by-Step Instructions:",
            "\n".join(processed_steps)
        ]
        
        if theme_tips:
            result.append("\n## Theme Customization:")
            result.extend(theme_tips)
        
        return "\n".join(result)

    def suggest_project(self, available_materials, theme=None, difficulty=None):
        """
        Suggest a suitable project based on available materials and preferences
        """
        suitable_projects = self.filter_projects_by_materials(available_materials, difficulty)
        
        if not suitable_projects:
            return "I couldn't find any suitable projects with your available materials. Try adding more basic craft supplies to your list."
        
        # If a theme is specified, try to prioritize projects that fit it well
        if theme:
            # Simple scoring - certain projects tend to work better with certain themes
            theme_affinities = {
                "seasonal": ["Decorative Jar", "Paper Garland", "Mixed Media Canvas"],
                "holiday": ["Paper Garland", "Rock Painting", "Paper Plate Craft"],
                "home decor": ["Decorative Jar", "Fabric Wall Hanging", "Woven Wall Art", "Mosaic Picture Frame", "Fabric Bowl"],
                "kids": ["Rock Painting", "Paper Plate Craft", "Upcycled Bottle Planter"],
                "gifts": ["Decorative Jar", "Rock Painting", "Fabric Bowl", "DIY Candle Holder"],
                "garden": ["Upcycled Bottle Planter", "Rock Painting", "Decorative Jar"],
                "organization": ["Fabric Bowl", "Decorative Jar", "Upcycled Bottle Planter"],
                "wearable": ["Paper Plate Craft"],
                "wall art": ["Woven Wall Art", "Mixed Media Canvas", "Paper Garland", "Fabric Wall Hanging"],
                "party decoration": ["Paper Garland", "Decorative Jar", "Paper Plate Craft"]
            }
            
            if theme in theme_affinities:
                # Boost score for projects that match the theme
                for project in suitable_projects:
                    if project["name"] in theme_affinities[theme]:
                        project["score"] += 5
                
                # Re-sort by updated scores
                suitable_projects.sort(key=lambda x: x["score"], reverse=True)
        
        # Get the top project
        top_project = suitable_projects[0]
        
        # Generate instructions for the top project
        instructions = self.generate_project_instructions(top_project["name"], available_materials, theme)
        
        return instructions

    def list_all_materials(self):
        """Return a formatted list of all available materials by category"""
        result = []
        for category, materials in self.materials_db.items():
            result.append(f"## {category.title()}")
            result.append("- " + "\n- ".join(materials))
            result.append("")
        
        return "\n".join(result)

    def list_all_themes(self):
        """Return a formatted list of all available themes"""
        return "Available themes:\n- " + "\n- ".join(self.themes)

    def list_all_projects(self):
        """Return a formatted list of all available project templates by difficulty"""
        result = []
        for difficulty, projects in self.project_templates.items():
            result.append(f"## {difficulty.title()} Projects")
            for project in projects:
                result.append(f"- {project['name']}")
                result.append(f"  - Required materials: {', '.join(project['required_materials'])}")
                result.append(f"  - Optional materials: {', '.join(project['optional_materials'])}")
            result.append("")
        
        return "\n".join(result)


# Command-line interface for the DIY Craft Generator
def main():
    generator = DIYCraftGenerator()
    
    print("🎨 Welcome to the DIY Craft Project Generator! 🎨")
    print("This tool will help you find the perfect craft project using materials you already have.")
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Get a project suggestion")
        print("2. See all available materials")
        print("3. See all available themes")
        print("4. See all project templates")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == "1":
            print("\nWhat materials do you have available? (comma-separated list)")
            materials_input = input("> ")
            materials = [m.strip() for m in materials_input.split(",")]
            
            print("\nDo you have a specific theme in mind? (optional)")
            print(f"Available themes: {', '.join(generator.themes)}")
            theme = input("> ").strip().lower() or None
            
            print("\nPreferred difficulty level? (easy, medium, advanced, or leave blank for any)")
            difficulty = input("> ").strip().lower() or None
            
            result = generator.suggest_project(materials, theme, difficulty)
            print("\n" + result)
            
        elif choice == "2":
            print("\nAvailable Materials:")
            print(generator.list_all_materials())
            
        elif choice == "3":
            print("\nAvailable Themes:")
            print(generator.list_all_themes())
            
        elif choice == "4":
            print("\nProject Templates:")
            print(generator.list_all_projects())
            
        elif choice == "5":
            print("\nThank you for using the DIY Craft Project Generator. Happy crafting!")
            break
            
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
