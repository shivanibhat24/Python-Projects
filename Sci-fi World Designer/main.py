"""
Sci-Fi World Designer
=====================
A comprehensive tool to generate rich sci-fi worlds from a simple story seed.
Uses LLaMA 3 for text generation, Stable Diffusion for imagery,
LangChain for workflow orchestration, and DungeonMap for map creation.
"""

import os
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# LangChain imports
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Stable Diffusion imports
from diffusers import StableDiffusionPipeline
import torch

# For map generation
import dungeonmap as dm

# Configuration
CONFIG = {
    "llama_model_path": "models/llama-3-8b-instruct.gguf",  # Path to LLaMA 3 model
    "output_dir": "generated_worlds",
    "stable_diffusion_model": "stabilityai/stable-diffusion-2-1",
}

class SciFiWorldDesigner:
    """Main class for the Sci-Fi World Designer."""
    
    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        """Initialize the world designer with configuration."""
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        
        # Initialize components
        self.llm = self._initialize_llm()
        self.image_generator = self._initialize_image_generator()
        self.world_data = {}
        
        print("✨ Sci-Fi World Designer initialized and ready for creation!")
    
    def _initialize_llm(self) -> LlamaCpp:
        """Initialize the LLaMA 3 model."""
        print("Initializing LLaMA 3 model...")
        return LlamaCpp(
            model_path=self.config["llama_model_path"],
            temperature=0.7,
            max_tokens=2048,
            verbose=True,
        )
    
    def _initialize_image_generator(self) -> StableDiffusionPipeline:
        """Initialize Stable Diffusion for image generation."""
        print("Initializing Stable Diffusion...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionPipeline.from_pretrained(
            self.config["stable_diffusion_model"], 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe = pipe.to(device)
        return pipe
    
    def generate_world_from_seed(self, story_seed: str, world_name: str = None) -> Dict[str, Any]:
        """Generate a complete sci-fi world from a story seed."""
        print(f"🌌 Generating sci-fi world from seed: '{story_seed}'")
        
        # Generate or use provided world name
        if not world_name:
            world_name = self._generate_world_name(story_seed)
        
        # Create world directory
        world_dir = self.output_dir / world_name
        world_dir.mkdir(exist_ok=True)
        
        # Generate core world elements
        world_concept = self._generate_world_concept(story_seed)
        civilization_data = self._generate_civilizations(world_concept)
        history_timeline = self._generate_history(world_concept, civilization_data)
        locations = self._generate_locations(world_concept, civilization_data)
        technology = self._generate_technology(world_concept)
        
        # Generate maps
        star_map = self._generate_star_map(world_concept, locations)
        planet_maps = self._generate_planet_maps(locations)
        
        # Compile all data
        self.world_data = {
            "name": world_name,
            "seed": story_seed,
            "concept": world_concept,
            "civilizations": civilization_data,
            "history": history_timeline,
            "locations": locations,
            "technology": technology,
            "maps": {
                "star_map": star_map,
                "planet_maps": planet_maps
            }
        }
        
        # Save world data
        self._save_world_data(world_dir)
        
        # Generate illustrations
        self._generate_illustrations(world_dir)
        
        print(f"✅ World '{world_name}' successfully generated and saved to {world_dir}")
        return self.world_data
    
    def _generate_world_name(self, story_seed: str) -> str:
        """Generate a fitting name for the sci-fi world."""
        prompt = PromptTemplate(
            input_variables=["seed"],
            template="""Generate a unique and memorable name for a science fiction 
            universe based on this story seed: {seed}. 
            Only respond with the name itself, no explanation."""
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        name = chain.run(seed=story_seed).strip()
        print(f"🏷️ Generated world name: {name}")
        return name
    
    def _generate_world_concept(self, story_seed: str) -> Dict[str, Any]:
        """Generate the core concept of the sci-fi world."""
        # Define output schemas
        response_schemas = [
            ResponseSchema(name="setting", description="The time period and physical setting of the world"),
            ResponseSchema(name="theme", description="The central theme or themes explored in this world"),
            ResponseSchema(name="tone", description="The overall tone and mood of the world"),
            ResponseSchema(name="unique_feature", description="A defining unique feature of this sci-fi universe"),
            ResponseSchema(name="physics", description="Any unique physics, dimensionality, or natural laws")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        
        prompt = PromptTemplate(
            input_variables=["seed", "format_instructions"],
            template="""Create a comprehensive science fiction world concept based on the 
            following story seed: {seed}
            
            Develop a rich, unique, and internally consistent universe.
            
            {format_instructions}"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(seed=story_seed, format_instructions=format_instructions)
        
        try:
            parsed_result = output_parser.parse(result)
            print(f"🌍 Generated world concept with theme: {parsed_result['theme']}")
            return parsed_result
        except Exception as e:
            print(f"Error parsing world concept: {e}")
            # Fallback to raw result
            return {"raw_concept": result}
    
    def _generate_civilizations(self, world_concept: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate details about the civilizations in the world."""
        response_schemas = [
            ResponseSchema(name="civilizations", description="A list of 2-5 civilizations, each with name, description, social_structure, technology_level, and values")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        
        prompt = PromptTemplate(
            input_variables=["concept", "format_instructions"],
            template="""Based on this sci-fi world concept:
            Setting: {concept['setting']}
            Theme: {concept['theme']}
            Unique Feature: {concept['unique_feature']}
            
            Generate 2-5 distinct civilizations or species that inhabit this universe.
            Each civilization should have internal consistency and reflect the world's themes.
            
            {format_instructions}"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(concept=world_concept, format_instructions=format_instructions)
        
        try:
            parsed_result = output_parser.parse(result)
            civs = parsed_result.get("civilizations", [])
            if isinstance(civs, str):
                # If it returned a string instead of a list, try to parse it as JSON
                civs = json.loads(civs)
            print(f"👥 Generated {len(civs)} civilizations")
            return civs
        except Exception as e:
            print(f"Error parsing civilizations: {e}")
            # Fallback
            return [{"name": "Unknown Civilization", "description": result}]
    
    def _generate_history(self, world_concept: Dict[str, Any], civilizations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate a historical timeline for the world."""
        civ_names = [civ["name"] for civ in civilizations]
        
        prompt = PromptTemplate(
            input_variables=["concept", "civilizations"],
            template="""Create a chronological timeline of 7-10 major historical events for a sci-fi universe with:
            Setting: {concept['setting']}
            Theme: {concept['theme']}
            
            These civilizations exist in this universe: {civilizations}
            
            Format each event as:
            {{
                "year": "(relative or absolute year)",
                "event": "(name of the event)",
                "description": "(description of what happened)",
                "impact": "(how this shaped the universe)"
            }}
            
            Return the events as a valid JSON list."""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(concept=world_concept, civilizations=civ_names)
        
        try:
            # Extract JSON list from result
            start_idx = result.find('[')
            end_idx = result.rfind(']') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = result[start_idx:end_idx]
                timeline = json.loads(json_str)
                print(f"📜 Generated timeline with {len(timeline)} historical events")
                return timeline
            else:
                raise ValueError("Could not find JSON list in result")
        except Exception as e:
            print(f"Error parsing timeline: {e}")
            # Fallback
            return [{"year": "Unknown", "event": "Historical Records", "description": result}]
    
    def _generate_locations(self, world_concept: Dict[str, Any], civilizations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate key locations in the sci-fi world."""
        civ_names = [civ["name"] for civ in civilizations]
        
        prompt = PromptTemplate(
            input_variables=["concept", "civilizations"],
            template="""Create 5-8 significant locations in a sci-fi universe with:
            Setting: {concept['setting']}
            Theme: {concept['theme']}
            
            These civilizations exist in this universe: {civilizations}
            
            For each location, include:
            {{
                "name": "(name of the location)",
                "type": "(planet, space station, dimension, city, etc.)",
                "description": "(physical description and significance)",
                "inhabitants": "(who lives here)",
                "points_of_interest": ["(list 2-4 specific sites within this location)"]
            }}
            
            Return the locations as a valid JSON list."""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(concept=world_concept, civilizations=civ_names)
        
        try:
            # Extract JSON list from result
            start_idx = result.find('[')
            end_idx = result.rfind(']') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = result[start_idx:end_idx]
                locations = json.loads(json_str)
                print(f"🏙️ Generated {len(locations)} locations")
                return locations
            else:
                raise ValueError("Could not find JSON list in result")
        except Exception as e:
            print(f"Error parsing locations: {e}")
            # Fallback
            return [{"name": "Unknown Location", "description": result}]
    
    def _generate_technology(self, world_concept: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate key technologies in the sci-fi world."""
        prompt = PromptTemplate(
            input_variables=["concept"],
            template="""Create 5-7 significant technologies for a sci-fi universe with:
            Setting: {concept['setting']}
            Theme: {concept['theme']}
            Unique Feature: {concept['unique_feature']}
            
            For each technology, include:
            {{
                "name": "(name of the technology)",
                "category": "(energy, transportation, communication, weapons, etc.)",
                "description": "(how it works and what it does)",
                "impact": "(how this technology affects society)",
                "limitations": "(what are its drawbacks or constraints)"
            }}
            
            Return the technologies as a valid JSON list."""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(concept=world_concept)
        
        try:
            # Extract JSON list from result
            start_idx = result.find('[')
            end_idx = result.rfind(']') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = result[start_idx:end_idx]
                technologies = json.loads(json_str)
                print(f"🔧 Generated {len(technologies)} technologies")
                return technologies
            else:
                raise ValueError("Could not find JSON list in result")
        except Exception as e:
            print(f"Error parsing technologies: {e}")
            # Fallback
            return [{"name": "Unknown Technology", "description": result}]
    
    def _generate_star_map(self, world_concept: Dict[str, Any], locations: List[Dict[str, Any]]) -> str:
        """Generate a star map of the sci-fi universe."""
        print("🌠 Generating star map...")
        
        # Extract planet/star systems from locations
        star_systems = [loc for loc in locations if loc.get("type", "").lower() in 
                       ["planet", "star system", "star", "solar system"]]
        
        # Use DungeonMap to create a star map
        try:
            # Basic star map implementation
            map_width, map_height = 800, 600
            star_map = dm.Map(map_width, map_height, background_color=(0, 0, 30))
            
            # Add stars and connections
            for i, system in enumerate(star_systems[:10]):  # Limit to 10 systems
                x = random.randint(50, map_width - 50)
                y = random.randint(50, map_height - 50)
                
                # Star size based on importance
                size = random.randint(3, 8)
                
                # Star color varies
                colors = [(255, 255, 200), (200, 200, 255), (255, 200, 200), 
                          (200, 255, 255), (255, 230, 180)]
                color = random.choice(colors)
                
                # Add star
                star_map.add_circle(x, y, size, color)
                
                # Add label
                star_map.add_text(system["name"], x + 10, y - 5, color=(200, 200, 200), size=10)
            
            # Save map
            map_path = str(self.output_dir / f"{world_concept.get('name', 'universe')}_star_map.png")
            star_map.save(map_path)
            return map_path
            
        except Exception as e:
            print(f"Error generating star map: {e}")
            return "Star map generation failed"
    
    def _generate_planet_maps(self, locations: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate maps for major planets and locations."""
        planet_maps = {}
        
        planets = [loc for loc in locations if loc.get("type", "").lower() in 
                  ["planet", "moon", "asteroid", "megastructure"]]
        
        print(f"🪐 Generating maps for {len(planets)} planets/locations...")
        
        for planet in planets[:3]:  # Limit to first 3 planets to save time
            try:
                # Create a map specific to the planet type
                map_width, map_height = 600, 600
                
                if "asteroid" in planet.get("type", "").lower():
                    planet_map = dm.Map(map_width, map_height, background_color=(50, 50, 50))
                    dm.generators.cavern_generator(planet_map, num_caverns=random.randint(3, 8))
                
                elif "megastructure" in planet.get("type", "").lower():
                    planet_map = dm.Map(map_width, map_height, background_color=(30, 30, 40))
                    dm.generators.structure_generator(planet_map, rooms=random.randint(15, 25))
                
                else:  # Regular planet
                    planet_map = dm.Map(map_width, map_height, background_color=(20, 60, 20))
                    dm.generators.continent_generator(planet_map, num_continents=random.randint(1, 5))
                
                # Add points of interest
                for poi in planet.get("points_of_interest", []):
                    x = random.randint(50, map_width - 50)
                    y = random.randint(50, map_height - 50)
                    planet_map.add_circle(x, y, 5, (255, 200, 50))
                    planet_map.add_text(poi, x + 10, y, color=(230, 230, 230), size=10)
                
                # Save map
                map_path = str(self.output_dir / f"{planet['name'].replace(' ', '_')}_map.png")
                planet_map.save(map_path)
                planet_maps[planet["name"]] = map_path
                
            except Exception as e:
                print(f"Error generating map for {planet['name']}: {e}")
                planet_maps[planet["name"]] = "Map generation failed"
        
        return planet_maps
    
    def _generate_illustrations(self, world_dir: Path) -> None:
        """Generate illustrations for the world using Stable Diffusion."""
        print("🎨 Generating illustrations...")
        
        try:
            # Generate world overview illustration
            world_prompt = f"Science fiction universe: {self.world_data['concept']['setting']}, {self.world_data['concept']['theme']}, highly detailed digital painting, epic scale"
            self._generate_image(world_prompt, world_dir / "world_overview.png")
            
            # Generate a civilization illustration
            if self.world_data["civilizations"]:
                civ = self.world_data["civilizations"][0]
                civ_prompt = f"Science fiction {civ['name']} civilization: {civ['description']}, detailed digital art"
                self._generate_image(civ_prompt, world_dir / f"{civ['name'].replace(' ', '_')}_illustration.png")
            
            # Generate a technology illustration
            if self.world_data["technology"]:
                tech = self.world_data["technology"][0]
                tech_prompt = f"Futuristic {tech['name']} technology: {tech['description']}, sci-fi concept art, detailed"
                self._generate_image(tech_prompt, world_dir / f"{tech['name'].replace(' ', '_')}_illustration.png")
            
        except Exception as e:
            print(f"Error generating illustrations: {e}")
    
    def _generate_image(self, prompt: str, output_path: str) -> None:
        """Generate an image using Stable Diffusion."""
        try:
            print(f"Generating image: {prompt[:50]}...")
            image = self.image_generator(prompt, guidance_scale=7.5).images[0]
            image.save(output_path)
            print(f"Image saved to {output_path}")
        except Exception as e:
            print(f"Error generating image: {e}")
    
    def _save_world_data(self, world_dir: Path) -> None:
        """Save all generated world data to JSON."""
        # Save main world data
        with open(world_dir / "world_data.json", "w") as f:
            json.dump(self.world_data, f, indent=2)
        
        # Save individual components for easier access
        components = {
            "concept": self.world_data["concept"],
            "civilizations": self.world_data["civilizations"],
            "history": self.world_data["history"],
            "locations": self.world_data["locations"],
            "technology": self.world_data["technology"]
        }
        
        for name, data in components.items():
            with open(world_dir / f"{name}.json", "w") as f:
                json.dump(data, f, indent=2)
        
        # Generate a human-readable world guide
        self._generate_world_guide(world_dir)
    
    def _generate_world_guide(self, world_dir: Path) -> None:
        """Generate a human-readable markdown world guide."""
        world_name = self.world_data["name"]
        
        guide = f"""# {world_name} - World Guide

## Overview
{self.world_data['concept']['setting']}

**Theme:** {self.world_data['concept']['theme']}  
**Tone:** {self.world_data['concept']['tone']}  
**Unique Feature:** {self.world_data['concept']['unique_feature']}

## Civilizations

"""
        
        for civ in self.world_data["civilizations"]:
            guide += f"### {civ['name']}\n"
            guide += f"{civ['description']}\n\n"
            guide += f"**Social Structure:** {civ.get('social_structure', 'Unknown')}\n"
            guide += f"**Technology Level:** {civ.get('technology_level', 'Unknown')}\n"
            guide += f"**Values:** {civ.get('values', 'Unknown')}\n\n"
        
        guide += "## Historical Timeline\n\n"
        
        for event in self.world_data["history"]:
            guide += f"### {event['year']} - {event['event']}\n"
            guide += f"{event['description']}\n\n"
            guide += f"**Impact:** {event['impact']}\n\n"
        
        guide += "## Key Locations\n\n"
        
        for location in self.world_data["locations"]:
            guide += f"### {location['name']} ({location.get('type', 'Location')})\n"
            guide += f"{location['description']}\n\n"
            guide += f"**Inhabitants:** {location.get('inhabitants', 'Unknown')}\n\n"
            
            if location.get('points_of_interest'):
                guide += "**Points of Interest:**\n"
                for poi in location['points_of_interest']:
                    guide += f"- {poi}\n"
                guide += "\n"
        
        guide += "## Technology\n\n"
        
        for tech in self.world_data["technology"]:
            guide += f"### {tech['name']} ({tech.get('category', 'Technology')})\n"
            guide += f"{tech['description']}\n\n"
            guide += f"**Impact:** {tech.get('impact', 'Unknown')}\n"
            guide += f"**Limitations:** {tech.get('limitations', 'Unknown')}\n\n"
        
        # Save guide
        with open(world_dir / "world_guide.md", "w") as f:
            f.write(guide)


def main():
    """Main entry point for the Sci-Fi World Designer."""
    parser = argparse.ArgumentParser(description="Generate a sci-fi world from a story seed.")
    parser.add_argument("--seed", type=str, help="Story seed to generate the world from")
    parser.add_argument("--name", type=str, help="Name for the generated world", default=None)
    parser.add_argument("--random_seed", type=int, help="Random seed for reproducibility", default=None)
    args = parser.parse_args()
    
    # Default story seed if none provided
    if not args.seed:
        args.seed = "A post-singularity civilization discovers ancient alien ruins that suggest a cyclical pattern of technological advancement and collapse across the galaxy."
    
    # Initialize and run the world designer
    designer = SciFiWorldDesigner(CONFIG, seed=args.random_seed)
    world = designer.generate_world_from_seed(args.seed, args.name)
    
    print("\n" + "="*50)
    print(f"World '{world['name']}' has been generated!")
    print(f"Output directory: {os.path.join(CONFIG['output_dir'], world['name'])}")
    print("="*50)


if __name__ == "__main__":
    main()
