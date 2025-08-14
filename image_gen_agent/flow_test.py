import base64
import re
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
import time
from google import genai
from google.genai import types
from openai import OpenAI
import json
import yaml
from datetime import datetime
import os
import replicate 
from settings import settings
from crew_init import main

from metrics.clip_score import CLIPAnalysisResult, CLIPScoreCalculator
from metrics.percentile_score import PercentileScoreCalculator, PerformanceAnalysisResult
from metrics.llm_as_judge import EvaluationResult, LLMAsJudgeScore

class Prompt(BaseModel):
    # adaptiq or gpt
    agent_prompt_type: str
    prompt: str

class CrewMetric(BaseModel):
    execution_count: int
    total_executions: int
    start_timestamp: str
    end_timestamp: str
    execution_time_seconds: float
    execution_time_minutes: float
    current_memory_mb: float
    peak_memory_mb: float
    total_tokens: int
    total_cost: float
    prompt_tokens: int
    prompt_cost: float
    completion_tokens: int
    completion_cost:float
    cached_prompt_tokens: int
    cached_prompt_cost:float
    successful_requests: int
    models_used: list
    function_name: str

class ImagePrompt(BaseModel):
    # adaptiq or gpt
    original_image: str
    agent_prompt_type: str
    execution_time: float
    positive_prompt: str
    negative_prompt: str
    metrics: CrewMetric

class MetricsResults(BaseModel):
    latency: Optional[PerformanceAnalysisResult]
    tokens: Optional[PerformanceAnalysisResult]
    clip: Optional[List[CLIPAnalysisResult]]
    prompt_quality_score: Optional[EvaluationResult]

class FlowTest:

    def __init__(self):
        

        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

        self.category_content_step = 0
        self.category_content_count = 20
        self.clip_calculator = CLIPScoreCalculator()
        self.percentile_calculator = PercentileScoreCalculator()
        self.llm_as_judge = LLMAsJudgeScore(api_key=settings.OPENAI_API_KEY)
        
        self.input_token_cost = 0.0000003 
        self.cached_token_cost = 0.00000075
        self.output_token_cost = 0.000012

        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.image_refs: List[str] = ["pp", "na", "ca", "as", "ad"]
        self.agent_prompt_path = "./config/tasks.yaml"
        self.prompts_path: str = "./config/prompts.json"
        self.images_general_path: str = "./images"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_folder = os.path.join("test_results", timestamp)

        os.makedirs(base_folder, exist_ok=True)

        self.image_prompts_results: str = os.path.join(base_folder, "image_prompts.json")
        self.metrics_results: str = os.path.join(base_folder, "metrics.json")
        self.generated_images_path: str = os.path.join(base_folder, "images")

        os.makedirs(self.generated_images_path, exist_ok=True)
        


    def _load_prompts(self) -> Tuple[List[Prompt], str, str]:
        """Load prompts from JSON file

        returns:
            + List of prompts
            + Original prompt
            + Feedback
        """
        try:
            with open(self.prompts_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            prompts = []
            if isinstance(data, list):
                for item in data:
                    prompts.append(Prompt(**item))
            elif isinstance(data, dict) and 'prompts' in data:
                for item in data['prompts']:
                    prompts.append(Prompt(**item))
            else:
                print(f"❌ Error: Invalid JSON structure in {self.prompts_path}")
                return [], "", ""
                
            print(f"✅ Loaded {len(prompts)} prompts from {self.prompts_path}")
            return [prompts[data["adaptiq_used_prompt_index"]] ,prompts[data["gpt_used_prompt_index"]]], (data["initial_prompts"][data["original_prompt_index"]])["original_prompt"], (data["initial_prompts"][data["original_prompt_index"]])["feedback"]
            
        except FileNotFoundError:
            print(f"❌ Error: File {self.prompts_path} not found")
            return [], "", ""
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in {self.prompts_path}: {e}")
            return [], "", ""
        except Exception as e:
            print(f"❌ Error loading prompts: {e}")
            return [], "", ""

    def _inject_prompt(self, prompt: str):
        # Read the existing YAML file
        with open(self.agent_prompt_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        
        # Update the description field
        if 'prompt_task' in data and 'description' in data['prompt_task']:
            data['prompt_task']['description'] = prompt
            print(f"✅ Description updated successfully")
        else:
            print("❌ Error: 'prompt_task.description' field not found in YAML")
            return False
        
        # Write back to the file
        with open(self.agent_prompt_path, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        return True
    
    def _append_image_result(self, result: ImagePrompt):
        """Append image result to JSON file, create file if it doesn't exist"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.image_prompts_results), exist_ok=True)
            
            # Check if file exists and load existing data
            if os.path.exists(self.image_prompts_results):
                with open(self.image_prompts_results, 'r', encoding='utf-8') as file:
                    try:
                        data = json.load(file)
                        if not isinstance(data, dict) or 'results' not in data:
                            data = {'results': []}
                    except json.JSONDecodeError:
                        data = {'results': []}
            else:
                data = {'results': []}
            
            # Append new result
            data['results'].append(result.model_dump())
            
            # Write back to file
            with open(self.image_prompts_results, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
                
            print(f"✅ Result appended to {self.image_prompts_results}")
            
        except Exception as e:
            print(f"❌ Error appending result: {e}")
    
    def _load_images_prompts(self) -> List[ImagePrompt]:
        """Load image prompts from JSON results file"""
        try:
            with open(self.image_prompts_results, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            image_prompts = []
            if isinstance(data, dict) and 'results' in data:
                for item in data['results']:
                    image_prompts.append(ImagePrompt(**item))
            else:
                print(f"❌ Error: Invalid JSON structure in {self.image_prompts_results}")
                return []
                
            print(f"✅ Loaded {len(image_prompts)} image prompts from {self.image_prompts_results}")
            return image_prompts
            
        except FileNotFoundError:
            print(f"❌ Error: File {self.image_prompts_results} not found")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in {self.image_prompts_results}: {e}")
            return []
        except Exception as e:
            print(f"❌ Error loading image prompts: {e}")
            return []

    def _run_agent(self, inputs: Dict[str, str], agent_prompt_type: str, original_image: str) -> ImagePrompt:
        start_time = time.time()
        exec_results = main(inputs=inputs)
        end_time = time.time()
        duration = end_time - start_time

        # Extract original_result as string
        original_result = exec_results.get("original_result", "")

        # Extract positive and negative prompts using regex
        pp_match = re.search(r"<positive_prompt>(.*?)</positive_prompt>", original_result, re.DOTALL)
        np_match = re.search(r"<negative_prompt>(.*?)</negative_prompt>", original_result, re.DOTALL)

        pp = pp_match.group(1).strip() if pp_match else ""
        np = np_match.group(1).strip() if np_match else ""

        metrics_res =exec_results.get("crew_metrics", [])[-1]
        metrics_res["prompt_cost"] = self.input_token_cost * metrics_res["prompt_tokens"]
        metrics_res["completion_cost"] = self.output_token_cost * metrics_res["completion_tokens"]
        metrics_res["cached_prompt_cost"] = self.cached_token_cost * metrics_res["cached_prompt_tokens"]
        metrics_res["total_cost"] = metrics_res["prompt_cost"] +  metrics_res["completion_cost"] + metrics_res["cached_prompt_cost"]

        return ImagePrompt(
            agent_prompt_type=agent_prompt_type,
            execution_time=duration,
            original_image=original_image,
            positive_prompt=pp,
            negative_prompt=np,
            metrics =CrewMetric(** metrics_res)
        )

    def _gen_image(self, prompt: str) -> Tuple[bytes, str]:
        """Generate image using Google's Imagen model with retries."""
        max_retries = 3
        delay_seconds = 10

        for attempt in range(1, max_retries + 1):
            try:
                
                output = replicate.run(
                    "black-forest-labs/flux-1.1-pro",
                    input={
                        "prompt": prompt,
                        "aspect_ratio": "1:1",
                        "output_format": "jpg",
                        "output_quality": 80,
                        "safety_tolerance": 1,
                        "prompt_upsampling": True
                    }
                )


                return output.read(), "jpg"

            except Exception as e:
                if attempt < max_retries:
                    print(f"[Attempt {attempt}/{max_retries}] Error generating image: {e}")
                    print(f"Retrying in {delay_seconds} seconds...")
                    time.sleep(delay_seconds)
                else:
                    print(f"[Attempt {attempt}/{max_retries}] Failed after {max_retries} attempts.")
                    raise

    def _save_generated_images(self, image: bytes, folder: str, mime_type: str, image_name: str)-> str:
        """Save generated images to specified folder"""
        try:
            # Create main generated images folder if it doesn't exist
            os.makedirs(self.generated_images_path, exist_ok=True)
            
            # Create category folder within generated images path
            folder_path = os.path.join(self.generated_images_path, folder)
            os.makedirs(folder_path, exist_ok=True)
            
            # Determine file extension from mime type
            if 'jpeg' in mime_type or 'jpg' in mime_type:
                extension = '.jpg'
            elif 'png' in mime_type:
                extension = '.png'
            elif 'webp' in mime_type:
                extension = '.webp'
            else:
                extension = '.jpg'  # default fallback
            
            # Create full file path
            file_path = os.path.join(folder_path, f"{image_name}{extension}")
            
            # Save image
            with open(file_path, 'wb') as f:
                f.write(image)
                
            print(f"✅ Image saved to {file_path}")

            return file_path
            
        except Exception as e:
            print(f"❌ Error saving image {image_name}: {e}")
    
    def _save_metrics(self, results: MetricsResults):
        os.makedirs(os.path.dirname(self.metrics_results), exist_ok=True)

        with open(self.metrics_results, 'w', encoding='utf-8') as file:
            json.dump(results.model_dump(), file, indent=2, ensure_ascii=False)
                
    def _run_judge(self, original_prompt: str, adaptiq_prompt: str, gpt_prompt: str, enhancement_feedback: str) -> EvaluationResult:
        quality_score = self.llm_as_judge.prompt_quality_score(original_prompt= original_prompt, adaptiq_prompt= adaptiq_prompt, gpt_prompt= gpt_prompt, enhancement_feedback= enhancement_feedback)

        return quality_score

        
        
    def start(self):
        """Main execution method"""
        print("🚀 Starting FlowTest execution...")
        
        prompts, original_prompt, feeback = self._load_prompts()

        if not prompts:
            print("❌ No prompts loaded. Exiting...")
            return

        print(f"📋 Processing {len(self.image_refs)} categories with {self.category_content_count} images each")
        print(f"🔄 Using {len(prompts)} different prompts")
        
        # Phase 1: Generate prompts for all images
        total_combinations = len(self.image_refs) * self.category_content_count * len(prompts)
        current_combination = 0
        
        for category in self.image_refs:
            print(f"\n📁 Processing category: {category}")
            
            for index in range(self.category_content_step, (self.category_content_count + self.category_content_step)):
                image_path = f"{self.images_general_path}/{category}/{category}_{index+1}.jpg"
                image_name = f"{category}_{index+1}"
                print(f"  🖼️  Processing image: {category}_{index+1}")
                
                for prompt in prompts:
                    current_combination += 1
                    print(f"    🔄 Running prompt {prompt.agent_prompt_type} ({current_combination}/{total_combinations})")
                    
                    # Inject prompt into agent configuration
                    if not self._inject_prompt(prompt=prompt.prompt):
                        print(f"    ❌ Failed to inject prompt, skipping...")
                        continue
                    
                    # Run agent and get results
                    try:
                        res = self._run_agent(
                            inputs={"image_name": image_name}, 
                            agent_prompt_type=prompt.agent_prompt_type, 
                            original_image=image_name
                        )
                        self._append_image_result(result=res)
                        print(f"    ✅ Completed in {res.execution_time:.2f}s")
                    except Exception as e:
                        print(f"    ❌ Error running agent: {e}")

        print("\n🎨 Phase 1 complete - Starting Percentile Calculations...")
        
        # Phase 2: Generate images from prompts
        images_prompts = self._load_images_prompts()
        if not images_prompts:
            print("❌ No image prompts found. Exiting...")
            return
        
        adaptiq_data = [
            item.metrics.execution_time_seconds 
            for i, item in enumerate(images_prompts) if i % 2 == 0
        ]

        gpt_data = [
            item.metrics.execution_time_seconds 
            for i, item in enumerate(images_prompts) if i % 2 == 1
        ]

        latency_results = self.percentile_calculator.analyze_performance(
            adaptiq_data=adaptiq_data,
            gpt_data=gpt_data,
            metric_name="Latency"
        )

        adaptiq_data = [
            item.metrics.total_tokens
            for i, item in enumerate(images_prompts) if i % 2 == 0
        ]

        gpt_data = [
            item.metrics.total_tokens
            for i, item in enumerate(images_prompts) if i % 2 == 1
        ]

        tokens_results = self.percentile_calculator.analyze_performance(
            adaptiq_data=adaptiq_data,
            gpt_data=gpt_data,
            metric_name="Tokens"
        )

        print("\n🎨 Phase 1.1 complete - Starting image generation...")
        
        print(f"🖼️  Generating images for {len(images_prompts)} prompts...")

        clip_results: List[CLIPAnalysisResult] = []
        
        # Process images in batches of step (2 prompts × category_content_count images per category)
        step = self.category_content_count * 2
        for index in range(0, len(images_prompts), step):
            try:
                # Calculate which category this batch belongs to
                category_index = index // step
                if category_index >= len(self.image_refs):
                    break
                    
                current_save_folder = self.image_refs[category_index]
                print(f"\n📁 Generating images for category: {current_save_folder}")
                
                # Process batch (up to step images)
                batch_end = min(index + step, len(images_prompts))
                counter = 0
                error_occur = False
                generated_images_paths: List[str] = []
                combined_promtps: List[str] = []
                for i in range(index, batch_end):
                    counter+=1
                    image_prompt = images_prompts[i]
                    try:
                        print(f"  🎨 Generating image {i+1}/{len(images_prompts)}: {image_prompt.original_image}_{image_prompt.agent_prompt_type}")
                        
                        # Generate image
                        combined_prompt = f"Positive prompt: {image_prompt.positive_prompt}"
                        if image_prompt.negative_prompt:
                            combined_prompt += f"\nNegative Prompt: {image_prompt.negative_prompt}"
                        
                        combined_promtps.append(combined_prompt)
                        image, mime = self._gen_image(prompt=combined_prompt)
                        
                        # Save image
                        generated_path = self._save_generated_images(
                            image=image, 
                            mime_type=mime, 
                            image_name=f"{image_prompt.original_image}_{image_prompt.agent_prompt_type}", 
                            folder=current_save_folder
                        )
                        generated_images_paths.append(generated_path)

                        if counter == 2:
                            

                            res = self.clip_calculator.analyze_clip_score(
                                original_image_path= f"{self.images_general_path}/{image_prompt.original_image.split("_")[0]}/{image_prompt.original_image}.jpg",
                                adaptiq_image_path= generated_images_paths[0], 
                                adaptiq_text_prompt= combined_promtps[0],
                                gpt_image_path=generated_images_paths[1],
                                gpt_text_prompt=combined_promtps[1]
                                )
                            
                            counter = 0
                            generated_images_paths.clear()
                            combined_promtps.clear()
                            error_occur = False

                            clip_results.append(res)

                        elif counter == 1 and error_occur:
                            counter = 0
                            generated_images_paths.clear()
                            combined_promtps.clear()
                            error_occur = False


                        time.sleep(10)
                        
                    except Exception as e:
                        counter = 0
                        generated_images_paths.clear()
                        combined_promtps.clear()
                        error_occur = True
                        print(f"  ❌ Error generating image for {image_prompt.original_image}: {e}")
                        
            except Exception as e:
                print(f"❌ Error processing batch starting at index {index}: {e}")

        prompt_quality_score = self._run_judge(original_prompt= original_prompt, adaptiq_prompt= prompts[0].prompt, gpt_prompt= prompts[1].prompt, enhancement_feedback= feeback)

        metrics_results = MetricsResults(latency=latency_results, tokens=tokens_results, clip=clip_results, prompt_quality_score=prompt_quality_score)

        self._save_metrics(results=metrics_results)
        print("\n🎉 FlowTest execution completed!")
        print(f"📊 Results saved to: {self.image_prompts_results}")
        print(f"🖼️  Generated images saved to: {self.generated_images_path}")

        


# Example usage
if __name__ == "__main__":
    flow_test = FlowTest()
    flow_test.start()