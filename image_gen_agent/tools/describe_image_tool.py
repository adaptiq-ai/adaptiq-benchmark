from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os
from pathlib import Path
import json


class DescribeImageInput(BaseModel):
    """Input schema for DescribeImageTool."""
    image_path: str = Field(..., description="Path to the image to describe.")


class DescribeImageTool(BaseTool):
    name: str = "describe_image"
    description: str = (
        "This tool takes the local image file path and returns a detailed description using Vision model."
    )
    args_schema: Type[BaseModel] = DescribeImageInput

    def _run(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            return f"Image not found at: {image_path}"

        try:
            image_name = Path(image_path).stem

            # Safe split: gets last "_" as index separator
            if "_" not in image_name:
                return f"Invalid image name format: '{image_name}'. Expected format 'category_index'."

            category, index = image_name.rsplit("_", 1)

            with open("./config/images_description.json", "r", encoding="utf-8") as json_file:
                images_description = json.load(json_file)

            if category not in images_description["images"]:
                return f"Category '{category}' not found in image descriptions."

            if index not in images_description["images"][category]:
                return f"Index '{index}' not found under category '{category}'."

            return images_description["images"][category][index]
        
        except Exception as e:
            return f"Error processing image: {e}"
