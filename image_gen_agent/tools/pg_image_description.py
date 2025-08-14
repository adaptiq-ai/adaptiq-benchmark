from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os
from pathlib import Path
import json


class PGImageInput(BaseModel):
    """Input schema for PGImageTool."""
    image_path: str = Field(..., description="Path to the image to describe.")


class PGImageTool(BaseTool):

    name: str = "get_image_description"
    description: str = (
        "This tool uses the image path to look for any saved description in the database"
    )
    args_schema: Type[BaseModel] = PGImageInput


    def _run(self, image_path: str) -> str:

        if not os.path.exists(image_path):
            return f"Image not found at: {image_path}"

        return f"Failed to get image description"
