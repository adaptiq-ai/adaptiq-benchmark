from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os

class PathFinderInput(BaseModel):
    """Input schema for PathFinderTool."""
    image_name: str = Field(..., description="Name of the image")


class PathFinderTool(BaseTool):
    name: str = "get_image_path"
    description: str = (
        "This tool uses the image name to look for the correct path"
    )
    args_schema: Type[BaseModel] = PathFinderInput

    def _run(self, image_name: str) -> str:

        path_prefix = "./images"
        folder_name = image_name.split("_")[0]
        full_path = f"{path_prefix}/{folder_name}/{image_name}.jpg"

        if not os.path.exists(full_path):
            return f"Failed to find image with the name {image_name}"
        
        return  f"Found path: {full_path}"
