import sys
import os

# ✅ Add the current directory to the Python path to enable relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ✅ Import CrewAI components
from crewai import  LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from settings import settings


from adaptiq.agents.crew_ai import create_crew_instrumental

crew_instrumental = create_crew_instrumental()

# ✅ Load environment variables
from dotenv import load_dotenv
load_dotenv()

llm = LLM(model="openai/gpt-4.1", api_key=settings.OPENAI_API_KEY)
from tools.describe_image_tool import DescribeImageTool
from tools.path_finder import PathFinderTool
from tools.pg_image_description import PGImageTool
from tools.validate_generated_prompt import PromptValidationTool

describe_tool = DescribeImageTool()
path_finder = PathFinderTool()
pg_image_description = PGImageTool()
validate_generated_prompt = PromptValidationTool()


@CrewBase
class GenericCrew():
    """🧠 Generic AI Crew
    A flexible blueprint for running AI agents on modular tasks.
    Replace tools, agents, and tasks as needed.
    """
    
    # 🔧 YAML configuration paths for agents and tasks
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    @crew_instrumental.agent_logger # ✅ Log reasoning, tool use, etc.
    @agent

    def prompt_engineer(self) -> Agent:
        return Agent(
			config=self.agents_config['prompt_engineer'],
            llm=llm,
            tools=[describe_tool, path_finder, pg_image_description, validate_generated_prompt],
			verbose=True
		)
    
    @crew_instrumental.task_logger  # ✅ Log task execution status
    @task
    
    def prompt_task(self) -> Task:
        return Task(
			config=self.tasks_config['prompt_task'],
			tools=[describe_tool, path_finder, pg_image_description, validate_generated_prompt]
		)
    
    @crew
    def crew(self) -> Crew:
        """👥 Assembles the agent-task pipeline as a Crew instance."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,  # 🔁 Change to Process.parallel if needed
            verbose=True
        )
