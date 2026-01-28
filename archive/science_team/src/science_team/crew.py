from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import os

# It's good practice to load environment variables at the start.
from dotenv import load_dotenv
load_dotenv()

# Define the LLM to be used by the agents.
# The Analyst agent requires a powerful multimodal model.
# gpt-4o-mini is specified here (supports vision). Ensure OPENAI_API_KEY is in your .env file.
# For free tier, use gpt-4o-mini. For better quality, use gpt-4o.
llm = LLM(model="openai/gpt-4o-mini", temperature=0.2)

# Import the custom tool for the generator agent.
from science_team.tools.plot_generator_tool import generate_plot

@CrewBase
class ScientificAnalysisCrew:
    """ScientificAnalysisCrew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def plot_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['plot_generator'],
            # The generator agent does not need a powerful LLM,
            # but we can assign one for consistency.
            # It primarily relies on its tool.
            llm=llm,
            tools=[generate_plot],
            verbose=True
        )

    @agent
    def gw_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['gw_analyst'],
            # The analyst MUST be assigned a multimodal-capable LLM.
            llm=llm,
            verbose=True
        )

    @task
    def generation_task(self) -> Task:
        return Task(
            config=self.tasks_config['generation_task'],
            agent=self.plot_generator(),
            markdown=False
        )

    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['analysis_task'],
            agent=self.gw_analyst(),
            # This is the critical link: the output of generation_task
            # becomes the context for analysis_task.
            context=[self.generation_task()]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the scientific analysis crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )