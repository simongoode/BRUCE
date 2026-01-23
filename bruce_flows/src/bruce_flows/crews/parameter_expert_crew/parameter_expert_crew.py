from typing import List

from crewai import Agent, Crew, LLM, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from bruce_flows.tools.read_analysis_tool import read_analysis_file
from bruce_flows.tools.write_file_tool import write_file

# Define the LLM to be used by the agents
# Using Gemini flash-2.5 for fast, cost-effective analysis
llm = LLM(model="google/gemini-2.5-flash", temperature=0.2)


@CrewBase
class ParameterExpertCrew:
    """Parameter Expert Crew for analyzing gravitational wave parameter estimation results"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def mass_expert(self) -> Agent:
        return Agent(
            config=self.agents_config["mass_expert"],  # type: ignore[index]
            tools=[read_analysis_file, write_file],
            llm=llm,
            verbose=True,
        )

    @task
    def analyze_mass_posteriors(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_mass_posteriors"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Parameter Expert Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
