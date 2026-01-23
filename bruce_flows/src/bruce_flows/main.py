#!/usr/bin/env python
from typing import List
from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from bruce_flows.crews.parameter_expert_crew.parameter_expert_crew import ParameterExpertCrew
from bruce_flows.tools.run_pe_tool import run_pe_script


class PEMassAnalysisState(BaseModel):
    round_number: int = 1
    previous_analyses: List[str] = []
    pe_report_path: str = "/home/sgoode/BRUCE/results/bruce_pe_report.md"
    max_rounds: int = 3


class PEMassAnalysisFlow(Flow[PEMassAnalysisState]):

    @start()
    def initialize_round(self, crewai_trigger_payload: dict = None):
        """Initialize the first round of analysis."""
        print(f"Initializing round {self.state.round_number}")
        self.state.round_number = 1
        self.state.previous_analyses = []
        self.state.max_rounds = 3

    @listen(initialize_round)
    def run_pe_script(self):
        """Run the parameter estimation script to generate a new PE report."""
        print(f"Running parameter estimation script for round {self.state.round_number}")
        result = run_pe_script("/home/sgoode/BRUCE/run_pe.py")
        print(f"PE script result: {result}")
        # The report should be at the standard location
        self.state.pe_report_path = "/home/sgoode/BRUCE/results/bruce_pe_report.md"

    @listen(run_pe_script)
    def analyze_mass_posteriors(self):
        """Have the mass expert analyze the PE report and write analysis."""
        print(f"Analyzing mass posteriors for round {self.state.round_number}")
        
        # Prepare inputs for the crew
        analysis_file_path = f"/home/sgoode/BRUCE/results/mass-expert-report-round-{self.state.round_number}.txt"
        
        # Build context with previous analyses
        previous_analyses_context = ""
        if self.state.previous_analyses:
            previous_analyses_context = "\nPrevious analysis files to review:\n"
            for prev_file in self.state.previous_analyses:
                previous_analyses_context += f"- {prev_file}\n"
        
        inputs = {
            "round_number": self.state.round_number,
            "pe_report_path": self.state.pe_report_path,
            "previous_analyses": previous_analyses_context,
            "analysis_file_path": analysis_file_path
        }
        
        result = (
            ParameterExpertCrew()
            .crew()
            .kickoff(inputs=inputs)
        )
        
        print(f"Mass expert analysis completed for round {self.state.round_number}")
        
        # Add this analysis to the list of previous analyses
        self.state.previous_analyses.append(analysis_file_path)

    @listen(analyze_mass_posteriors)
    def check_and_continue(self):
        """Check if we should continue to the next round."""
        print(f"Completed round {self.state.round_number} of {self.state.max_rounds}")
        
        if self.state.round_number < self.state.max_rounds:
            # Increment round number and trigger the next iteration
            self.state.round_number += 1
            print(f"Continuing to round {self.state.round_number}")
            # Directly call the next step to continue the loop
            self.run_pe_script()
        else:
            print(f"All {self.state.max_rounds} rounds completed!")


def kickoff():
    pe_flow = PEMassAnalysisFlow()
    pe_flow.kickoff()


def plot():
    pe_flow = PEMassAnalysisFlow()
    pe_flow.plot()


def run_with_trigger():
    """
    Run the flow with trigger payload.
    """
    import json
    import sys

    # Get trigger payload from command line argument
    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    # Create flow and kickoff with trigger payload
    # The @start() methods will automatically receive crewai_trigger_payload parameter
    pe_flow = PEMassAnalysisFlow()

    try:
        result = pe_flow.kickoff({"crewai_trigger_payload": trigger_payload})
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the flow with trigger: {e}")


if __name__ == "__main__":
    kickoff()
