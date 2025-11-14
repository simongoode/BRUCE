#!/usr/bin/env python
from science_team.crew import ScientificAnalysisCrew

def run():
    """
    Run the scientific analysis crew.
    This function can be parameterized to run analyses for different events.
    """
    # For this PoC, inputs are not strictly necessary as the script path
    # is hardcoded in the task. For a production system, you might pass
    # a data file path or event ID here.
    inputs = {
        'event_id': 'GW150914' # Example input, not used in current task config
    }
    ScientificAnalysisCrew().crew().kickoff(inputs=inputs)

if __name__ == "__main__":
    run()