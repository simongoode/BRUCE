import subprocess
import os
from crewai.tools import tool


def _run_pe_script_impl(script_path: str = "/home/sgoode/BRUCE/run_pe.py") -> str:
    """
    Internal implementation for running the parameter estimation script.
    This can be called directly from flow code.
    """
    try:
        # Ensure the script path is valid
        if not os.path.exists(script_path):
            return f"Error: Script not found at path: {script_path}"

        # Execute the script using subprocess
        # Use the virtual environment Python interpreter
        result = subprocess.run(
            ['/home/sgoode/BRUCE/.venv/bin/python', script_path],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )

        # Check if the expected output file was created
        expected_output = '/home/sgoode/BRUCE/results/bruce_pe_report.md'
        if os.path.exists(expected_output):
            return f"Successfully ran parameter estimation script. Report generated at: {expected_output}"
        else:
            return (f"Script executed but expected output file not found at {expected_output}. "
                    f"Script output: {result.stdout.strip()}")

    except subprocess.CalledProcessError as e:
        # This block catches errors if the script exits with a non-zero status.
        error_message = e.stderr.strip() if e.stderr else "No error message available"
        return (f"Error executing script '{script_path}'. "
                f"Return code: {e.returncode}. "
                f"Error output: {error_message}")

    except Exception as e:
        # This is a general catch-all for other unexpected errors.
        return f"An unexpected error occurred: {str(e)}"


@tool("Parameter Estimation Script Runner")
def run_pe_script(script_path: str = "/home/sgoode/BRUCE/run_pe.py") -> str:
    """
    Executes the parameter estimation script (run_pe.py) to generate a PE report.
    The script generates a markdown report at '/home/sgoode/BRUCE/results/bruce_pe_report.md'.

    Args:
        script_path (str): The path to the run_pe.py script. Defaults to
                          '/home/sgoode/BRUCE/run_pe.py'.

    Returns:
        str: Success message with the path to the generated report, or an error message
             if the script fails.
    """
    return _run_pe_script_impl(script_path)
