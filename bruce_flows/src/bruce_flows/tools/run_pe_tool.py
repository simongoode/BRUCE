import subprocess
import os
import sys
from crewai.tools import tool


def _run_pe_script_impl(script_path: str = None) -> str:
    """
    Internal implementation for running the parameter estimation script.
    This can be called directly from flow code.
    Assumes cwd is bruce_flows/ (i.e., run via 'crewai run' from bruce_flows/).
    """
    try:
        # If no script path provided, use default relative path from bruce_flows/ cwd
        if script_path is None:
            script_path = "src/scripts/run_pe.py"
        
        # Ensure the script path is valid
        if not os.path.exists(script_path):
            return f"Error: Script not found at path: {script_path}"

        # Execute the script using subprocess
        # Use the current Python interpreter and explicitly set PYTHONPATH
        env = os.environ.copy()
        
        # Ensure PYTHONPATH includes the directories where packages are installed
        python_paths = [
            '/app/bruce_flows/src',
            '/app',
            '/usr/local/lib/python3.11/dist-packages',
        ]
        
        # Add existing PYTHONPATH if present
        if 'PYTHONPATH' in env:
            python_paths.append(env['PYTHONPATH'])
        
        env['PYTHONPATH'] = ':'.join(python_paths)
        
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            env=env
        )

        # Check if the expected output file was created
        # Path relative to bruce_flows/ cwd
        expected_output = "results/bruce_pe_report.md"
        
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
def run_pe_script(script_path: str = None) -> str:
    """
    Executes the parameter estimation script (run_pe.py) to generate a PE report.
    The script generates a markdown report at 'results/bruce_pe_report.md'.
    
    Assumes cwd is bruce_flows/ (i.e., run via 'crewai run' from bruce_flows/).

    Args:
        script_path (str, optional): The path to the run_pe.py script relative to bruce_flows/.
                          If None, defaults to 'src/scripts/run_pe.py'.

    Returns:
        str: Success message with the path to the generated report, or an error message
             if the script fails.
    """
    return _run_pe_script_impl(script_path)
