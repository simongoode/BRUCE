# In file: tools/plot_generator_tool.py

import subprocess
import os, sys
from crewai.tools import tool

@tool("Corner Plot Generation Tool")
def generate_plot(script_path: str) -> str:
    """
    Executes a specified Python script to generate a corner plot for a
    gravitational-wave event. The script is expected to print the absolute
    file path of the generated image to standard output upon successful
    completion.

    Args:
        script_path (str): The relative or absolute path to the Python
                           script to be executed.

    Returns:
        str: The absolute file path of the generated image if successful,
             or an error message string if the script fails.
    """
    try:
        # Ensure the script path is valid
        if not os.path.exists(script_path):
            return f"Error: Script not found at path: {script_path}"

        # Execute the script using subprocess. The 'python' command
        # should be in the system's PATH.
        # capture_output=True captures stdout and stderr.
        # text=True decodes them as text.
        # check=True raises a CalledProcessError if the script returns a non-zero exit code.
        result = subprocess.run(
            ['/home/sgoode/BRUCE/.venv/bin/python', script_path],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )

        full_output = result.stdout.strip()
        if not full_output:
            return "Error: The script ran but produced no output."

        image_path = full_output.splitlines()[-1].strip()

        # Verify that the path returned by the script actually points to a file.
        if not os.path.isfile(image_path):
            return (f"Error: The script ran but did not return a valid file path. "
                    f"Script output: '{image_path}'")

        return image_path

    except subprocess.CalledProcessError as e:
        # This block catches errors if the script exits with a non-zero status.
        error_message = e.stderr.strip()
        return (f"Error executing script '{script_path}'. "
                f"Return code: {e.returncode}. "
                f"Error output: {error_message}")

    except Exception as e:
        # This is a general catch-all for other unexpected errors.
        return f"An unexpected error occurred: {str(e)}"