import os
from crewai.tools import tool


@tool("Read Analysis File Tool")
def read_analysis_file(file_path: str) -> str:
    """
    Reads the contents of a previous analysis file so the agent can recall
    insights from previous rounds.

    Args:
        file_path (str): The absolute path to the analysis file to read.

    Returns:
        str: The contents of the file, or an error message if the file cannot be read.
    """
    try:
        # Ensure the file path is valid
        if not os.path.exists(file_path):
            return f"Error: File not found at path: {file_path}"

        # Read the file contents
        with open(file_path, 'r', encoding='utf-8') as f:
            contents = f.read()

        if not contents.strip():
            return f"Warning: File at {file_path} is empty."

        return f"Contents of {file_path}:\n\n{contents}"

    except PermissionError:
        return f"Error: Permission denied when trying to read {file_path}"

    except Exception as e:
        return f"An error occurred while reading {file_path}: {str(e)}"
