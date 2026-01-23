import os
from crewai.tools import tool


@tool("Write File Tool")
def write_file(file_path: str, content: str) -> str:
    """
    Writes content to a file at the specified path. Creates the file if it doesn't exist,
    and overwrites it if it does.

    Args:
        file_path (str): The absolute path to the file to write.
        content (str): The content to write to the file.

    Returns:
        str: Success message with the file path, or an error message if writing fails.
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Write the content to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return f"Successfully wrote content to {file_path}"

    except PermissionError:
        return f"Error: Permission denied when trying to write to {file_path}"

    except Exception as e:
        return f"An error occurred while writing to {file_path}: {str(e)}"
