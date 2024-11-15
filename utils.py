import git
import os
import shutil

def load_markdown_file(file_path):
    """
    Load the contents of a markdown file.
    Args:
        file_path (str): Path to the markdown file to load.
    Returns:
        str: The content of the markdown file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        return markdown_content
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def clone_repository(repo_url, clone_dir):
    try:
        # Check if the directory already exists
        if os.path.exists(clone_dir):
            # If it exists, remove it first to avoid cloning errors
            shutil.rmtree(clone_dir)

        # Clone the repository
        git.Repo.clone_from(repo_url, clone_dir)
        print(f"Repository cloned to {clone_dir}")

    except git.exc.GitCommandError as e:
        print(f"Git command error while cloning repository: {e}")
    except Exception as e:
        print(f"Unexpected error while cloning repository: {e}")

def load_repo_files(repo_path):
  """Loads all files from a GitHub repository into a dictionary.

  Args:
    repo_path: The local path of the cloned GitHub repository.

  Returns:
    A dictionary where keys are file paths relative to the repo root and
    values are file contents as strings.
  """
  file_contents = {}
  for root, _, files in os.walk(repo_path):
    for file in files:
      file_path = os.path.join(root, file)
      relative_path = os.path.relpath(file_path, repo_path)
      try:
        with open(file_path, 'r', encoding='utf-8') as f:
          file_contents[relative_path] = f.read()
          print(f"Loaded file: {relative_path}")
      except UnicodeDecodeError:
        print(f"Skipping file due to encoding error: {file_path}")
  return file_contents