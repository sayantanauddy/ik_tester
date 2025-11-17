import subprocess
from pathlib import Path

def get_repo_root() -> Path:
    """Returns the absolute path of the git repository root."""
    git_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True
    ).stdout.strip()
    return Path(git_root)
