"""
Reading the settings a run needs but does not configure, which on this
project means the wandb key.
"""
import os
from pathlib import Path
from typing import Optional


# The root of the repository, which is where the .env file lives: three
# directories up from this one, since the package is installed from it.
_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(env_file: Optional[Path] = None) -> None:
    """
    Read the .env file at the root of the repository into the
    environment, leaving any name already set alone and doing nothing at
    all when there is no such file.
    """
    env_file = Path(env_file) if env_file is not None else _ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))


__all__ = [
    "load_dotenv",
]
