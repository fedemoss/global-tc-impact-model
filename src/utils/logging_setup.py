import logging
import sys

from src.config import BASE_DIR

LOG_FILE = BASE_DIR / "main.log"
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for the whole pipeline.

    Writes to both stdout and a single ``main.log`` at the repository root.
    Safe to call multiple times — re-invocation replaces existing handlers
    on the root logger so behaviour stays predictable across entry points.
    """
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_LOG_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
