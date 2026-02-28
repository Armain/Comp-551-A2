import configparser
from pathlib import Path
from typing import Literal
import warnings

from pydantic import BaseModel

CONFIG_PATH = Path(__file__).parent.parent / "config.ini"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Settings(BaseModel):
    random_seed: int
    show_inline_plots: bool
    verbose: bool
    tuning_method: Literal["random", "optuna"]


def load_config(path: Path = CONFIG_PATH) -> Settings:
    """Load and validate config.ini via pydantic.

    Args:
        path: Path to the config.ini file.

    Returns:
        Validated Settings instance.
    """
    parser = configparser.ConfigParser()
    parser.read(path)
    return Settings(
        random_seed=parser.getint("settings", "random_seed"),
        show_inline_plots=parser.getboolean("settings", "show_inline_plots"),
        verbose=parser.getboolean("settings", "verbose"),
        tuning_method=parser.get("settings", "tuning_method"),
    )


config = load_config()
