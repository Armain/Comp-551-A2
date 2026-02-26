import configparser
from pathlib import Path
from pydantic import BaseModel

CONFIG_PATH = Path(__file__).parent.parent / "config.ini"


class Settings(BaseModel):
    random_seed: int
    show_inline_plots: bool
    verbose: bool


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
    )


config = load_config()
