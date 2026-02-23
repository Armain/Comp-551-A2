import configparser
from pathlib import Path
from pydantic import BaseModel

CONFIG_PATH = Path(__file__).parent.parent / "config.ini"


class Settings(BaseModel):
    random_seed: int
    show_inline_plots: bool


def load_config(path=CONFIG_PATH):
    """Load and validate config.ini using pydantic."""
    parser = configparser.ConfigParser()
    parser.read(path)
    return Settings(
        random_seed=parser.getint("settings", "random_seed"),
        show_inline_plots=parser.getboolean("settings", "show_inline_plots"),
    )


config = load_config()
