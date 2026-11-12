"""Ground station entry point: live stereo depth from the Pico.

  uv run main.py [--algo sgbm|census-sgm|elas] [--headless] ...

Same as `python -m modules.depth`; see --help for every option.
"""
from modules.depth.cli import main

if __name__ == "__main__":
    main()
