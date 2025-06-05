from pathlib import Path
from typing import Dict

import pandas as pd

FIBER_CHANNELS = ("red", "green", "iso")


def get_fiber_channel_data(fiber_directory: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files in a fiber photmetry directory as pandas DataFrames.

    Parameters
    ----------
    fiber_directory : Path
        Path to the directory containing the CSV files.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary, where each key is a channel, and the value is the corresponding csv read as a dataframe
    """
    fiber_data = {}
    for channel in FIBER_CHANNELS:
        channel_csv_path = tuple(fiber_directory.glob(f"{channel}.csv"))
        if not channel_csv_path:
            raise FileNotFoundError(
                f"No {channel} csv found in fiber directory path {fiber_directory}"
            )

        fiber_data[channel] = pd.read_csv(channel_csv_path[0])

    return fiber_data
