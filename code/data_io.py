import csv
import json
import logging
from pathlib import Path
from typing import Union

import numpy as np


def load_json_file(file_path: Union[Path, str]) -> dict:
    """Load JSON data from a file.

    Parameters
    ----------
    file_path : Path | str
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON contents.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: {file_path} not found.")


def load_csv_data(file_path: Union[Path, str]) -> np.ndarray:
    """Load FIP channel CSV data into a NumPy array.

    Handles both legacy files (with a header row) and headerless files.
    If the file contains a broken last row (fewer columns than expected),
    that row is silently dropped.

    Parameters
    ----------
    file_path : Path | str
        Path to the CSV file.

    Returns
    -------
    np.ndarray
        2D float32 array of shape (n_frames, n_columns). Returns an empty
        array if the file exists but contains no data rows.
    """
    expected_header = ["SoftwareTS", "ROI0", "ROI1", "ROI2", "ROI3", "ROI4_sensorfloor", "HarpTS"]

    try:
        # Check for legacy header
        with open(file_path, newline='') as f:
            first_line = f.readline().strip().split(",")
        skip_first_row = first_line == expected_header

        rows = []
        with open(file_path, newline='') as f:
            reader = csv.reader(f)
            if skip_first_row:
                next(reader)  # skip header row
            for row in reader:
                # Drop 'HarpTS' if present as last column
                if skip_first_row:
                    row = row[:-1]
                for i, cell in enumerate(row):
                    if cell == "" and i > 0:
                        row[i] = row[i-1]
                        logging.error(f"Error: {file_path} csv file is found but broken -- contains empty string.")
                rows.append(row)

        return np.array(rows, dtype=np.float32)

    except FileNotFoundError:
        logging.error(f"Error: {file_path} not found.")

    except ValueError:
        with open(file_path) as f:
            reader = csv.reader(f)
            rows = [row for row in reader]

        max_cols = max(len(row) for row in rows)
        if len(rows[-1]) < max_cols:    #eliminating the broken last row
            rows.pop()
        return np.array(rows, dtype=np.float32)
