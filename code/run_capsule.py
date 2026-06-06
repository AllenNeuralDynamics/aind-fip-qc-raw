import csv
import glob
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytz
from aind_data_schema.core.quality_control import (
    QCEvaluation,
    QCMetric,
    QCStatus,
    QualityControl,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, welch
from scipy.stats import median_abs_deviation

from aind_logging import setup_logging


def Bool2Status(boolean_value, t=None):
    """Convert a boolean value to a QCStatus object."""
    if boolean_value:
        return QCStatus(
            evaluator="Automated", status=Status.PASS, timestamp=t.isoformat()
        )
    else:
        return QCStatus(
            evaluator="Automated", status=Status.FAIL, timestamp=t.isoformat()
        )


def load_json_file(file_path):
    """Load JSON data from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: {file_path} not found.")


def load_csv_data(file_path):
    """Load CSV data into a NumPy array."""

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

        # Convert to NumPy array
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


def create_evaluation(
    name,
    description,
    metrics,
    modality=Modality.FIB,
    stage=Stage.RAW,
    allow_failed=False,
):
    """Create a QC evaluation object."""
    return QCEvaluation(
        name=name,
        modality=modality,
        stage=stage,
        metrics=metrics,
        allow_failed_metrics=allow_failed,
        description=description,
    )


def check_empty_channel_csvs(channel_names, channel_file_paths, local_tz):
    """
    Return a QCEvaluation with one metric per channel that has CSV file(s).

    Each metric checks whether the channel's CSV contains data. A PASS means
    the channel has valid data rows. A FAIL means the file exists but is empty
    (zero data rows, possibly header-only), which indicates a likely hardware
    failure — e.g. a flaky RedCMOS trigger cable.

    Channels with no CSV file at all are not included — a missing file simply
    means that channel was not recorded (not enabled for this session, or not
    present on this rig) and is not an error condition.
    """
    channel_metrics = []
    for name, files in zip(channel_names, channel_file_paths):
        if not files:
            continue
        has_data = all(len(load_csv_data(f)) > 0 for f in files)
        if not has_data:
            logging.warning(
                f"Empty {name} channel CSV detected — likely a hardware failure "
                f"(e.g. faulty RedCMOS trigger cable): {[str(f) for f in files]}"
            )
        channel_metrics.append(
            QCMetric(
                name=f"{name} channel CSV contains data",
                value=int(has_data),
                status_history=[
                    Bool2Status(has_data, t=datetime.now(local_tz))
                ],
            )
        )
    evaluation = create_evaluation(
        "Channel data present",
        "Per-channel check that each CSV file contains data. "
        "A FAIL indicates a hardware failure (e.g. faulty RedCMOS trigger cable). "
        "Channels with no CSV file are excluded (not an error).",
        channel_metrics,
    )
    return evaluation


def generate_metrics(data_lists, loaded_channels, rising_time, falling_time):
    """Generate QC metrics based on data."""
    """Limits are set to 265 for all CMOSFloorDark metrics."""
    CMOSFloorDark_Limit = 265
    sudden_change_limit = 2000
    channel_lengths = [len(data) for _, data in loaded_channels]
    floor_aves = {name: float(np.mean(data[:, -1])) for name, data in loaded_channels}
    metrics = {
        "IsDataSizeSame": len(set(channel_lengths)) == 1,
        "IsDataLongerThan15min": channel_lengths[0] > 18000,
        "IsSyncPulseSame": len(rising_time) == len(falling_time),
        "IsSyncPulseSameAsData": len(rising_time) in channel_lengths,
        "NoNan": {name: not np.isnan(data).any() for name, data in loaded_channels},
        "CMOSFloorDark": {name: floor_aves[name] < CMOSFloorDark_Limit for name, _ in loaded_channels},
        "FloorAves": floor_aves,
        "NoSuddenChangeInSignal": all(
            np.max(np.diff(data[10:-2, 1])) < sudden_change_limit
            for _, data in loaded_channels
        ),
        "IsSingleRecordingPerSession": len(data_lists[0]) == 1,
    }
    return metrics


def plot_cmos_trace_data(loaded_channels, results_folder, rig_id, experimenter):
    """Plot raw frame and cmos data and save to a file."""
    color_map = {"Green": "darkgreen", "Iso": "purple", "Red": "magenta"}
    n_channels = len(loaded_channels)
    fig, axes = plt.subplots(n_channels * 4, 1, figsize=(16, 20), sharex=True)
    for ch_idx, (name, data) in enumerate(loaded_channels):
        for i_panel in range(4):
            ax = axes[ch_idx * 4 + i_panel]
            ax.plot(data[:, i_panel + 1], color=color_map[name])
            if ch_idx == 0 and i_panel == 0:
                ax.set_title(f"{name}Ch ROI:{i_panel} rig: {rig_id} by: {experimenter}")
                ax.set_ylabel("CMOS pixel val")
            else:
                ax.set_title(f"{name}Ch ROI:{i_panel}")
    axes[-1].set_xlabel("frames (20Hz)")

    fig.tight_layout()

    fig.savefig(f"{results_folder}/raw_traces.png")
    fig.savefig(f"{results_folder}/raw_traces.pdf")
    plt.show()


def plot_sensor_floor(loaded_channels, results_folder):
    """
    Plot histograms for sensor floor values for each loaded channel.

    Parameters:
        loaded_channels: list of (name, data) tuples for each channel with data.
        results_folder (str): Path to save the output plots.
    """
    color_map = {"Green": "green", "Iso": "purple", "Red": "red"}
    n_channels = len(loaded_channels)
    plt.figure(figsize=(8, 4))
    for ch_idx, (name, data) in enumerate(loaded_channels):
        floor_ave = np.mean(data[:, -1])
        # Top row: zoomed range
        plt.subplot(2, n_channels, ch_idx + 1)
        plt.hist(data[:, -1], bins=100, range=(255, 270), color=color_map[name], alpha=0.7)
        plt.xlim(255, 270)
        plt.title(f"{name}Ch FloorAve: {floor_ave:.2f}")
        plt.xlabel("CMOS pixel val")
        plt.ylabel("counts")
        # Bottom row: all data
        plt.subplot(2, n_channels, n_channels + ch_idx + 1)
        plt.hist(data[:, -1], bins=100, color=color_map[name], alpha=0.7)
        plt.title(f"{name}Floor - All data")
        plt.xlabel("CMOS pixel val")
        plt.ylabel("counts")

    plt.subplots_adjust(wspace=0.8)
    plt.subplots_adjust(hspace=0.8)

    # Save and show the plot
    plt.savefig(f"{results_folder}/CMOS_Floor.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{results_folder}/CMOS_Floor.pdf")
    plt.show()


def plot_sync_pulse_diff(rising_time, results_folder):
    """
    Plot a histogram of the differences in rising times and save the plot.

    Parameters:
        rising_time (array-like): An array of rising time values.
        save_path (str): The path to save the generated plot.
    """
    # Compute differences
    diffs = np.diff(rising_time)

    # Create the plot
    plt.figure(figsize=(4, 2))
    plt.subplot(1, 2, 1)
    plt.hist(diffs, bins=100, range=(0, 0.2))
    plt.title("sync pulse diff")
    plt.ylabel("counts")
    plt.xlabel("ms")

    plt.subplot(1, 2, 2)
    plt.hist(diffs, bins=100)
    plt.title("sync pulse diff - all")
    plt.ylabel("counts")
    plt.xlabel("ms")
    plt.subplots_adjust(wspace=0.8)

    # Save and show the plot
    plt.savefig(f"{results_folder}/SyncPulseDiff.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{results_folder}/SyncPulseDiff.pdf")
    plt.show()


def detect_psd_peaks(
    signal, fs=20, nperseg=1024, threshold_mad=6.0, size=51, ax=None, ch="Green"
):
    freqs, psd = np.array(welch(signal, fs=fs, nperseg=nperseg))[:, 1:-1]
    log_psd = np.log(psd)
    trend = median_filter(log_psd, size=size, mode="nearest")
    residuals = log_psd - trend
    mad = median_abs_deviation(residuals)
    thresh = np.median(residuals) + threshold_mad * mad
    peaks_idx, _ = find_peaks(residuals, distance=2)
    mask = (residuals > thresh) & np.isin(np.arange(len(residuals)), peaks_idx)
    if ax is not None:
        c = {"Green": "#009E73", "Iso": "#0072B2", "Red": "#D55E00"}.get(ch, "C0")
        ax[0].semilogy(freqs, psd, c=c, label=f"PSD of {ch} channel")
        ax[0].semilogy(freqs, np.exp(trend), c="C1", label="trend")
        ax[0].semilogy(freqs[mask], psd[mask], "x", c="C3", ms=8)
        ax[1].plot(freqs, residuals, c=c, label="log(PSD - trend)")
        ax[1].axhline(thresh, c="C3", label="threshold")
        ax[1].plot(freqs[mask], residuals[mask], "x", c="C3", ms=8)
        for f, r in zip(freqs[mask], residuals[mask]):
            ax[1].annotate(
                f"{f:.2f} Hz",
                xy=(f, r),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    return freqs[mask], residuals[mask], thresh


def plot_psd(loaded_channels, results_folder):
    peaks = []
    fig = plt.figure(figsize=(15, 20))
    gs_outer = fig.add_gridspec(3, 1, hspace=0.1)  # group by channel
    ax = np.empty((12, 2), dtype=object)
    for i in range(3):
        gs_inner = gs_outer[i].subgridspec(4, 2, hspace=0.1, wspace=0.1)
        for r in range(4):
            for c in range(2):
                sharex = ax[0, c] if (i > 0 or r > 0) else None
                ax[4 * i + r, c] = fig.add_subplot(gs_inner[r, c], sharex=sharex)

    for i, (ch, data) in enumerate(loaded_channels):
        for roi in range(4):
            row = 4 * i + roi
            try:
                peaks.append(detect_psd_peaks(data[:, roi + 1], ax=ax[row], ch=ch))
            except Exception:
                logging.exception(f"Failed to plot PSD for {ch} ROI {roi}")
            ax[row, 0].set_ylabel(f"ROI {roi}")
        ax[4 * i, 0].set_title(f"PSD of {ch} channel", x=1)
        for a in ax[4 * i]:
            a.legend()
    for row in range(11):
        for a in ax[row]:
            a.tick_params(labelbottom=False)
    for a in ax[-1]:
        a.set_xlabel("Frequency [Hz]")
    fig.savefig(f"{results_folder}/power_spectrum.png", bbox_inches="tight", pad_inches=0.05)
    max_peaks = np.array(
        [[p[0][np.argmax(p[1])], max(p[1]) / p[2]] for p in peaks if len(p[0])]
    )
    return max_peaks[np.argmax(max_peaks[:, 1]), 0] if len(max_peaks) else None


def main():
    # Paths and setup
    fiber_base_path = Path(os.getenv("FIBER_DATA_PATH", "/data/fiber_raw_data"))
    process_name = os.getenv("PROCESS_NAME")
    data_disc_json = load_json_file(fiber_base_path / "data_description.json")
    asset_name = data_disc_json.get("name")
    setup_logging(
        process_name,
        acquisition_name=asset_name,
        process_name=process_name,
        pipeline_name=os.getenv("PIPELINE_NAME","")
    )
    logging.info("Begin processing...", extra={"event_type": "stage_start"})
    fiber_raw_path = fiber_base_path / "fib"
    results_folder = Path("../results/")
    results_folder.mkdir(parents=True, exist_ok=True)
    qc_folder = Path("../results/qc-raw")
    qc_folder.mkdir(parents=True, exist_ok=True)

    ref_folder = Path("qc-raw")
    fiber_exists = True

    # Load JSON files
    subject_data = load_json_file(fiber_base_path / "subject.json")
    subject_id = subject_data.get("subject_id")
    if not subject_id:
        logging.error("Error: Subject ID is missing from subject.json.")


    session_data = load_json_file(fiber_base_path / "session.json")
    rig_id = session_data.get("rig_id")
    experimenter = session_data.get("experimenter_full_name")[0]

    # Assuming `fiber_raw_path` is defined earlier in the code
    fiber_channel_patterns = ["FIP_DataG*", "FIP_DataIso_*", "FIP_DataR_*"]

    # Use a list comprehension to find matching files
    channel_file_paths = [
        sorted(fiber_raw_path.glob(fiber_channel))  #sorted based on DAQ time
        for fiber_channel in fiber_channel_patterns
    ]

    # Process if at least one channel CSV exists. Two-channel recordings produce
    # only two CSV files (no Red); this is not an error, just a different rig
    # configuration. (Distinct from the case where three CSVs exist but one is
    # empty, which is handled separately by check_empty_channel_csvs.)
    fiber_exists = any(channel_file_paths)

    if fiber_exists:
        data_lists = [[load_csv_data(file) for file in file_list] for file_list in channel_file_paths]
        data1_list, data2_list, data3_list = data_lists #keep all csv files

        if len(data_lists[0]) > 1:
            logging.error("Multiple recording files found in this session. Only the largest file was used for QC.")

        data1 = max(data1_list, key=lambda x: x.shape[0]) if data1_list else np.array([]) #using only the longest file for each ch
        data2 = max(data2_list, key=lambda x: x.shape[0]) if data2_list else np.array([])
        data3 = max(data3_list, key=lambda x: x.shape[0]) if data3_list else np.array([])

        channel_names = ["Green", "Iso", "Red"]
        loaded_channels = [
            (name, data)
            for name, data in zip(channel_names, [data1, data2, data3])
            if len(data) > 0
        ]
        n_channels = len(loaded_channels)

        seattle_tz = pytz.timezone("America/Los_Angeles")
        evaluations = [
            check_empty_channel_csvs(channel_names=channel_names, channel_file_paths=channel_file_paths, local_tz=seattle_tz)
        ]

        if n_channels >= 2:

            # Load behavior JSON (dynamic foraging specific)
            # Regex pattern is <subject_id>_YYYY-MM-DD_HH-MM-SS.json
            pattern = "/data/fiber_raw_data/behavior/[0-9]*_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"
            matching_behavior_files = glob.glob(pattern)
            if matching_behavior_files:
                behavior_json = load_json_file(matching_behavior_files[0])
                rising_time = behavior_json["B_PhotometryRisingTimeHarp"]
                falling_time = behavior_json["B_PhotometryFallingTimeHarp"]
            else:
                logging.info("NO BEHAVIOR JSON — Non-dynamicforaging or simply missing")
                # preparing fake syncpulses
                rising_time = list(range(0, len(loaded_channels[0][1]), 50))
                falling_time = list(range(0, len(loaded_channels[0][1]), 50))

            # Generate metrics
            metrics = generate_metrics(
                data_lists,
                loaded_channels,
                rising_time,
                falling_time,
            )

            # Plot data
            plot_cmos_trace_data(
                loaded_channels=loaded_channels,
                results_folder=results_folder,
                rig_id=rig_id,
                experimenter=experimenter,
            )
            plot_sensor_floor(loaded_channels, results_folder)
            plot_sync_pulse_diff(rising_time, results_folder)
            freq_of_peak = plot_psd(loaded_channels, results_folder)

            # Create evaluations with our timezone
            evaluations += [
                create_evaluation(
                    "Data length check",
                    "Pass when data_length for Green/Iso/Red are same and the session is >15min",
                    [
                        QCMetric(
                            name="Data length same",
                            value=len(loaded_channels[0][1]),
                            status_history=[
                                Bool2Status(
                                    metrics["IsDataSizeSame"], t=datetime.now(seattle_tz)
                                )
                            ],
                            reference=str(ref_folder / "raw_traces.png"),
                        ),
                        QCMetric(
                            name="Session length >15min",
                            value=len(loaded_channels[0][1]) / 20 / 60,
                            status_history=[
                                Bool2Status(
                                    metrics["IsDataLongerThan15min"],
                                    t=datetime.now(seattle_tz),
                                )
                            ],
                            reference=str(ref_folder / "raw_traces.png"),
                        ),
                    ],
                ),
                create_evaluation(
                    "Complete Synchronization Pulse",
                    "Pass when 1)rising and falling give the same length; 2)sync Pulse number equals data length",
                    [
                        QCMetric(
                            name="Rising/Falling of Sync pulses same length (Value: Rising edge of synch pulse)",
                            value=len(rising_time),
                            status_history=[
                                Bool2Status(
                                    metrics["IsSyncPulseSame"], t=datetime.now(seattle_tz)
                                )
                            ],
                            reference=str(ref_folder / "SyncPulseDiff.png"),
                        ),
                        QCMetric(
                            name="Data length same as one of data length (Value: Falling edge of synch pulse)",
                            value=len(falling_time),
                            status_history=[
                                Bool2Status(
                                    metrics["IsSyncPulseSameAsData"],
                                    t=datetime.now(seattle_tz),
                                )
                            ],
                            reference=str(ref_folder / "SyncPulseDiff.png"),
                        ),
                    ],
                    allow_failed=True,
                ),
                create_evaluation(
                    "No NaN values in data",
                    "Pass when no NaN values in the data",
                    [
                        QCMetric(
                            name=f"No NaN in {name} channel",
                            value=float(np.sum(np.isnan(data))),
                            status_history=[
                                Bool2Status(metrics["NoNan"][name], t=datetime.now(seattle_tz))
                            ],
                        )
                        for name, data in loaded_channels
                    ],
                    allow_failed=False,
                ),
                create_evaluation(
                    "CMOS Floor signal",
                    "Pass when CMOS dark floor is <265 in all channel",
                    [
                        QCMetric(
                            name=f"Floor average signal in {name} channel",
                            value=metrics["FloorAves"][name],
                            status_history=[
                                Bool2Status(
                                    metrics["CMOSFloorDark"][name],
                                    t=datetime.now(seattle_tz),
                                )
                            ],
                            reference=str(ref_folder / "CMOS_Floor.png"),
                        )
                        for name, _ in loaded_channels
                    ],
                ),
                create_evaluation(
                    "No sudden changes in signals",
                    "Pass when no sudden change in signal",
                    [
                        QCMetric(
                            name="Max 1st derivative",
                            value=float(
                                np.max([
                                    np.max(np.diff(data[10:-2, 1]))
                                    for _, data in loaded_channels
                                ])
                            ),
                            status_history=[
                                Bool2Status(
                                    metrics["NoSuddenChangeInSignal"],
                                    t=datetime.now(seattle_tz),
                                )
                            ],
                            reference=str(ref_folder / "raw_traces.png"),
                        ),
                    ],
                ),
                create_evaluation(
                    "Single data file per channel in the session",
                    "Pass when the session folder has only one data per channel",
                    [
                        QCMetric(
                            name="Number of data files per channel",
                            description="When FIP-Bonsai workflow starts/stops multiple times, it would generate multiple CSVs, RawMovie files, etc",
                            value=len(data_lists[0]),
                            status_history=[
                                Bool2Status(
                                    metrics["IsSingleRecordingPerSession"],
                                    t=datetime.now(seattle_tz),
                                )
                            ],
                        ),
                    ],
                ),
                create_evaluation(
                    "No spurious PSD peaks",
                    "Pass when no ROI shows an anomalous narrowband peak. A FAIL indicates periodic interference (e.g. electrical noise).",
                    [
                        QCMetric(
                            name="Dominant spectral peak frequency [Hz]",
                            value=float(freq_of_peak) if freq_of_peak is not None else float("nan"),
                            status_history=[
                                Bool2Status(
                                    freq_of_peak is None,
                                    t=datetime.now(seattle_tz),
                                )
                            ],
                            reference=str(ref_folder / "power_spectrum.png"),
                        ),
                    ],
                ),
            ]

            # We'd like to have our files organized such that QC is in the
            # results directory while plots are in a named folder.
            # This allows the final results asset to have the same structure
            # We need to generate QC in the parent to ensure it works with the
            # Web portal
            excluded_file = "quality_control.json"
            # Iterate over files in the results directory
            for filename in os.listdir(results_folder):
                source_path = os.path.join(results_folder, filename)
                destination_path = os.path.join(qc_folder, filename)

                # Move everything except the excluded file
                if os.path.isfile(source_path) and filename != excluded_file:
                    shutil.move(source_path, destination_path)

        # Create QC object and save
        qc = QualityControl(evaluations=evaluations)
        qc.write_standard_file(output_directory=str(results_folder))

    else:
        logging.info("FIP data files are missing. This may be a behavior session.")
        logging.info("No Fiber Data to QC")
        qc_file_path = results_folder / "no_fip_to_qc.txt"
        # Create an empty file
        with open(qc_file_path, "w") as file:
            file.write("FIP data files are missing. This may be a behavior session.")

        logging.info(f"Empty file created at: {qc_file_path}")
    logging.info("Pipeline stage completed", extra={"event_type": "stage_complete"})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception(
            "Pipeline stage failed",
            extra={"event_type": "stage_error"}
        )
        raise
