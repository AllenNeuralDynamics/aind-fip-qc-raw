import logging
import csv
import json
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from aind_log_utils.log import setup_logging
from aind_data_schema.core.quality_control import (
    QCEvaluation,
    QCMetric,
    QCStatus,
    Stage,
    Status,
    QualityControl,
)
from aind_data_schema_models.modalities import Modality


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
        logging.info(f"Error: {file_path} not found.")
        sys.exit(1)


def load_csv_data(file_path):
    """Load CSV data into a NumPy array."""
    try:
        with open(file_path) as f:
            reader = csv.reader(f)
            return np.array([row for row in reader], dtype=np.float32)
    except FileNotFoundError:
        logging.info(f"Error: {file_path} not found.")
        sys.exit(1)


def generate_metrics(
    data1,
    data2,
    data3,
    rising_time,
    falling_time,
    green_floor_ave,
    iso_floor_ave,
    red_floor_ave,
):
    """Generate QC metrics based on data."""
    """Limits are set to 265 for all CMOSFloorDark metrics."""
    CMOSFloorDark_Green_Limit = 265
    CMOSFloorDark_Iso_Limit = 265
    CMOSFloorDark_Red_Limit = 265
    sudden_change_limit = 5000
    metrics = {
        "IsDataSizeSame": len(data1) == len(data2) == len(data3),
        "IsDataLongerThan15min": len(data1) > 18000,
        "IsSyncPulseSame": len(rising_time) == len(falling_time),
        "IsSyncPulseSameAsData": len(rising_time)
        in [len(data1), len(data2), len(data3)],
        "NoGreenNan": not np.isnan(data1).any(),
        "NoIsoNan": not np.isnan(data2).any(),
        "NoRedNan": not np.isnan(data3).any(),
        "CMOSFloorDark_Green": green_floor_ave < CMOSFloorDark_Green_Limit,
        "CMOSFloorDark_Iso": iso_floor_ave < CMOSFloorDark_Iso_Limit,
        "CMOSFloorDark_Red": red_floor_ave < CMOSFloorDark_Red_Limit,
        "NoSuddenChangeInSignal": all(
            np.max(np.diff(data[10:-2, 1])) < sudden_change_limit
            for data in [data1, data2, data3]
        ),
    }
    return metrics


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


def plot_cmos_trace_data(data_list, colors, results_folder):
    """Plot raw frame and cmos data and save to a file."""
    data1 = data_list[0]
    data2 = data_list[1]
    plt.figure(figsize=(8, 4))
    for i_panel in range(4):
        plt.subplot(8, 1, i_panel + 1)
        plt.plot(data1[:, i_panel + 1], color=colors[0])
        plt.title("GreenCh ROI:" + str(i_panel))
        plt.xlabel("frames")
        plt.ylabel("CMOS pixel val")

    for i_panel in range(4):
        plt.subplot(8, 1, i_panel + 5)
        plt.plot(data2[:, i_panel + 1], color=colors[1])
        plt.title("RedCh ROI:" + str(i_panel))
        plt.xlabel("frames")
        plt.ylabel("CMOS pixel val")
    plt.savefig(f"{results_folder}/raw_traces.png")
    plt.savefig(f"{results_folder}/raw_traces.pdf")
    plt.show()


def plot_sensor_floor(data1, data2, data3, results_folder):
    """
    Plot histograms for sensor floor values of three data sets.

    Parameters:
        data1 (numpy.ndarray): Data for GreenCh.
        data2 (numpy.ndarray): Data for IsoCh.
        data3 (numpy.ndarray): Data for RedCh.
        results_folder (str): Path to save the output plots.
    """
    plt.figure(figsize=(8, 2))

    # GreenCh Floor
    plt.subplot(1, 3, 1)
    plt.hist(data1[:, -1], bins=100, range=(255, 270), color="green", alpha=0.7)
    plt.xlim(255, 270)
    GreenChFloorAve = np.mean(data1[:, -1])
    plt.title(f"GreenCh FloorAve: {GreenChFloorAve:.2f}")
    plt.xlabel("CMOS pixel val")
    plt.ylabel("counts")

    # IsoCh Floor
    plt.subplot(1, 3, 2)
    plt.hist(data2[:, -1], bins=100, range=(255, 270), color="purple", alpha=0.7)
    plt.xlim(255, 270)
    IsoChFloorAve = np.mean(data2[:, -1])
    plt.title(f"IsoCh FloorAve: {IsoChFloorAve:.2f}")
    plt.xlabel("CMOS pixel val")
    plt.ylabel("counts")

    # RedCh Floor
    plt.subplot(1, 3, 3)
    plt.hist(data3[:, -1], bins=100, range=(255, 270), color="red", alpha=0.7)
    plt.xlim(255, 270)
    RedChFloorAve = np.mean(data3[:, -1])
    plt.title(f"RedCh FloorAve: {RedChFloorAve:.2f}")
    plt.xlabel("CMOS pixel val")
    plt.ylabel("counts")

    plt.subplots_adjust(wspace=0.8)

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
    plt.figure()
    plt.hist(diffs, bins=100, range=(0, 0.2))
    plt.title("sync pulse diff")
    plt.ylabel("counts")
    plt.xlabel("ms")

    # Save and show the plot
    plt.savefig(f"{results_folder}/SyncPulseDiff.png")
    plt.savefig(f"{results_folder}/SyncPulseDiff.pdf")
    plt.show()


def main():
    # Paths and setup
    fiber_base_path = Path("/data/fiber_raw_data")
    fiber_raw_path = fiber_base_path / "fib"
    results_folder = Path("../results")
    results_folder.mkdir(parents=True, exist_ok=True)

    # Load JSON files
    subject_data = load_json_file(fiber_base_path / "subject.json")
    subject_id = subject_data.get("subject_id")
    if not subject_id:
        logging.info("Error: Subject ID is missing from subject.json.")
        sys.exit(1)

    date_data = load_json_file(fiber_base_path / "data_description.json")
    date = date_data.get("creation_time")
    if not date:
        session_data = load_json_file(fiber_base_path / "session.json")
        date = session_data.get("session_start_time")

    asset_name = f"behavior_{subject_id}_{date}"
    setup_logging("aind-fip-qc-raw", mouse_id=subject_id, session_name=asset_name)

    try:
        channel_file_paths = [
            next(fiber_raw_path.glob(fiber_channel))
            for fiber_channel in ["FIP_DataG*", "FIP_DataIso_*", "FIP_DataR_*"]
        ]
    except StopIteration:
        logging.info("Error: FIP data files are missing.")
        sys.exit(1)

    data1, data2, data3 = [load_csv_data(file) for file in channel_file_paths]

    # Load behavior JSON
    behavior_json = load_json_file(next(fiber_base_path.glob("behavior/*.json")))
    rising_time = behavior_json["B_PhotometryRisingTimeHarp"]
    falling_time = behavior_json["B_PhotometryFallingTimeHarp"]

    # Calculate floor averages
    green_floor_ave = np.mean(data1[:, -1])
    iso_floor_ave = np.mean(data2[:, -1])
    red_floor_ave = np.mean(data3[:, -1])

    # Generate metrics
    metrics = generate_metrics(
        data1,
        data2,
        data3,
        rising_time,
        falling_time,
        green_floor_ave,
        iso_floor_ave,
        red_floor_ave,
    )

    # Plot data
    plot_cmos_trace_data(
        data_list=[data1, data2],
        colors=["darkgreen", "magenta"],
        results_folder=results_folder,
    )
    plot_sensor_floor(data1, data2, data3, results_folder)
    plot_sync_pulse_diff(rising_time, results_folder)

    # Create evaluations
    # TODO: Should datetime be using utc?
    evaluations = [
        create_evaluation(
            "Data length check",
            "Pass when GreenCh_data_length==IsoCh_data_length and the session is >15min",
            [
                QCMetric(
                    name="Data length same",
                    value=len(data1),
                    status_history=[
                        Bool2Status(
                            metrics["IsDataSizeSame"], t=datetime.now(timezone.utc)
                        )
                    ],
                    referenc=str(results_folder / "raw_traces.png"),
                )
            ],
        ),
        create_evaluation(
            "Data length check",
            "Pass when GreenCh_data_length==IsoCh_data_length and the session is >15min",
            [
                QCMetric(
                    name="Data length same",
                    value=len(data1),
                    status_history=[
                        Bool2Status(
                            metrics["IsDataSizeSame"], t=datetime.now(timezone.utc)
                        )
                    ],
                    reference=str(results_folder / "raw_traces.png"),
                ),
                QCMetric(
                    name="Session length >15min",
                    value=len(data1) / 20 / 60,
                    status_history=[
                        Bool2Status(
                            metrics["IsDataLongerThan15min"],
                            t=datetime.now(timezone.utc),
                        )
                    ],
                    reference=str(results_folder / "raw_traces.png"),
                ),
            ],
        ),
        create_evaluation(
            "Complete Synchronization Pulse",
            "Pass when Sync Pulse number equals data length, and when rising and falling give same lengths",
            [
                QCMetric(
                    name="Data length same (Rising)",
                    value=len(rising_time),
                    status_history=[
                        Bool2Status(
                            metrics["IsSyncPulseSame"], t=datetime.now(timezone.utc)
                        )
                    ],
                    reference=str(results_folder / "SyncPulseDiff.png"),
                ),
                QCMetric(
                    name="Data length same (Falling)",
                    value=len(falling_time),
                    status_history=[
                        Bool2Status(
                            metrics["IsSyncPulseSameAsData"],
                            t=datetime.now(timezone.utc),
                        )
                    ],
                    reference=str(results_folder / "SyncPulseDiff.png"),
                ),
            ],
            allow_failed=True,
        ),
        create_evaluation(
            "No NaN values in data",
            "Pass when no NaN values in the data",
            [
                QCMetric(
                    name="No NaN in Green channel",
                    value=float(np.sum(np.isnan(data1))),
                    status_history=[
                        Bool2Status(metrics["NoGreenNan"], t=datetime.now(timezone.utc))
                    ],
                ),
                QCMetric(
                    name="No NaN in Iso channel",
                    value=float(np.sum(np.isnan(data2))),
                    status_history=[
                        Bool2Status(metrics["NoIsoNan"], t=datetime.now(timezone.utc))
                    ],
                ),
                QCMetric(
                    name="No NaN in Red channel",
                    value=float(np.sum(np.isnan(data3))),
                    status_history=[
                        Bool2Status(metrics["NoRedNan"], t=datetime.now(timezone.utc))
                    ],
                ),
            ],
            allow_failed=False,
        ),
        create_evaluation(
            "CMOS Floor signal",
            "Pass when CMOS dark floor is <265 in all channel",
            [
                QCMetric(
                    name="Floor average signal in Green channel",
                    value=float(green_floor_ave),
                    status_history=[
                        Bool2Status(
                            metrics["CMOSFloorDark_Green"], t=datetime.now(timezone.utc)
                        )
                    ],
                    reference=str(results_folder / "CMOS_Floor.png"),
                ),
                QCMetric(
                    name="Floor average signal in Iso channel",
                    value=float(iso_floor_ave),
                    status_history=[
                        Bool2Status(
                            metrics["CMOSFloorDark_Iso"], t=datetime.now(timezone.utc)
                        )
                    ],
                    reference=str(results_folder / "CMOS_Floor.png"),
                ),
                QCMetric(
                    name="Floor average signal in Red channel",
                    value=float(red_floor_ave),
                    status_history=[
                        Bool2Status(
                            metrics["CMOSFloorDark_Red"], t=datetime.now(timezone.utc)
                        )
                    ],
                    reference=str(results_folder / "CMOS_Floor.png"),
                ),
            ],
        ),
    ]

    # Create QC object and save
    qc = QualityControl(evaluations=evaluations)
    qc.write_standard_file(output_directory=str(results_folder))


if __name__ == "__main__":
    main()
