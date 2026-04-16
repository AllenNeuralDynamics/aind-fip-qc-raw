import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from aind_data_schema.core.quality_control import (
    QCEvaluation,
    QCMetric,
    QCStatus,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality

from data_io import load_csv_data


def Bool2Status(boolean_value: bool, t: Optional[datetime] = None) -> QCStatus:
    """Convert a boolean pass/fail result to a QCStatus object.

    Parameters
    ----------
    boolean_value : bool
        True maps to Status.PASS, False maps to Status.FAIL.
    t : datetime | None
        Timestamp for the status entry. Should be timezone-aware.

    Returns
    -------
    QCStatus
        Automated QC status with the given pass/fail result and timestamp.
    """
    if boolean_value:
        return QCStatus(
            evaluator="Automated", status=Status.PASS, timestamp=t.isoformat()
        )
    else:
        return QCStatus(
            evaluator="Automated", status=Status.FAIL, timestamp=t.isoformat()
        )


def create_evaluation(
    name: str,
    description: str,
    metrics: List[QCMetric],
    modality: Modality = Modality.FIB,
    stage: Stage = Stage.RAW,
    allow_failed: bool = False,
) -> QCEvaluation:
    """Create a QCEvaluation object with project-standard defaults.

    Parameters
    ----------
    name : str
        Short display name for the evaluation.
    description : str
        Human-readable description of what the evaluation checks.
    metrics : list[QCMetric]
        One or more QCMetric objects that make up this evaluation.
    modality : Modality, optional
        Data modality. Defaults to Modality.FIB.
    stage : Stage, optional
        Pipeline stage. Defaults to Stage.RAW.
    allow_failed : bool, optional
        If True, individual metric failures do not fail the overall evaluation.
        Defaults to False.

    Returns
    -------
    QCEvaluation
        Configured evaluation object ready to be added to a QualityControl instance.
    """
    return QCEvaluation(
        name=name,
        modality=modality,
        stage=stage,
        metrics=metrics,
        allow_failed_metrics=allow_failed,
        description=description,
    )


def check_empty_channel_csvs(
    channel_file_paths: List[List[Path]], local_tz: timezone
) -> QCEvaluation:
    """Return a QCEvaluation flagging channel CSVs that exist but contain no data.

    An empty CSV (a file that exists but has zero data rows, possibly with only
    a header) indicates a likely hardware failure — e.g. a flaky RedCMOS trigger
    cable that causes the acquisition system to create the file but write nothing
    to it. This is distinct from a channel CSV that does not exist at all, which
    simply means that channel was not recorded (not enabled for this session, or
    not present on this rig) and is not an error condition.

    Parameters
    ----------
    channel_file_paths : list[list[Path]]
        Outer list has one entry per channel; each inner list contains the Path(s)
        to that channel's CSV file(s) as returned by glob.
    local_tz : timezone
        Timezone used to timestamp the QC status entry.

    Returns
    -------
    QCEvaluation
        PASS if all CSVs contain data; FAIL if any CSV exists but is empty,
        with the metric value set to the count of empty files.
    """
    empty_channel_files = [
        str(f)
        for channel_files in channel_file_paths
        for f in channel_files
        if len(load_csv_data(f)) == 0
    ]
    if empty_channel_files:
        logging.warning(
            "Empty channel CSV file(s) detected — likely a hardware failure "
            f"(e.g. faulty RedCMOS trigger cable): {empty_channel_files}"
        )
    evaluation = create_evaluation(
        "No empty channel CSV files",
        "Fail if any FIP channel CSV file exists but contains no data, "
        "indicating a hardware failure (e.g. faulty RedCMOS trigger cable).",
        [
            QCMetric(
                name="Channel CSV files are not empty",
                value=len(empty_channel_files),
                status_history=[
                    Bool2Status(
                        len(empty_channel_files) == 0,
                        t=datetime.now(local_tz),
                    )
                ],
            )
        ],
    )
    return evaluation
