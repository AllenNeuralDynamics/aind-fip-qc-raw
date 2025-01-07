""" top level run script """

#%% Import
import csv
import glob
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from datetime import datetime, timezone
import boto3
import requests
import kachery_cloud as kcl
from aws_requests_auth.aws_auth import AWSRequestsAuth
import argparse 
from pathlib import Path
#import logging
from aind_log_utils.log import setup_logging
from aind_data_access_api.document_db import MetadataDbClient
from aind_data_schema_models.modalities import Modality
from aind_data_schema.core.quality_control import QCEvaluation, QualityControl, QCMetric, Stage, Status, QCStatus


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="aind-fip-qc-raw")

    data_folder = Path("../data")
    results_folder = Path("../results")

    parser.add_argument("--asset-name", type=str, default = 'behavior_754430_2024-12-19_13-05-45')

    # Parse the command-line arguments
    args = parser.parse_args()
    asset_name = args.asset_name

    if not results_folder.is_dir():
        results_folder.mkdir(parents=True)

    if asset_name is not None and asset_name == "":
        asset_name = None
        
    if asset_name is not None:
        fiber_json_path = Path("/data/fiber_raw_data")
        # Load subject data
        subject_json_path = fiber_json_path / "subject.json"
        with open(subject_json_path, "r") as f:
            subject_data = json.load(f)

        # Grab the subject_id and times for logging
        subject_id = subject_data.get("subject_id", None)

        # Raise an error if subject_id is None
        if subject_id is None:
            logging.info("NO SUBJECT ID IN SUBJECT FILE")
            raise ValueError("subject_id is missing from the subject_data.")

        # Load data description
        data_description_path = fiber_json_path / "data_description.json"
        with open(data_description_path, "r") as f:
            date_data = json.load(f)

        # Attempt to extract the creation time
        date = date_data.get("creation_time", None)

        # Fallback to session start time if date is missing
        if date is None:
            session_path = fiber_json_path / "session.json"
            with open(session_path, "r") as f:
                session_data = json.load(f)
            date = session_data.get("session_start_time", None)
        asset_name = "behavior_" + subject_id + "_" + date

        setup_logging("aind-fip-qc-raw", mouse_id=mouse_id, session_name = asset_name)



        #need to skip the entire QC if FIP files don't exist
        try:
            file1  = glob.glob(fibfolder + os.sep + "FIP_DataG*")[0]
            file2 = glob.glob(fibfolder + os.sep + "FIP_DataIso_*")[0]
            file3 = glob.glob(fibfolder + os.sep + "FIP_DataR_*")[0]
        except:
            #logging.info("FIP Data don't exist, skipping the QC capsule")
            sys.exit(1)

        with open(file1) as f:
            reader = csv.reader(f)
            datatemp = np.array([row for row in reader])
            data1 = datatemp[:,:].astype(np.float32)
            #del datatemp
            
        with open(file2) as f:
            reader = csv.reader(f)
            datatemp = np.array([row for row in reader])
            data2 = datatemp[:,:].astype(np.float32)
            #del datatemp
            
        with open(file3) as f:
            reader = csv.reader(f)
            datatemp = np.array([row for row in reader])
            data3 = datatemp[:,:].astype(np.float32)
            #del datatemp

        #%% read behavior json file
        behavior_json_path = glob.glob(sessionfolder + '/behavior/*' + sessionname + '.json')[0]


        try:
            with open(behavior_json_path, 'r', encoding='utf-8') as f:
                behavior_json = json.load(f)
        except:
            #logging.info("behavior json file don't exist, skipping the QC capsule")
            sys.exit(1)        

        RisingTime=behavior_json['B_PhotometryRisingTimeHarp']
        FallingTime=behavior_json['B_PhotometryFallingTimeHarp']

        #%%raw data
        plt.figure(figsize=(8, 4))
        for i_panel in range(4):
            plt.subplot(2,4,i_panel+1)
            plt.plot(data1[:,i_panel+1],color='darkgreen')
            plt.title('GreenCh ROI:' + str(i_panel))
            plt.xlabel('frames')
            plt.ylabel('CMOS pixel val')

        for i_panel in range(4):
            plt.subplot(2,4,i_panel+5)
            plt.plot(data2[:,i_panel+1],color='magenta')
            plt.title('RedCh ROI:' + str(i_panel))
            plt.xlabel('frames')
            plt.ylabel('CMOS pixel val')

        plt.subplots_adjust(wspace=0.8, hspace=0.8)

        plt.show()
        plt.savefig(str(results_folder) + '/raw_traces.png')
        plt.savefig('/root/capsule/results/raw_traces.pdf')

        #%%
        #sensor floor (last ROI)
        plt.figure(figsize=(8, 2))

        plt.subplot(1,3,1)
        plt.hist(data1[:,-1],bins=100, range=(255, 270))
        plt.xlim(255,270)
        GreenChFloorAve=np.mean(data1[:,-1])
        plt.title('GreenCh FloorAve:' + str(GreenChFloorAve))
        plt.xlabel('CMOS pixel val')
        plt.ylabel('counts')

        plt.subplot(1,3,2)
        plt.hist(data2[:,-1],bins=100, range=(255, 270))
        plt.xlim(255,270)
        IsoChFloorAve=np.mean(data2[:,-1])
        plt.title('IsoCh FloorAve:' + str(IsoChFloorAve))
        plt.xlabel('CMOS pixel val')
        plt.ylabel('counts')

        plt.subplot(1,3,3)
        plt.hist(data3[:,-1],bins=100, range=(255, 270))
        plt.xlim(255,270)
        RedChFloorAve=np.mean(data3[:,-1])
        plt.title('RedCh FloorAve:' + str(RedChFloorAve))
        plt.xlabel('CMOS pixel val')
        plt.ylabel('counts')

        plt.subplots_adjust(wspace=0.8)

        plt.show()
        plt.savefig(str(results_folder) + '/CMOS_Floor.png',dpi=300, bbox_inches='tight')
        plt.savefig('/root/capsule/results/CMOS_Floor.pdf')

        #%% sync pulse diff
        plt.figure()
        plt.hist(np.diff(RisingTime), bins=100, range=(0, 0.2))
        plt.title("sync pulse diff")
        plt.ylabel("counts")
        plt.xlabel("ms")

        plt.show()
        plt.savefig('/root/capsule/results/SyncPulseDiff.png')

        #%%

        '''
        # %%
        file_path0 = "/root/capsule/results/raw_traces.png"
        uri0 = kcl.store_file(file_path0, label=file_path0)

        file_path1 = "/root/capsule/results/CMOS_Floor.png"
        uri1 = kcl.store_file(file_path1, label=file_path1)

        file_path2 = "/root/capsule/results/SyncPulseDiff.png"
        uri2 = kcl.store_file(file_path2, label=file_path2)
        '''
        #%%
        Metrics = dict()

        if len(data1) == len(data2) and len(data2) == len(data3):
            Metrics["IsDataSizeSame"] = True
        else:
            Metrics["IsDataSizeSame"] = False
            #logging.info("DataSizes are not the same")

        if len(data1) > 18000:
            Metrics["IsDataLongerThan15min"] = True
        else:
            Metrics["IsDataLongerThan15min"] = False
            #logging.info("The session is shorter than 15min")

        if len(RisingTime) == len(FallingTime):
            Metrics["IsSyncPulseSame"] = True
        else:
            Metrics["IsSyncPulseSame"] = False
            #logging.info("# of Rising and Falling sync pulses are not the same")

        if len(RisingTime) == len(data1) or len(RisingTime) == len(data2) or len(RisingTime) == len(data3):
            Metrics["IsSyncPulseSameAsData"] = True
        else:
            Metrics["IsSyncPulseSameAsData"] = False
            #logging.info("# of sync pulses are not the same as Data")

        if np.isnan(data1).any():
            Metrics["NoGreenNan"] = False
            #logging.info("Green Ch has NaN")
        else:
            Metrics["NoGreenNan"] = True

        if np.isnan(data2).any():
            Metrics["NoIsoNan"] = False
            #logging.info("Green Ch has NaN")
        else:
            Metrics["NoIsoNan"] = True

        if np.isnan(data1).any():
            Metrics["NoRedNan"] = False
            #logging.info("Red Ch has NaN")
        else:
            Metrics["NoRedNan"] = True


        if GreenChFloorAve < 265:
            Metrics["CMOSFloorDark_Green"] = True
        else:
            Metrics["CMOSFloorDark_Green"] = False
            #logging.info("CMOS Floor is not dark; potential light leak or error in ROI allocation")

        if IsoChFloorAve < 265:
            Metrics["CMOSFloorDark_Iso"] = True
        else:
            Metrics["CMOSFloorDark_Iso"] = False
            #logging.info("CMOS Floor is not dark; potential light leak or error in ROI allocation")

        if RedChFloorAve < 265:
            Metrics["CMOSFloorDark_Red"] = True
        else:
            Metrics["CMOSFloorDark_Red"] = False
            #logging.info("CMOS Floor is not dark; potential light leak or error in ROI allocation")


        if np.max(np.diff(data1[10:-2,1])) < 5000 and np.max(np.diff(data2[10:-2,1])) < 5000 and np.max(np.diff(data3[10:-2,1])) < 5000:
            Metrics["NoSuddenChangeInSignal"] = True
        else:
            Metrics["NoSuddenChangeInSignal"] = False
            #logging.info("Sudden change in signal; potential movement in fiber coupling and etc")


        #%% AIND QC schema embeding
        t = datetime.now(timezone.utc)

        # Build some status objects
        sp = QCStatus(evaluator="Automated", status=Status.PASS, timestamp=t.isoformat())
        sf = QCStatus(evaluator="Automated", status=Status.FAIL, timestamp=t.isoformat())
        spend = QCStatus(evaluator="Automated", status=Status.PENDING, timestamp=t.isoformat())

        def Bool2Status(boolean_value):
            if boolean_value:
                return QCStatus(evaluator="Automated", status=Status.PASS, timestamp=t.isoformat())
            else:
                return QCStatus(evaluator="Automated", status=Status.FAIL, timestamp=t.isoformat())

        eval0 = QCEvaluation(
            name="Data length check",
            modality=Modality.FIB,
            stage=Stage.RAW,
            metrics=[
                QCMetric(name="Data length same", value=len(data1), status_history=[Bool2Status(Metrics["IsDataSizeSame"])], reference=r'/root/capsule/results/raw_traces.png'),
                QCMetric(name="Session length >15min", value=len(data1)/20/60, status_history=[Bool2Status(Metrics["IsDataLongerThan15min"])], reference=r'/root/capsule/results/raw_traces.png')
            ],
            description="Pass when GreenCh_data_length==IsoCh_data_length and GreenCh_data_length==RedCh_data_length, and the session is >15min",
        )

        eval1 = QCEvaluation(
            name="Complete Synchronization Pulse",
            modality=Modality.FIB,
            stage=Stage.RAW,
            metrics=[
                QCMetric(name="Data length same", value=len(RisingTime), status_history=[Bool2Status(Metrics["IsSyncPulseSame"])],reference="/root/capsule/results/SyncPulseDiff.png"),
                QCMetric(name="Data length same", value=len(FallingTime), status_history=[Bool2Status(Metrics["IsSyncPulseSameAsData"])],reference="/root/capsule/results/SyncPulseDiff.png") 
            ],
            allow_failed_metrics=True, # most incoplete sync pulses can be fixed/recovered during preprocessing (alignment module)
            description="Pass when Sync Pulse number equals data length, and when rising and falling give same lengths",
        )

        eval2 = QCEvaluation(
            name="No NaN values in data",
            modality=Modality.FIB,
            stage=Stage.RAW,
            metrics=[
                QCMetric(name="No NaN in Green channel", value=float(np.sum(np.isnan(data1))), status_history=[Bool2Status(Metrics["NoGreenNan"])]),
                QCMetric(name="No NaN in Iso channel", value=float(np.sum(np.isnan(data2))), status_history=[Bool2Status(Metrics["NoIsoNan"])]),
                QCMetric(name="No NaN in Red channel", value=float(np.sum(np.isnan(data3))), status_history=[Bool2Status(Metrics["NoRedNan"])])
            ],
            allow_failed_metrics=False,
            description="Pass when no NaN values in the data"
        )

        eval3 = QCEvaluation(
            name="CMOS Floor signal",
            modality=Modality.FIB,
            stage=Stage.RAW,
            metrics=[
                QCMetric(name="Floor average signal in Green channel", value=float(GreenChFloorAve), status_history=[Bool2Status(Metrics["CMOSFloorDark_Green"])],reference="/root/capsule/results/CMOS_Floor.png"),
                QCMetric(name="Floor average signal in Iso channel", value=float(IsoChFloorAve), status_history=[Bool2Status(Metrics["CMOSFloorDark_Iso"])],reference="/root/capsule/results/CMOS_Floor.png"),
                QCMetric(name="Floor average signal in Red channel", value=float(RedChFloorAve), status_history=[Bool2Status(Metrics["CMOSFloorDark_Red"])],reference="/root/capsule/results/CMOS_Floor.png")
            ],
            allow_failed_metrics=False,
            description="Pass when CMOS dark floor is <265 in all channel"
        )

        eval4 = QCEvaluation(
            name="Sudden change in signal",
            modality=Modality.FIB,
            stage=Stage.RAW,
            metrics=[
                QCMetric(name="1st derivative of Green channel", value=float(np.max(np.diff(data1[10:-2,1]))), status_history=[Bool2Status(Metrics["NoSuddenChangeInSignal"])],reference="/root/capsule/results/raw_traces.png")
            ],
            allow_failed_metrics=True,
            description="Pass when 1st derivatives of signals are < 5000"
        )

        qceval_list = [eval0, eval1, eval2, eval3, eval4]
        qc = QualityControl(evaluations=qceval_list)
        qc.write_standard_file(output_directory="/results")


        #%% DocDB

        def query_docdb_id(asset_name: str):
            """
            Returns docdb_id for asset_name.
            Returns empty string if asset is not found.
            """

            # Resolve DocDB id of data asset
            API_GATEWAY_HOST = "api.allenneuraldynamics.org"
            DATABASE = "metadata_index"
            COLLECTION = "data_assets"

            docdb_api_client = MetadataDbClient(
            host=API_GATEWAY_HOST,
            database=DATABASE,
            collection=COLLECTION,
            )

            response = docdb_api_client.retrieve_docdb_records(
            filter_query={"name": asset_name},
            projection={"_id": 1},
            )

            if len(response) == 0:
                return ""
            docdb_id = response[0]["_id"]
            return docdb_id

        docdb_id = query_docdb_id(os.path.basename(sessionfoldername))
