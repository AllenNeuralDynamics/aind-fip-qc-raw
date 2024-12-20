# aind-fip-qc-raw (ver 0.1)

QC capsule for fiber photometry (modality:fib, device:fip) raw data acquired together with HARP/Bonsai-based behavior (e.g. Dynamic Foraging)

run_capsule.py : main script

run_capsule_dev.py : for develipment, in VScode + jupyter-extension env.

Following the "alternate-workflow" with which you don't need to make a new asset, instead directly pushing QC.json to DocDB.

https://github.com/AllenNeuralDynamics/aind-qc-portal?tab=readme-ov-file#alternate-workflow

___
**Steps:**

1.reading rawdata

2.generating figures and metrics

3.submitting figures to kachery to obtain unique url

4.Composing QC/QCevals/QCmetrics

5.Pushing QC,josn to DocDB

6.visualizing, manual QCing under AIND-QCportal-app

https://qc.allenneuraldynamics.org/qc_portal_app

