# aind-fip-qc-raw

QC capsule for fiber photometry currently in development. The capsule creates metrics and writes a `quality_control.json`, that can be visualized for manual inspection. More information can be found here: https://github.com/AllenNeuralDynamics/aind-qc-portal.

### Input
The input is a fiber photometry asset acquired following a standard defined here: [Fiber Photometry acquisition standard](https://github.com/AllenNeuralDynamics/aind-file-standards/blob/main/file_formats/fip.md).

### Output
The output is a `quality_control.json` that contains several metrics and evaluations for QCing fiber photometry data. In addition, the following list of images are generated in a `qc-raw` folder:

```plaintext
📦qc-raw
┣ 📜CMOS_Floor.pdf
┣ 📜CMOS_Floor.png
┣ 📜raw_traces.pdf
┣ 📜raw_traces.png
```


