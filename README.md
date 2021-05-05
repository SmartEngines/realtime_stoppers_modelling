# Supplementary materials

This repository contains supplementary materials for the paper "Determining optimal frame processing strategies for real-time document recognition systems".

In this supplementary materials you can find the necessary input data and code to fully reproduce the experiments conducted in the paper. To run the analysis and construct the plots, run `realtime_modelling.ipynb`. For our experiments the code was executed used Python 3.9.5 running under JupyterLab version 3.0.7.

Description of each source code file and data directory follows.

## Code overview

1\. `realtime_modelling.ipynb` - Jupyter notebook containing the code for reproducing the experimental evaluation and constructing the plots;

2\. `metrics.py` - python module containing implementations for basic functions required for the experiments;

3\. `combination.py` - python module containing implementation of the text string combination algorithm, used for computing the expected distance estimation using the base stopping method;

4\. `combination_with_estimation.py` - python module containing a modified implementation of the text string combination algorithm with fast approximations of the stopping method;

5\. `treap.py` - python module containing implementation of a balanced binary search tree - treap with random priorities;

6\. `model_basic.py` - python module containing implementations of the combination algorithm models for full results combination, single best selection, and combination of the three best samples;

7\. `model_realtime.py` - python module containing implementations of the combination strategies models with real-time stoppers application.

## Overview of data directories

1\. `midv500_ocr/` - directory with text field recognition results, used as a dataset for the experiments. Fields were taken from MIDV-500 dataset, recognized using text field recognition subsystem of Smart IDReader. Each field clip is stored in a Pickle format and has the filename format `<FIELDTYPE>_<CLIPID>_<FIELDNAME>.pkl`. Ground truth is provided in `midv500_ocr/gt.json`.

2\. `midv500_ocr/cache/` - directory containing the cached modelling events, which are used to perform the analysis and constructing the plots in `realtime_modelling.ipynb`. If the notebook is launched directly, the cached values are used and only the plots are constructed. If the `cache` subdirectory is removed before the notebook is launched, the full modelling is performed and the `cache` subdirectory will be recreated.

3\. `midv500_focus.json` - a file with focus estimations for frames in MIDV-500.