# Introduction

This repository contains all the code used in the AID-telerehab research project. The README below details how to reproduce our work, from pre-training ResNet-50 on MoVi motion capture data, to re-training for KIMORE exercise classification and assessment, to integrating this network in a DepthAI pipeline for running the system on OAK-D hardware.

In this repository, you will find:
- `docs`: the poster, thesis and paper written about our project
- `movi`: code to transform MoVi files into time-pose images, and subsequently to pre-train ResNet-50 on motion capture data 
- `kimore`: code to transform KIMORE files into time-pose images and exercise ratings, and subsequently to re-train our network to classify and assess KIMORE rehabilitation exercises
- `depthai-blazepose-telerehab`: the working AID-telerehab pipeline - a fork of an existing DepthAI BlazePose repository, integrated with our telerehabilitation system


# MoVi

The publicly-available MoVi dataset can be found [here](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/JRHDRN). This project uses the two files `F_Subjects_1_45.tar` and `F_Subjects_46_90.tar`, which both contain subject data as `.mat` files. We unzipped and combined these `.tar` files into one folder named `F_Subjects_Data`. 

Another essential tool from MoVi is a utilities file `utils.py` found in the [MoVi-Toolbox repo]("https://github.com/saeed1262/MoVi-Toolbox/blob/master/"). We use two functions from this script: `mat2dict` and `dict2ntuple`.

To reproduce our work, download the `utils.py` file to the `MoVi` repo, and move the `F_Subjects_Data` folder into this repo too.

These are the scripts included in our MoVi folder:
- `dataset_info.py`: prints out information about the mat files
- `dataset_info.sh`: calls `dataset_info.py` on all `.mat` files in `F_Subjects_Data`
- `movi2img.py`: generates an image from a 120-frame window of each MoVi motion capture sequence, after applying normalisation to the skeleton 
- `movi2img.sh`: calls `mocap2img.py` on all `F_Subjects_Data`
- `movi_resnet.py`: trains a resnet image classification network to classify MoVi mocap actions from the image data. Uses different parameters, saves each model and also saves loss and accuracy graphs to allow for choosing the model with the best set of parameters.


# KIMORE

The publicly-available KIMORE dataset can be found [here](https://vrai.dii.univpm.it/content/KiMoRe-dataset). There are two folders: CG and GPP, containing the control group and patient group data respectively. To reproduce our work, download both the CG and GPP folders to the KIMORE repo.

These are the scripts included in our KIMORE repo:
- `dataset_analysis.ipynb`: collates all KIMORE data into a DataFrame and plots the data in various graphs
- `kimore2img.py`: generates an image from 90-frame sliding windows WITH 45-FRAME OVERLAP of each KIMORE motion capture sequence, after applying normalisation to the skeleton
- `kimore_preprocessing.ipynb`: generates a `.h5` list of dicts, where each dict contains an image and its data (subject, exercise, clinical score, rating, etc). This object is used as input to train the neural network.
- `kimore_resnet.py`: retrains the pre-trained Resnet-50 network to classify both KIMORE exercises and their ratings from the image data.

Samples of the first and
last 90 frames of each sequence were omitted, to prevent the use of data in which
the subject was not constantly performing the exercise.



# DepthAI-BlazePose Integration

This is a fork of an existing DepthAI-BlazePose Github repository, which can be found [here](https://github.com/geaxgx/depthai_blazepose).

To use the pre-trained ResNet-50 network in this pipeline, the TensorFlow model had to be converted to a BLOB file. We used an online Blob Converter tool to do this. 

To run our work, convert your saved `.h5` model to a `.blob` file, and place the converted model in the `models` directory. 

These are the scripts that we have edited, which differ from those in the original implementation:
- `BlazeposeDepthaiEdge_WorkingPipeline.py`: a pipeline for running BlazePose on-device. We added functionality to collect pose data in a 90-frame buffer, convert this data into RGB images, and input the images to our pre-trained ResNet network.
- `template_manager_script_withRehab.py`: a script for managing the inputs and outputs of the OAK-D and host systems. We appended the exercise and rating outputs to a dictionary of data sent to the host computer, enabling the display of this information to the subject.
- `main_test.py`: found in the `examples` directory. This is the Python file used for data collection. The system instructs the subject on which exercise to perform at a time, and saves the results of data collection in a `.csv` file. 

