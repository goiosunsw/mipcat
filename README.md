# MIPCAT -- a Musical Instrument Performance Capture and Analysis Toolbox

MIPCAT is a toolbox to capture different physical aspects of performance on a musical instrument and analyse the resulting data.

The project consists of hardware and software modules. This repository mostly organises the different software components. Blueprints for parts of the hardware are also given in the Hardware folder

# Installation
- Install Anaconda if you don't have it. Anaconda is a scientific python distribution available for Windows, Mac OS and Linux
- Open a conda terminal
- Download (or clone) this repository and decompress it. It should create a `mipcat` folder
- Change into the `mipcat` folder
- Run `conda env create -n mipcat -f environment.yaml`

# Example: MIPCAT article figure generation

If you read article [1], the figures can be generated following the procedure in [doc/MIPCAT_Article.md](doc/MIPCAT_Article.md). 

Download the sample dataset [here](https://cloudstor.aarnet.edu.au/plus/s/1d1oeOAmsJU4nJ2). When prompted, use the password `MIPCAT_unsw2022`

[1] Almeida, A., Li W., Schubert, E., Smith, J. and Wolfe, J. "MIPCAT -- a Music Instrument Performance Capture and Analysis Toolbox"

# Workflow

This section is more generic than the previous one and can in principle be used with different dataset, including, in future, a complete dataset of recordings performed in the context of the ARC research project `DPDP200100963`.

Everything has to be run inside the `mipcat` conda environment.

For this workflow, please change folders to the `mipcat` root folder (created when extracting the code archive)

- Open a terminal in the `mipcat` environment in Anaconda
or run
```
conda activate mipcat
```
in a command prompt or shell


## Create metadata lists
3 different files are needed for the following sequence:

### Channel description
A YAML file containing the description of the channels. Several descriptions may be included if channel configurations changed along the project.

A YAML channel file for the example data exists at `runsheets/channel_desc.yaml`
### List of recordings
A CSV file containing a list of recordings, with subject IDs, channel configuration sets, and music scores corresponding to the recording

An example CSV file is given at `runsheets/wav_list.csv`

### Music scores
A YAML file with music scores

An example score file is given at `runsheets/melodies.yaml`

## Auto-generate descriptor time-series
Documentation of time-series generation scripts: [[Time-series generation]]

To generate time-series for all the recordings
```bash
python -m mipcat.signal.ts_gen_from_csv csv -c metadata/channel_desc.yaml -r original -o calc metadata/wav_list.csv
```
Where the main argument is now the CSV file containing the list of wave files to process, and the `-o` switch informs about the output folder where the time-series are stored.

## Audio and Video alignment
Video and audio do not have a synchronisation signal. They are synchronised using audio
An automated way of synchronising audio and video is provided through fingerprinting.
To synchronise a video file to the main signals use:
```bash
	python -m mipcat.align.align_keypoints -c 6 original/S0/S0_Mozart_Mouthpiece.mp4 original/S0/S0_Mozar_Signals.wav
```
This will output the delay and matching accuracy (about >5% is good).

## Video time-series
Video time-series are not provided for our main participants for ethics reasons. An example is given for one of the researchers. For the renaming, extracted data is provided.

### Mouthpiece video
#### Color and reference calibration
The analysis of mouthpiece covering requires calibration of the color range in which the green strip is found and a reference in the scale (usually the number 25). Calibration is done using the GUI: run
```bash
python -m mipcat.video.mouthpiece_gui data/S0/S0_Mozart_Mouthpiece.mp4
```
- The most important setting are the top and bottom values of the Hue. 
- Provide a rectangular selection of the scale reference or click "find template" for an automated attempt.
- Move the slider underneath the video to check that the color range is adequate for the entire length of the video
- Click **write** to write the configuration to a JSON file

#### Run the analysis
```bash
python -m mipcat.video.mouthpiece_template data/S0/S0_Mozart_Mouthpiece.mp4
```
This will generate a JSON file with the the analysis of exposed green area for each frame.

### Player and instrument views
For the analysis of instrument motion using the AruCo tags, run
```bash
python -m mipcat.video.aruco_tracker data/S0/S0_Mozart_Front.mp4
```
and for the side view (where caemera was rotated 90 degrees):
```bash
python -m mipcat.video.aruco_tracker -r 270 data/S0/S0_Mozart_Side.mp4
```

## Segment 
Documentation of segmentation scripts: [[Segmentation]]

To segment all recordings declared in the CSV, directed by a musical score:
```bash
python -m mipcat.segment.segment runsheets/wav_list.csv -o output/ -m runsheets/melodies.yaml
```

[Praat](https://www.fon.hum.uva.nl/praat/) TextGrid files are generated for each recording. These can be adjusted in Praat to correct for wrongly detected notes

## Note features
The next step is to extract note features and build a note database

```bash
python -m mipcat.db.build runsheets/wav_list.csv -i 
```
