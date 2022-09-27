# Emotion in music processing

This project contains a set of scripts and configurations 
to analyse signals and videos with recordings of clarinet 
performances

# Install

Create a conda environment with the file `environment.yml`

```python
conda env create -f environment.yml
```

# Workflow

For a recent workflow see [these recipes](doc/recipes.md)

## Video

### Mouthpiece videos

#### Color based processing

Use `video_processing/mouthpiece_color_based_process.py` whenever 
there is a patch with a distinctive color (green) that can be measured 
in the video

Use `video_gui.py` to set the parameters for this script

### Performer videos 

These are front and side videos of the performer. 

There are different tracking methods: 

* `video_processing/face/face_detect.py` for face feature tracking
* `video_processing/aruco_tracker.py` for clarinet tracking
* *still to come* for tracking the clarinet without markers (based on face + blob tracking)

Also in progress, a neural network to track clarinet without further references

### Synchronising video to main signals

The main signals are recorded by the DAQ system, with several minute length excerpts.

These must be synchronised to the video signals. For this use `signal_processing/sound_align.py`

### Collecting signals

Signals for clarinet, face and mouthpiece are collected into pickle files, that are added to 
the recording runsheet



## Signal processing

Recording data and parameters are collected in **yaml** files, 
in the `runsheets` folder.

These are used by the collection scripts to generate the database.

### Collection scripts

* `collection/excerpt_segmenter.py` does a basic segmentation using the external audio, generating Praat Textgrids that can be editted in Praat. This should be run twice:
  * 1st pass generates excerpt regions
  * 2nd pass, after adjusting excerpt regions in praat, to correctly identify the notes corresponding to each excerpt. These can be further checked in praat.
* `collection/timeseries_generator.py` generates various timeseries from raw audio, such as 
  * fundamental frequency
  * amplitude envelope
  * DC components, etc.
* `collection/build_note_database.py` collects signal statistics within each note, generating 3 tables:
  * `melody_scores.csv` contains a list of all notes for all excerpts
  * `versions_played.csv` contains information about all the versions of each excerpt played by each subject.
  * `played_notes.csv` contains all the events recorded in all sessions

