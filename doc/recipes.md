
# Process for a single recording

## Generate a time-series

The basic processing method generates time-series for every channel in the recording. 

```bash
python -m clemotion.signal.ts_gen_from_csv single E:\Data\2021_SensorClarinet\original\20211214\6-ResonantFingerings\ResonantFingerings.wav -o E:\Data\2021_SensorClarinet\intermediate\20211214\6-ResonantFingerings\ResonantFingerings_ts.pickle -c ..\runsheets\channel_desc.yaml -s endevco2
```

Argument description:
* `single` is the command to process a single file
* The main argument is the source wavfile
* `-o` indicates the destination filename, a python pickle file with several timeseries for each channel
* `-c` a YAML file with description of channels, containing several channel sets
* `-s` the name of the channel set to use from the YAML file

## Segment notes and match to a score

### Single melody played a few times

Method used to process recordings with several repetitions of a single melody

The following format uses the score from a YAML file containing several melodies. It segments the amplitude and frequency time-series from a time-series file generated in the previous step, and then tries to match to several repeats of the same melody.


### Several melodies in a recording

In this method the recording has several melodies, and a TextGrid file contains labels for each melody. The melody is the label of the melody to extract from the YAML file

```bash
python note_matcher.py clips E:\Data\2021_SensorClarinet\intermediate\20211203\6-ResonantFingerings\ResonantFingering_ts.pickle -m ..\runsheets\melodies.yaml -t E:\Data\2021_SensorClarinet\intermediate\20211203\6-ResonantFingerings\ResonantFingering_tunes.TextGrid -o E:\Data\2021_SensorClarinet\intermediate\20211203\6-ResonantFingerings\ResonantFingering_notes.TextGrid
```

Argument description:
* `clips` directs the script to read the melody names from a TextGrid file
* the **main argument** is the python pickle file with time series 
* `-m` the YAML file with melody scores
* `-t` the TextGrid file with excerpt-level segmentation, melody names in one of the tiers (other information than the melody name may be included in the labels, e.g. beethoven - happy)
* `-o` output TextGrid with note-level segmentation. The original tiers are also included

## Process time series for a set of recordings

Generate a `csv` file with the list of recordings, and channel sets, then

```bash
python -m clemotion.signal.ts_gen_from_csv csv -o E:\Data\2021_SensorClarinet\intermediate -r E:\Data\2021_SensorClarinet\original -c runsheets\channel_desc.yaml E:\Data\2021_SensorClarinet\original\wav_melody_list_manual.csv
```

## Build a note database for a series of recordings

Build a CSV file with the list of recordings. Then

```bash
python build_note_database.py "E:\Data\2021_SensorClarinet\original\Resonant_wav_list.csv" -m ..\runsheets\melodies.yaml -r E:\Data\2021_SensorClarinet\intermediate
```

Argument list:
* **main argument**: a CSV file containing a coulmn `wavefile` with the recored files
* `-r` the root folder for the timeseries and TextGrid files
* `-m` a YAML file with melody scores

# Older processes

This are older methods based on YAML runsheets. 

### Process all recordings in a folder

For example, a new subject comes in and we need to process all the recordings
in the new folder

### Create a **runsheet**

Pick an existing runsheet and modify the parameters, in particular the root
folder, then check the filenames. Remove the textgrid mentions

### Run the segmenters

This is an old segmenter. 

```bash
python excerpt_segmenter.py ..\runsheets\runsheet_20210526.yaml  -d "E:\Data\2021_SensorClarinet"
```


