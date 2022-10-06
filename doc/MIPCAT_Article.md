# Generating article figures

This page refers to the the example dataset that is provided as supporting material for the article

Almeida, A., Li W., Schubert, E., Smith, J. and Wolfe, J. "MIPCAT -- a Music Instrument Performance Capture and Analysis Toolbox"

The dataset contains several performances by 2 professional and 1 amateur musician of the first bars of Mozarts Clarinet Concerto K.622. For the 2 professionals, there are 3 excerpts, 2 played with the customised clarinet from our Laboratory, fitted with sensors and one played with their own clarinet.

For the amateur, only one excerpt is given, but with the original videos. These are meant to show how to use the video feature extraction tools and alignment with other signals. 

The file structure of the dataset is as follows:


mozart_sample
- `channel_desc.yaml`: Description of channels in each `.wav`-file
- `melodies.yaml`: Note sequence expected in recordings
- `mozart_notes.csv`: Segmentation of recordings (gathered together)
- `P5`: Professional player A
    - `Lab`: Recording with sensor-fitted clarinet
        - `P5_Mozart_Lab_Front.json`: Clarinet marker positions extracted from front view camera
        - `P5_Mozart_Lab_Mouthpiece.json`: Mouthpiece covering measurement  
        - `P5_Mozart_Lab_Side.json`: Clarinet marker positions extracted from side view camera
        - `P5_Mozart_Lab_ts.pickle`: Time-series extracted from `.wav` file
        - `P5_Mozart_Lab.wav`: Sensor recordings 
    - `Own`: Recording with player's own instrument (similar as `Lab` but sensor channels absent and no mouthpiece camera)
        - `P5_Mozart_Own_Front.json`
        - `P5_Mozart_Own_Side.json`
        - `P5_Mozart_Own_ts.pickle`
        - `P5_Mozart_Own.wav`
- `P6`: Same structure as `P5`
    - ...
- `P10`: Amateur recording
    - `Lab`
        - `P10_Mozart_Lab_Front.MP4`
        - `P10_Mozart_Lab_Mouthpiece.mp4`
        - `P10_Mozart_Lab_Side.MP4`
        - `P10_Mozart_Lab.wav`



# Regenerating intermediate files from sources

All the intermediate data required for plotting the figures in the article are provided in the dataset, however they can be regenerated using the code in this repository and following the recipes below. They involve recalculating sound descriptors and a note-by-note segmentation of the recording.

After creating the mipcat environment as described in the [README file](../README.md), run

```bash
conda activate mipcat
```

## Generate a Time-series

`_ts.pickle` files contain sound descriptors extracted from the original signals in the `.wav` file. These are mostly calculated with at constant intervals of 10ms. They can be recalculated by running the following command in a folder with a `.WAV` file, for instance `P5/Lab/`: 

```bash
python -m mipcat.signal.ts_gen_from_csv single -c ../../channel_desc.yaml -o P5_Mozart_Own_ts.pickle -s ext_only P5_Mozart_Own.wav
```

## Sgment acording to a melody

Note-by-note segmentation is provided in the file `mozart_notes.csv` in the root of the dataset. The most important data in this file are the start and end times of each note, but the file also includes metadata such as which participant played each note, with which instrument, and the expected note as per the score.

To regenerate this file, first the note-by-note segmentation and excerpt segmentation have to be obtaine for each recording. This is done by aligning the score to the recording, so that knowledge of the score is needed and provided by the file `melodies.yaml` in the root folder of the dataset. To obtain a segmentation that can be checked and edited in `praat`, use the following, in the folder with the timeseries generated above (eg `P5/Lab/`):

```bash
python -m mipcat.signal.note_matcher notes -m ../../melodies.yaml  -t mozart -o P5_Mozart_Own.TextGrid P5_Mozart_Own_ts.pickle
```

## Adjust TextGrid if needed 

Use ***praat***

- Open both the WAV file and the TextGrid file with the `Open` menu in Praat main window
- Select both files and click *View and edit*
- Check that each note is well bounded and adjust the region corresponding for the full excerpt


## Repeat for every recording 

Repeat the 3 above steps for every `.wav` file in the dataset. 

# Video tasks

For privacy concerns, only one example is provided with video files. For the remaining, processed data is provided.

## Adjust mouthpiece video settings

This step creates a configuration file that selects a reference region in the mouthpiece and a color range matching the green strip in the mouthpiece. 

The reference region should be a unique object that can be followed throughout the whole movie. The number *25* is usually a good choice, as it is usually not covered by the musician's mouth and is distinguishable from other parts of the image. 

For the color range, the hue minimum and maximum values are the most important. They can vary slightly during the video as lighting conditions change, so scan through the movie to make sure that none of the green region is missing. The minimum saturation value can be increased slightly to prevent grey areas from connecting to the detected region. Try different settings to see the effects.

In the folder `P10/Lab`, run:

```bash
python -m mipcat.video.mouthpiece_video_gui P10_Mozart_Lab_Mouthpiece.mp4
```

- Adjust the hue values so that the green strip is easily separated from other detected areas
- Slightly increase the minimum saturation
- Select a rectangle including the number *25* in the video 
- Activate the *Process* check box
- Select a rectangle including the number *25* (again, this is because of a bug)
- Scroll through the video and check that the detection works for most of the frames (especially those where the player has his mouth on the mouthpiece)
- Click the *Write* button to write the video configuration

A configuration file should have been created with the name `P10_Mozart_Lab_Mouthpiece_conf.json`

## Run the mouthpiece covering measurement

Run 
```bash
python -m mipcat.video.mouthpiece_tracker P10_Mozart_Lab_Mouthpiece.mp4
```

A measurement file should have been created with the name `P10_Mozart_Lab_Mouthpiece_video_analysis.json`

## Run the clarinet tracker for the GOPRO videos

For each of the `_Front` and `_Side` video files, run:

```bash
python -m mipcat.video.aruco_tracker P10_Mozart_Lab_Front.MP4
```

A file should be created with the name `P10_Mozart_Lab_Front.json` with the positions of the detected markers.

Repeat for `P10_Mozart_Lab_Side.MP4`

# Find the delay between the signal file and the videos

Run
```bash
python -m mipcat.align.align_keypoints -t -c 5 P10/Lab/P10_Mozart_Lab_Front.MP4 P10/Lab/P10_Mozart_Lab.WAV >> wav_video_delays.csv
```

This calculates the delay between the `WAV` recording and the video recording, using channel 6 (numbering starts at 0) in the `WAV` file and adds a line to the `wav_video_delays.csv` file containing both filenames, the channel, the number of seconds to add to the time into the video file to obtain the time in the signal file, and a number indicating the quality of the audio match.


# Collect all the note regions into a general note database

From the root folder of the dataset (`mozart_sample`)

```bash
python -m mipcat.signal.textgrid_collector . > mozart_notes.csv
```

This generates a CSV file `mozart_notes.csv` with the source testgrid files, the excerpt label and the note label as well as start and end times. 

