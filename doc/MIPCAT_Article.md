# Regenerating intermediate files from sources

## Generate a Time-series

Run
```bash
python -m mipcat.signal.ts_gen_from_csv single -c ../../channel_desc.yaml -o P5_Mozart_Own_ts.pickle -s ext_only P5_Mozart_Own.wav
```

## Sgment acording to a melody

Run
```bash
python -m mipcat.signal.note_matcher notes -m /opt/conda/lib/python3.10/site-packages/mipcat/resources/melodies.yaml  -t mozart -o P5_Mozart_Own.TextGrid P5_Mozart_Own_ts.pickle
```

## Adjust TextGrid if needed 

Use ***praat***

- Open both the WAV file and the TextGrid file with the `Open` menu in Praat main window
- Select both files and click *View and edit*
- Check that each note is well bounded and adjust the region corresponding for the full excerpt


## Repeat for every recording 

# Video tasks

For privacy concerns, only one example is provided with video files. For the remaining, processed data is provided.

## Adjust mouthpiece video settings

Run
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
python -m mipcat.align.align_keypoints -c 5 P10_Mozart_Lab_Front.MP4 P10_Mozart_Lab.WAV
```

This calculates the delay between the `WAV` recording and the video recording, using channel 6 (numbering starts at 0) in the `WAV` file.

A bunch of warnings appear, and the last line should say

`27.813179131929964 sec, 48.29 % matches`

This means that the beginning of the video file is **27.81 seconds** into the `WAV` file


# Collect all the note regions into a general note database

From the root folder of the dataset (`mozart_sample`)

```bash
python -m mipcat.signal.textgrid_collector . > mozart_notes.csv
```

This generates a CSV file `mozart_notes.csv` with the source testgrid files, the excerpt label and the note label as well as start and end times. 

