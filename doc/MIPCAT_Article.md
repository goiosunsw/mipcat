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
python 
```

# Collect all the note regions into a general note database


