import sys
import tgt

args = sys.argv[1:]

def fix_utf(x):
    return str(x).replace('\u266f','#').encode('utf-8').decode('utf-8')

print("text_grid,clip,start,end,label")

for ii, arg in enumerate(args):
    tg = tgt.read_textgrid(arg)
    for tiername in ['note','notes','Notes','Note']:
        try:
            tier = tg.get_tier_by_name(tiername)
        except ValueError:
            pass
    for tiername in ['clip','clips','Clips','Clip']:
        try:
            clip_tier = tg.get_tier_by_name(tiername)
        except ValueError:
            pass
            
    for note in tier:
        start = note.start_time
        end = note.end_time
        try:
            clip = clip_tier.get_annotations_by_time((start+end)/2)[0]
        except IndexError:
            clip = ""
        print(f"{arg},{fix_utf(clip.text)},{start:.3f},{end:.3f},{fix_utf(note.text)}")
