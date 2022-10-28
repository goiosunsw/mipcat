import pstats
import cProfile
import time

from mipcat.video.mouthpiece_tracker import MouthpieceTracker
from tqdm import trange

mt = MouthpieceTracker()
mt.set_video_source('original/20211111/Endoscope/WIN_20211111_10_45_32_Pro.mp4')
mt.read_config('intermediate/20211111/Endoscope/WIN_20211111_10_45_32_Pro_conf.json')

profs = []
times = [0.3]
p=cProfile.Profile()

for ii in trange(20000):
    ts=time.time()
    p.enable()
    mt.process(mt.get_frame()[1])
    p.disable()
    dt=time.time()-ts
    if dt>mean(times)*3:
        profs.append(pstats(p))
        print(dt,mt.time)
    p.clear()
    times.append(dt)

