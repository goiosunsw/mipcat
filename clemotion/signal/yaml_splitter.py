"""
Perform the segmentations of multiple files into independent musical excerpts and output them as text grid files
"""

import sys
import yaml

with open(sys.argv[1]) as f:
    all_yaml = yaml.full_load(f)

for k, v in all_yaml.items():
    print(k)
    with open(k+'.yaml','w') as outf:
        yaml.dump(v, outf)
         




