import argparse

def get_sample_rate(filename):
    """Get sample rate from WAV file

    Args:
        filename (str): name of wave file

    Returns:
        rate, Bps
        int: sample rate
        int: Bits Per Second
    """
    with open(args.wavfile,'rb') as f:
        head = (f.read(44))

    cur_rate = int.from_bytes(head[24:28],"little")
    cur_Bps =  int.from_bytes(head[28:32],"little")

    return cur_rate, cur_Bps

    
def fix_sample_rate(filename, rate):
    """Relplace sample rate in WAV file header
       WARNING: This is done in-place, no backup file is left behind

    Args:
        filename (str): name of wave file
        rate (int): new sample rate

    Returns:
        None
    """
    cur_rate, cur_Bps = get_sample_rate(filename)
    mult = cur_Bps//cur_rate

    with open(filename,'r+b') as f:
        f.seek(24)
        f.write(rate.to_bytes(4,"little"))
        f.write((new_sr*mult).to_bytes(4,"little"))

def parse_args():
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    optional = parser.add_argument_group('optional arguments')
    required = parser.add_argument_group('required arguments')
    required.add_argument('wavfile', help='WAV filename to edit')
    required.add_argument('-r', '--rate', type=int,
      help='New sample rate')
    optional.add_argument('-i', '--info', action="store_true",
      help='Print original info and exit (without modification)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    new_sr = args.rate


    if args.info:
        cur_rate, _ = get_sample_rate(args.wavfile) 
        print(f'Current sample rate: {cur_rate}')
        exit(1)
    
    if args.rate:
        fix_sample_rate(args.wavfile, args.rate)

    
 