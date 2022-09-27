import argparse
import tgt


def parse_args():
    # Same main parser as usual
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='verbose', action='store_true')

    sub_parsers = parser.add_subparsers(dest='command', help='commands', required=True)    
    
    parser_sil= sub_parsers.add_parser('rename', help='Estimate silence and voice thresholds')
    parser_sil.add_argument('file', help='timeseries file')

    parser_notes = sub_parsers.add_parser('notes', help='Segment notes')
    parser_notes.add_argument('file', help='timeseries file')
    parser_notes.add_argument('-m','--melody-file',  help='YAML file containing melodies (leave out for note segmentation without alignment)')
    parser_notes.add_argument('-t','--tune',  help='melody name')
    parser_notes.add_argument('-o','--output',  help='output file')
    
    parser_csv = sub_parsers.add_parser('csv', help='File list from csv')
    parser_csv.add_argument('file', help='csv file (contains file, melody columns)')
    parser_csv.add_argument('-m','--melody-file',  help='YAML file containing melodies (leave out for note segmentation without alignment)')
    parser_csv.add_argument('-r','--root-dir', help='root folder (defaults to .)',
                            default='.')
    parser_csv.add_argument('-o','--output',  help='output dir')

    return parser.parse_args()