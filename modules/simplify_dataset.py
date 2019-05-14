import os
import sys
import argparse


def process_file(filename):
    ''' Simplify .PHN file from 61 to 39 phonemes. '''
    with open(filename, "rt") as fin:
        with open(filename+"tmp", "wt") as fout:
                for line in fin:
                    if 'ao' in line:
                        fout.write(line.replace('ao', 'aa'))
                    elif 'ax' in line:
                        fout.write(line.replace('ax', 'ah'))
                    elif 'ax-h' in line:
                        fout.write(line.replace('ax-h', 'ah'))
                    elif 'ah-h' in line:
                        fout.write(line.replace('ah-h', 'ah'))
                    elif 'ahr' in line:
                        fout.write(line.replace('ahr', 'er'))
                    elif 'axr' in line:
                        fout.write(line.replace('axr', 'er'))
                    elif 'hv' in line:
                        fout.write(line.replace('hv', 'hh'))
                    elif 'ix' in line:
                        fout.write(line.replace('ix', 'ih'))
                    elif 'el' in line:
                        fout.write(line.replace('el', 'l'))
                    elif 'em' in line:
                        fout.write(line.replace('em', 'm'))
                    elif 'en' in line:
                        fout.write(line.replace('en', 'n'))
                    elif 'nx' in line:
                        fout.write(line.replace('nx', 'n'))
                    elif 'eng' in line:
                        fout.write(line.replace('eng', 'ng'))
                    elif 'zh' in line:
                        fout.write(line.replace('zh', 'sh'))
                    elif 'ux' in line:
                        fout.write(line.replace('ux', 'uw'))
                    elif 'q' in line:
                        fout.write(line.replace('q', 'sil'))
                    elif 'pcl' in line:
                        fout.write(line.replace('pcl', 'sil'))
                    elif 'tcl' in line:
                        fout.write(line.replace('tcl', 'sil'))
                    elif 'kcl' in line:
                        fout.write(line.replace('kcl', 'sil'))
                    elif 'bcl' in line:
                        fout.write(line.replace('bcl', 'sil'))
                    elif 'dcl' in line:
                        fout.write(line.replace('dcl', 'sil'))
                    elif 'gcl' in line:
                        fout.write(line.replace('gcl', 'sil'))
                    elif 'h#' in line:
                        fout.write(line.replace('h#', 'sil'))
                    elif 'pau' in line:
                        fout.write(line.replace('pau', 'sil'))
                    elif 'epi' in line:
                        fout.write(line.replace('epi', 'sil'))
                    else:
                        fout.write(line)
    os.remove(filename)
    os.rename(filename+"tmp", filename)


def simplify_phonemes_file(filename):
    ''' Simplify phonemes.txt file from 61 to 39 phonemes. '''
    with open(filename, "rt") as fin:
        with open(filename+"tmp", "wt") as fout:
                for line in fin:
                    if 'ao' in line or \
                        'ax' in line or \
                        'ax-h' in line or \
                        'ah-h' in line or \
                         'ahr' in line or \
                         'axr' in line or \
                         'hv' in line or \
                         'ix' in line or \
                         'el' in line or \
                         'em' in line or \
                         'en' in line or \
                         'nx' in line or \
                         'eng' in line or \
                         'zh' in line or \
                         'ux' in line or \
                         'q' in line or \
                         'pcl' in line or \
                         'tcl' in line or \
                         'kcl' in line or \
                         'bcl' in line or \
                         'dcl' in line or \
                         'gcl' in line or \
                         'pau' in line or \
                        'epi' in line:
                        pass
                    elif 'h#' in line:
                        fout.write('sil\n')
                    else:
                        fout.write(line)
    os.remove(filename)
    os.rename(filename+"tmp", filename)


def simplify_dataset(path):
    ''' Simplify dataset from 61 to 39 phonemes. '''
    # extract basename of files and remove duplicates
    filelist = os.listdir(path)
    for i in range(0, len(filelist)):
        filelist[i] = os.path.splitext(os.path.basename(filelist[i]))[0]
    filelist = list(dict.fromkeys(filelist))

    # process files
    for filename in filelist:

        # get list of directories
        files = os.listdir(path)
        files.sort()

        # if path points to directory do recursive call
        if(os.path.isdir(path + '/' + filename)):
            simplify_dataset(path + '/' + filename)

        # otherwise process files inside directory
        else:
	        process_file(path + '/' + filename + ".PHN")


if __name__ == "__main__":
    # initiate the parser
    parser = argparse.ArgumentParser()

    # add long and short argument
    parser.add_argument("--dataset",
                        "-d",
                        help="path to dataset root directory",
                        dest="dataset",
                        required=True)
    parser.add_argument("--phonemes",
                        "-p",
                        help="path to feasible phonemes file",
                        dest="phonemes",
                    required=True)

    # read arguments from the command line
    args = parser.parse_args()

    simplify_dataset(args.dataset)
    simplify_phonemes_file(args.phonemes)
