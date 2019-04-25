import os
import sys

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
    simplify_dataset(sys.argv[1])
