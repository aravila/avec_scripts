import os, sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../util'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
from frontend import Audiofrontend, MFStats
import numpy as np


def listfolders(pathfeat, pathfeatstats, pathfeatmovstats, pathfeatcombstats):
    if(len(sys.argv) < 3):
        print("\nUsage:")
        print("python extract_srmr.py ./condition_folders 0")
        print("Where")
        print("param1: path")
        print("param2: '0' for clean, '1' for noise and '2' enhanced")
        sys.exit(0)

    path = sys.argv[1:][0]
    mode = sys.argv[2:][0]
    for dirname in os.listdir(path):
        feat = ""
        featstats = ""
        featmovstats = ""
        featcombstats = ""
        if os.path.isdir("%s/%s"%(path, dirname)):
            if mode == '0':
                print("clean")
                feat = "%s/%s" % (pathfeat,"clean/arousal")
                featstats = "%s/%s" % (pathfeatstats,"clean/arousal")
                featmovstats = "%s/%s" % (pathfeatmovstats,"clean/arousal")
                featcombstats = "%s/%s" % (pathfeatcombstats,"clean/arousal")
            elif mode == '1':
                print("noise")
                feat = "%s/%s/%s/%s" % (pathfeat,"noise",dirname,"arousal")
                featstats = "%s/%s/%s/%s" % (pathfeatstats,"noise",dirname,"arousal")
                featmovstats = "%s/%s/%s/%s" % (pathfeatmovstats,"noise",dirname,"arousal")
                featcombstats = "%s/%s/%s/%s" % (pathfeatcombstats,"noise",dirname,"arousal")
            elif mode == '2':
                print("enhanced")
                feat = "%s/%s/%s/%s" % (pathfeat,"noise",dirname,"arousal")
                featstats = "%s/%s/%s/%s" % (pathfeatstats,"noise",dirname,"arousal")
                featmovstats = "%s/%s/%s/%s" % (pathfeatmovstats,"noise",dirname,"arousal")
                featcombstats = "%s/%s/%s/%s" % (pathfeatcombstats,"noise",dirname,"arousal")
            convert(path, dirname, feat, featstats, featmovstats, featcombstats)

def get_no_examples(mf, no_examples = 7501):
    if len(mf) < no_examples:
        tmp = np.reshape(mf[len(mf) - 1], (1, len(mf[len(mf) - 1])))
        tmp = np.repeat(tmp, no_examples - len(mf), axis=0)
        mf = np.vstack((mf, tmp))
    else:
        mf = mf[0:no_examples]
    return mf

def write_arff(path, file, features):
    f = open("%s/%s"%(path, file), "w")
    f.write("@RELATION %s\n"%(file[:-5]))
    f.write("\n")

    for col in range(0, len(features[0])):
        f.write("@attribute att%s numeric\n" % col)

    f.write("\n")
    f.write("@data\n")
    f.write("\n")

    for row in range(0, len(features)):
        str = ""
        for col in range(0, len(features[0])):
            str += "%d, "%(features[row][col])
        print(str[:-2])
        f.write("%s\n" % str[:-2])

    f.close()


def convert(path, dirname, featpath, featstatspath, featmovstatspath, featcombstatspath):
    audio = Audiofrontend()
    dirs = os.listdir("%s/%s"%(path, dirname))
    for file in dirs:
        if file.endswith('.wav'):
            print('%s/%s' % (path, file))
            mf = audio.srmr_audio('', path, file)
            mf = np.einsum('ijk->kij', mf)
            mf = get_no_examples(mf,7501)
            stats = MFStats(mf)

            mfeat = np.reshape(mf, (mf.shape[0], mf.shape[1] * mf.shape[2]))
            write_arff(featpath, "%s%s"%(file[:-4], ".arff"), mfeat)

            mfeatstats = stats.get_stats()
            write_arff(featstatspath, "%s%s"%(file[:-4], ".arff"), mfeatstats)

            mfeatmovstats = stats.moving_stats(mfeat, 10)
            write_arff(featmovstatspath, "%s%s"%(file[:-4], ".arff"), mfeatmovstats)

            mfeatcombstats = stats.moving_stats(mfeatstats, 10)
            write_arff(featcombstatspath, "%s%s"%(file[:-4], ".arff"), mfeatcombstats)


pathfeat = 'F:/AVEC/mf_features'
pathfeatstats = 'F:/AVEC/mf_features_stats'
pathfeatmovstats = 'F:/AVEC/mf_features_movstats'
pathfeatcombstats = 'F:/AVEC/mf_features_combstats'

listfolders(pathfeat, pathfeatstats, pathfeatmovstats, pathfeatcombstats)



