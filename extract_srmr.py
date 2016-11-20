import os, sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../util'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
from frontend import Audiofrontend, MFStats
import numpy as np
import shutil


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
    dirname = ""
    if mode == '0':
        feat = "%s/%s" % (pathfeat, "clean/arousal")
        featstats = "%s/%s" % (pathfeatstats, "clean/arousal")
        featmovstats = "%s/%s" % (pathfeatmovstats, "clean/arousal")
        featcombstats = "%s/%s" % (pathfeatcombstats, "clean/arousal")
        convert(path, dirname, feat, featstats, featmovstats, featcombstats)
    else:
        for dirname in os.listdir(path):
            feat = ""
            featstats = ""
            featmovstats = ""
            featcombstats = ""
            if os.path.isdir("%s/%s" % (path, dirname)):
                if mode == '1':
                    feat = "%s/%s/%s/%s" % (pathfeat, "noise", dirname, "arousal")
                    featstats = "%s/%s/%s/%s" % (pathfeatstats, "noise", dirname, "arousal")
                    featmovstats = "%s/%s/%s/%s" % (pathfeatmovstats, "noise", dirname, "arousal")
                    featcombstats = "%s/%s/%s/%s" % (pathfeatcombstats, "noise", dirname, "arousal")
                elif mode == '2':
                    feat = "%s/%s/%s/%s" % (pathfeat, "noise", dirname, "arousal")
                    featstats = "%s/%s/%s/%s" % (pathfeatstats, "noise", dirname, "arousal")
                    featmovstats = "%s/%s/%s/%s" % (pathfeatmovstats, "noise", dirname, "arousal")
                    featcombstats = "%s/%s/%s/%s" % (pathfeatcombstats, "noise", dirname, "arousal")
                convert(path, dirname, feat, featstats, featmovstats, featcombstats)


def get_no_examples(mf, no_examples = 7501):
    if len(mf) < no_examples:
        tmp = np.reshape(mf[len(mf) - 1], (1, mf.shape[1], mf.shape[2]))
        tmp = np.repeat(tmp, no_examples - len(mf), axis=0)
        mf = np.vstack((mf, tmp))
    else:
        mf = mf[0:no_examples]
    return mf

def write_arff(path, file, features):
    if not os.path.exists(path):
        os.makedirs(path)
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
            str += "%f,"%(features[row][col])
        f.write("%s\n" % str[:-2])

    f.close()


def convert(path, dirname, featpath, featstatspath, featmovstatspath, featcombstatspath):
    audio = Audiofrontend()
    dirs = os.listdir("%s/%s"%(path, dirname))
    for file in dirs:
        if file.endswith('.wav'):
            print('%s/%s' % (path, file))
            if dirname == "":
                mf = audio.srmr_audio(path, file)
            else:
                mf = audio.srmr_audio("%s/%s" % (path, dirname), file)
            mf = np.einsum('ijk->kij', mf)
            mf = get_no_examples(mf,7501)
            stats = MFStats(mf)

            r = np.arange(0, 300.04, 0.04)
            r = np.reshape(r, (len(r),1))

            mfeat = np.reshape(mf, (mf.shape[0], mf.shape[1] * mf.shape[2]))
            mfeat = np.concatenate((r, mfeat), axis=1)
            write_arff(featpath, "%s%s"%(file[:-4], ".arff"), mfeat)
            np.savetxt('%s/test.txt'%(featpath), mfeat, delimiter=',')

            mfeatstats = stats.get_stats()
            mfeatstats = mfeatstats[0:7501][:]
            mfeatstats = np.concatenate((r, mfeatstats), axis=1)
            write_arff(featstatspath, "%s%s"%(file[:-4], ".arff"), mfeatstats)

            mfeatmovstats = stats.moving_stats(mfeat, 10)
            mfeatmovstats = mfeatmovstats[0:7501][:]
            mfeatmovstats = np.concatenate((r, mfeatmovstats), axis=1)
            write_arff(featmovstatspath, "%s%s"%(file[:-4], ".arff"), mfeatmovstats)

            mfeatcombstats = stats.moving_stats(mfeatstats, 10)
            mfeatcombstats = mfeatcombstats[0:7501][:]
            mfeatcombstats = np.concatenate((r, mfeatcombstats), axis=1)
            write_arff(featcombstatspath, "%s%s"%(file[:-4], ".arff"), mfeatcombstats)

    shutil.copytree(featpath, "%s/../valence"(featpath))
    shutil.copytree(featstatspath, "%s/../valence"(featstatspath))
    shutil.copytree(featmovstatspath, "%s/../valence"(featmovstatspath))
    shutil.copytree(featcombstatspath, "%s/../valence"(featcombstatspath))

# Windows
pathfeat = 'F:/AVEC/mf_features'
pathfeatstats = 'F:/AVEC/mf_features_stats'
pathfeatmovstats = 'F:/AVEC/mf_features_movstats'
pathfeatcombstats = 'F:/AVEC/mf_features_combstats'

# Mac
#pathfeat = '../../output/mf_features'
#pathfeatstats = '../../output/mf_features_stats'
#pathfeatmovstats = '../../output/mf_features_movstats'
#pathfeatcombstats = '../../output/mf_features_combstats'

listfolders(pathfeat, pathfeatstats, pathfeatmovstats, pathfeatcombstats)



