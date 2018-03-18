import os, sys
from frontend import MFStats, srmr_audio
import numpy as np
import h5py


def listfolders():
    if(len(sys.argv) < 6):
        print("\nUsage:")
        print("python modspec_avec.py ./in_path ./out_path 78 type size")
        print("Where")
        print("param1: path containing list of folders with wav files")
        print("param2: path where the features will be generated")
        print("param3: 78 (zero padding in ms to align windows with target framstep) ")
        print("param4: file type (1 - .arff, 2 - .h5, 3 - .csv) ")
        print("param5: pooling size (Number of frames) ")
        sys.exit(0)

    pathin = sys.argv[1:][0]
    pathout = sys.argv[2:][0]
    pad = int(sys.argv[3:][0])
    ftype = sys.argv[4:][0]
    poolingsize = int(sys.argv[5:][0])

    for dirname in os.listdir(pathin):
        print("Processing: %s/%s" % (pathin, dirname))
        if os.path.isdir("%s/%s" % (pathin, dirname)):
            mrs_dir = "%s/%s/%s" % (pathout, dirname, "mrs")
            msf_dir = "%s/%s/%s" % (pathout, dirname, "msf")
            pooling_1_dir = "%s/%s/%s%d" % (pathout, dirname, "mrs+pooling+", poolingsize)
            pooling_2_dir = "%s/%s/%s%d" % (pathout, dirname, "msf+pooling+", poolingsize)
            print("ftype has been defined as : %s", ftype)
            convert(pathin, dirname, mrs_dir, msf_dir, pooling_1_dir, pooling_2_dir, pad, ftype, poolingsize)


def write_file(path, file, features, ftype):
    if int(ftype) == 1:
        write_arff(path, file, features)
    elif int(ftype) == 2:
        write_hdf5(path, file, features)
    else:
        write_csv(path, file, features)

def write_hdf5(path, file, features):
    if not os.path.exists(path):
        os.makedirs(path)

    f = h5py.File("%s/%s"%(path, "%s%s" % (file[:-4], ".h5")), "w")
    f.create_dataset("mod", data=features)
    f.close()

def write_arff(path, file, features):
    if not os.path.exists(path):
        os.makedirs(path)
    f = open("%s/%s"%(path, "%s%s" % (file[:-4], ".arff")), "w")
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

def write_csv(path, file, features):
    if not os.path.exists(path):
        os.makedirs(path)

    np.savetxt('%s/%s' % (path, "%s%s" % (file[:-4], ".csv")), features, delimiter=",")

def get_no_examples(mf, no_examples):
    if len(mf) < no_examples:
        tmp = np.reshape(mf[len(mf) - 1], (1, mf.shape[1]))
        tmp = np.repeat(tmp, no_examples - len(mf), axis=0)
        mf = np.vstack((mf, tmp))
    else:
        mf = mf[0:no_examples]
    return mf

def convert(path, dirname, mrs_dir, msf_dir, pooling_1_dir, pooling_2_dir, pad = 0, ftype = 3, poolingsize = 10):
    dirs = os.listdir("%s/%s"%(path, dirname))
    for file in dirs:
        if file.endswith('.wav'):
            print('%s/%s/%s' % (path, dirname, file))
            if dirname == "":
                mf = srmr_audio(path, file, pad)
            else:
                mf = srmr_audio("%s/%s" % (path, dirname), file, pad)
            mf = np.einsum('ijk->kij', mf)

            stats = MFStats(mf)

            mrs = np.reshape(mf, (mf.shape[0], mf.shape[1] * mf.shape[2]))
            nFrame = mrs.shape[0]
            write_file(mrs_dir, file, mrs, ftype)

            msf = stats.get_stats()
            write_file(msf_dir, file, msf, ftype)

            mrs_pooling = stats.moving_stats(mrs, poolingsize)
            mrs_pooling = get_no_examples(mrs_pooling,nFrame)

            mrs_pooling = np.concatenate((mrs_pooling, mrs), axis=1)
            write_file(pooling_1_dir, file, mrs_pooling, ftype)

            msf_pooling = stats.moving_stats(msf, poolingsize)
            msf_pooling = get_no_examples(msf_pooling,nFrame)

            msf_pooling = np.concatenate((msf_pooling, mrs), axis=1)
            write_file(pooling_2_dir, file, msf_pooling, ftype)

listfolders()



