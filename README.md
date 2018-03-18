# Feature pooling on modulation spectrum features

This scripts applies feature pooling on modulation spectrum features. 

Before using this scripts, intall the python implementation of SRMR which can be found at https://github.com/jfsantos/SRMRpy.

# Usage

After installing the SRMR toolbox, a quick test can be done by entering the script folder and executing the following command:

python modspec_avec.py ./wav ./features 0 3 20


Where

param1: path containing list of folders with wav files

param2: path where the features will be generated

param3: 78 (zero padding in ms to align windows with target framstep)

param4: file type (1 - .arff, 2 - .h5, 3 - .csv)

param5: pooling size (Number of frames)
