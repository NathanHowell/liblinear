% This make.m is used under Windows

mex -O -c ..\blas\*.c -outdir ..\blas
mex -O -c ..\lr.cpp
mex -O -c lr_model_matlab.c -I..\
mex -O lrtrain.c -I..\ lr.obj lr_model_matlab.obj ..\blas\*.obj
mex -O lrpredict.c -I..\ lr.obj lr_model_matlab.obj ..\blas\*.obj
mex -O read_sparse.c

