# This Makefile is used under Linux

MATLABDIR ?= /usr/local/matlab
CXX ?= g++
#CXX ?= g++-3.3
CFLAGS = -Wall -O3 -fPIC -I$(MATLABDIR)/extern/include -I..

MEX = $(MATLABDIR)/bin/mex
MEX_OPTION = CC\=$(CXX) CXX\#$(CXX) CFLAGS\#"$(CFLAGS)" CXXFLAGS\#"$(CFLAGS)"
# uncomment the following line if you use MATLAB 7.3 (or above) on 64-bit computer
MEX_OPTION += -largeArrayDims
MEX_EXT = $(shell $(MATLABDIR)/bin/mexext)

all: lrtrain.$(MEX_EXT) lrpredict.$(MEX_EXT) read_sparse.$(MEX_EXT)

lrtrain.$(MEX_EXT): lrtrain.c ../lr.h lr.o lr_model_matlab.o
	$(MEX) $(MEX_OPTION) lrtrain.c lr.o lr_model_matlab.o -lm

lrpredict.$(MEX_EXT): lrpredict.c ../lr.h lr.o lr_model_matlab.o
	$(MEX) $(MEX_OPTION) lrpredict.c lr.o lr_model_matlab.o -lm

read_sparse.$(MEX_EXT): read_sparse.c
	$(MEX) $(MEX_OPTION) read_sparse.c

lr_model_matlab.o: lr_model_matlab.c ../lr.h
	$(CXX) $(CFLAGS) -c lr_model_matlab.c

lr.o: ../lr.cpp ../lr.h
	$(CC) $(CFLAGS) -fPIC -c ../lr.cpp

clean:
	rm -f *~ *.o; rm -f *.mex*; rm -f *.obj

