CXXC ?= g++
CC ?= gcc
CFLAGS ?= -Wall -O3
LIBS ?= blas/blas.a
#LIBS ?= -lblas

all: lr-train lr-predict

lr-train: lr.o lr-train.c blas/blas.a
	$(CXXC) $(CFLAGS) -o lr-train lr-train.c lr.o $(LIBS)

lr-predict: lr.o lr-predict.c blas/blas.a
	$(CXXC) $(CFLAGS) -o lr-predict lr-predict.c lr.o $(LIBS)

blas/blas.a:
	cd blas; make OPTFLAGS='$(CFLAGS)' CC='$(CC)';

clean:
	cd blas;	make clean
	rm -f *~ lr.o lr-train lr-predict

