CXX ?= g++
CFLAGS = -static-libgcc -static-libstdc++ -O3 -fPIC -w -Wfatal-errors
SHVER = 2
OS = $(shell uname)

all: svm-train svm-predict HEMKit 

lib: svm.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libsvm.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,libsvm.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} svm.o -o libsvm.so.$(SHVER)

svm-predict: svm-predict.cpp svm.o hrsvm_common.o 
	$(CXX) $(CFLAGS) -o svm-predict svm-predict.cpp svm.o hrsvm_common.o -lm

svm-train: svm-train.cpp svm.o hrsvm_common.o
	$(CXX) $(CFLAGS) -o svm-train svm-train.cpp svm.o hrsvm_common.o -lm

svm-scale: svm-scale.c
	$(CXX) $(CFLAGS) svm-scale.c -o svm-scale
svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c svm.cpp
hrsvm_common.o: hrsvm_common.cpp svm.h svm.o
	$(CXX) $(CFLAGS) -c hrsvm_common.cpp

HEMKit: hemkit.o bestPathfinder.o graph.o subGraphCreator.o
	g++ hemkit.cpp bestPathfinder.cpp graph.cpp subGraphCreator.cpp -o HEMKit

hemkit.o: hemkit.cpp
	g++ -c hemkit.cpp 

bestPathfinder.o: bestPathfinder.cpp
	g++ -c bestPathfinder.cpp

graph.o: graph.cpp
	g++ -c graph.cpp

subGraphCreator.o: subGraphCreator.cpp
	g++ -c subGraphCreator.cpp

clean:
	rm -f *~ *.o svm-train svm-predict svm-scale libsvm.so.$(SHVER)
	rm -f *~ *.o svm-train.exe svm-predict svm-scale.exe libsvm.so.$(SHVER)
	rm -f *~ *.o HEMKit
	rm -f *~ *.o HEMKit.exe