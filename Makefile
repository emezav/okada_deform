
.PHONY: all build

SOURCES=$(wildcard src/*.cpp src/*.cu)
EXECUTABLE=okada_deform

all: build

debug: $(SOURCES)
	nvcc -o $(EXECUTABLE) --std=c++17 -g -I./include  $(SOURCES)

build: $(SOURCES)
	nvcc -o $(EXECUTABLE) --std=c++17 -I./include  $(SOURCES)

clean:
	rm -rf $(EXECUTABLE) samples/*.bil samples/*.hdr samples/*.prj docs

doc:
	doxygen

run: build
	./run.sh samples/test.txt

