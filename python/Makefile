PYTHON=python3
PYTHON_VERSION ?= $(shell $(PYTHON) -c 'import sys; print ("%d.%d" % sys.version_info[:2])')
PYTHON_OS = $(shell uname -s | tr -d '/ ' | tr '[A-Z]' '[a-z]')
PYTHON_ARCH = $(shell uname -m)
BUILD_DIR = $(shell pwd)/build/lib.$(PYTHON_OS)-$(PYTHON_ARCH)-$(PYTHON_VERSION)

all: module

module: toolkit.c setup.py
	${PYTHON} setup.py build

test:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}/${PETSC_DIR}/${PETSC_ARCH}/lib PYTHONPATH=${BUILD_DIR} ${PYTHON} ./test.py

