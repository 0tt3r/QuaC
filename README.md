# QuaC

QuaC (Quantum in C) is a parallel time dependent open quantum systems solver, written by Matthew Otten (otten@anl.gov). QuaC utilizes PETSc (www.mcs.anl.gov/petsc) for parallelization and linear algebra and features are still regularly being added.

QuaC strives to make the process of translating the physics equations into parallel code as simple as possible. QuaC supports both features for both general open quantum systems (such as Jaynes-Cummings models or simulations of physical systems such as neutral atoms, nitrogen vacancies, spin-boson models, etc) as well as specific quantum information features for qubits, such as gates and circuits. 

## A Tutorial

### Building

Before building QuaC, you'll need [PETSc](http://www.mcs.anl.gov/petsc) and [SLEPc](http://slepc.upv.es/) configured to use complex scalar types (note that real scalar types are the default) and, if you plan to run exceptionally large systems, you will want to use 64 bit integers. Once those packages are installed, and the environmental variables `PETSC_ARCH`, `PETSC_DIR`, and `SLEPC_DIR` are set (`PETSC_ARCH` should be set to something like linux-gnu-c-complex-int64-sprng), you'll be able to build QuaC using make.

For example, to install on a standard Linux system, something along these lines should work:

```
git clone https://gitlab.com/petsc/petsc.git petsc
cd petsc
git checkout v3.13.3

export PETSC_DIR=${PWD}
export PETSC_ARCH=linux-gnu-c-complex-int64-sprng

./configure --with-scalar-type=complex --download-mpich --download-fblaslapack=1 \
  --with-debugging=no COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3 --with-64-bit-indices --download-sprng
make PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} all

cd ..
git clone https://gitlab.com/slepc/slepc
cd slepc

export SLEPC_DIR=${PWD}
./configure
make SLEPC_DIR=${SLEPC_DIR}

cd ..
git clone -b diagonalize https://github.com/0tt3r/QuaC
cd QuaC
make test
```
The output of ```make test``` should be a lot of compilation warnings. At the end, it should print all of the errors - if nothing is printed below 'All failures listed below', the tests passed! The full test results can be viewed in the file test_results
