# QuaC

QuaC (Quantum in C) is a parallel time dependent open quantum systems solver, written by Matthew Otten (otten@anl.gov). QuaC utilizes PETSc (www.mcs.anl.gov/petsc) for parallelization and linear algebra and features are still regularly being added.

QuaC strives to make the process of translating the physics equations into parallel code as simple as possible. QuaC supports both idealized (instantaneous) "quantum gate" operators and time-dependent operators (for modeling pulses and the like).

QuaC has both a C API and a Python interface.

## A Tutorial

### Building

Before building QuaC, you'll need PETSc (http://www.mcs.anl.gov/petsc) and SLEPc (http://slepc.upv.es/) configured to use complex scalar types (note that real scalar types are the default). Once those packages are installed, and the environmental variables `PETSC_ARCH`, `PETSC_DIR`, and `SLEPC_DIR` are set (`PETSC_ARCH` should be set to something like linux-gnu-c-complex), you'll be able to build QuaC using make.

To use the Python interface, make sure that you checkout the `python-interface` branch, and after you build QuaC itself, go into the `python` subdirectory, and run make there as well.

For example, to install on a standard Linux system, something along these lines should work:

```
git clone -b maint https://bitbucket.org/petsc/petsc petsc
cd petsc

export PETSC_DIR=${PWD}
export PETSC_ARCH=linux-gnu-c-complex 

./configure --with-scalar-type=complex --download-mpich --download-fblaslapack=1 --with-debugging=no COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3 --with-64-bit-indices
make PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} all

cd ..
git clone -b maint https://bitbucket.org/slepc/slepc
cd slepc

export SLEPC_DIR=${PWD}
./configure
make SLEPC_DIR=${SLEPC_DIR}

cd ..
git clone -b python-interface https://github.com/0tt3r/QuaC
cd QuaC
make
cd python
make
```

Remember to add `-j<number of cores>` to your make commands to build in parallel.



