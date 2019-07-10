# QuaC

QuaC (Quantum in C) is a parallel time dependent open quantum systems solver, written by Matthew Otten (otten@anl.gov). QuaC utilizes PETSc (www.mcs.anl.gov/petsc) for parallelization and linear algebra and features are still regularly being added.

QuaC strives to make the process of translating the physics equations into parallel code as simple as possible. QuaC supports both idealized (instantaneous) "quantum gate" operators and time-dependent operators (for modeling pulses and the like).

QuaC has both a C API and a Python interface.

Web site: [https://0tt3r.github.io/QuaC/](https://0tt3r.github.io/QuaC/)

## A Tutorial

### Building

Before building QuaC, you'll need [PETSc](http://www.mcs.anl.gov/petsc) and [SLEPc](http://slepc.upv.es/) configured to use complex scalar types (note that real scalar types are the default). Once those packages are installed, and the environmental variables `PETSC_ARCH`, `PETSC_DIR`, and `SLEPC_DIR` are set (`PETSC_ARCH` should be set to something like linux-gnu-c-complex), you'll be able to build QuaC using make.

To use the Python interface, make sure that you checkout the `python-interface` branch, and after you build QuaC itself, go into the `python` subdirectory, and run make there as well.

For example, to install on a standard Linux system, something along these lines should work:

```
git clone -b maint https://bitbucket.org/petsc/petsc petsc
cd petsc

export PETSC_DIR=${PWD}
export PETSC_ARCH=linux-gnu-c-complex 

./configure --with-scalar-type=complex --download-mpich --download-fblaslapack=1 \
  --with-debugging=no COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3 --with-64-bit-indices
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

### A Simple Circuit

Once everything is built, it's time for a simple quantum circuit (with, of course, some noise thrown in).

```python
import quac
```

The first thing that your application must do is to initialize QuaC:

```python
quac.initialize()
```

And, with that done, you can create an instance of the QuaC simulator:

```python
q = quac.Instance()
```

The QuaC instance contains a number of useful variables that you might access:

```python
print(q)
print("node_id: {0:d}".format(q.node_id))
print("num_nodes: {0:d}".format(q.num_nodes))
```

We can create a simple quantum circuit:

```python
c = quac.Circuit()
print(c)
```

and initialize it (we provide the maximum number of gates):

```python
c.initialize(25)
print(c)
```

and then add the quantum gates, and for each, add a time:

```python
c.add_gate(gate="x", qubit1=0)
c.add_gate(gate="rz", qubit1=0, angle=0.447, time=0.5)
c.add_gate(gate="y", qubit1=1, time=2.5)
c.add_gate(gate="cnot", qubit1=0, qubit2=1, time=3)
print(c)
```

The following gates are provided: CZX, CmZ, CZ, CXZ, CNOT, H, X, Y, Z, I (the identity), RX, RY, RZ.

Our circuit uses two qubits, and we should make sure that our QuaC instance is also setup to simulate that number of qubits:

```python
q.num_qubits = 2
q.create_qubits()
print(q)
```

Now comes the interesting point: we'll add some noise operators to our system:

```python
for i in range(0, 2):
  q.add_lindblad_emission(i, 1e-4)
  q.add_lindblad_dephasing(i, gamma_2=1e-5)
  q.add_lindblad_thermal_coupling(i, 1e-5)
  for j in range(0, 2):
    if i != j:
      q.add_lindblad_cross_coupling(i, j, 1e-6)
```

Next we create the density matrix and add our circuit to the QuaC instance:

```python
q.create_density_matrix()
q.start_circuit_at(c)
```

QuaC is a physics-based quantum simulator, and simulates the evolution of the system in time. We don't need to do anything to observe the system as it is evolving, but if we would like to do so, we can install a time-step monitor. This is a function that gets called by time time stepper as the system is evolving in time (the frequency with which the monitor is called is dynamically determined):

```python
def mon(q1, s, t):
  print("monitor: {0}: step {1:d}, time {2:f}".format(q1, s, t))
  q1.print_density_matrix(filename="dm-test-s{0}".format(s))

q.ts_monitor = mon
```

Now we can actually evolve the system:

```python
q.run(4, dt=0.1)
```

And, finally, we can observe the final density matrix:

```python
q.print_density_matrix()
```


