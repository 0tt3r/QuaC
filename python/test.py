import quac
from math import exp, pow, cos

quac.initialize()
q = quac.Instance()
print(q)
print("node_id: {0:d}".format(q.node_id))
print("num_nodes: {0:d}".format(q.num_nodes))

us = 1e-6
ns = 1e-9

# Some reasonable values for superconducting qubits...
t1 = 25 * us
t2 = 20 * us

t1q = 20 * ns
t2q = 40 * ns # An optimistic value from https://arxiv.org/pdf/1903.02492.pdf

# TODO: NV center numbers (e.g., from https://arxiv.org/pdf/1905.02094.pdf):
# Measurement time: 10 ms
# Gate duration: 500 to 2000 us (for both 1- and 2-qubit gates); ^14N is 300 us.
# T2 times are 4-18 ms; ^14N is 25 ms.
# T1 = 4e3 seconds.

c = quac.Circuit()
print(c)
c.initialize(25)
print(c)
c.add_gate(gate="x", qubit1=0)
c.add_gate(gate="rz", qubit1=0, angle=0.447, time=t1q)
c.add_gate(gate="y", qubit1=1, time=2*t1q)
c.add_gate(gate="cnot", qubit1=0, qubit2=1, time=3*t1q)
print(c)

q.num_qubits = 2
q.create_qubits()
print(q)

gamma_1 = 1.0/t1
gamma_2 = 1.0/t2 - 1.0/(2.0*t1)

for i in range(0, 2):
  q.add_lindblad_emission(i, gamma_1)
  q.add_lindblad_dephasing(i, gamma_2=gamma_2)
  q.add_lindblad_thermal_coupling(i, 1e-5)
  for j in range(0, 2):
    if i != j:
      q.add_lindblad_cross_coupling(i, j, 1e-6)

q.create_density_matrix()
q.start_circuit_at(c)

def pulse(t, t0):
  print("computing pulse at: {0:f}, t0 is {1:f}".format(t, t0))
  return -exp(-0.5*pow(t - t0, 2))*cos(0.1*t)

q.add_ham_num_time_dep(0, lambda t: pulse(t, 1));
q.add_ham_cross_coupling_time_dep(0, 1, lambda t: pulse(t, 0.8));

def mon(q1, s, t):
  print("monitor: {0}: step {1:d}, time {2:f}".format(q1, s, t))
  q1.print_density_matrix(filename="dm-test-s{0}".format(s))

q.ts_monitor = mon

q.run(4*us, dt=ns)
q.print_density_matrix()
print(q.get_bitstring_probs())

