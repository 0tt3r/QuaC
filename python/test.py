import quac

quac.initialize()
q = quac.Instance()
print(q)
print("node_id: {0:d}".format(q.node_id))
print("num_nodes: {0:d}".format(q.num_nodes))

c = quac.Circuit()
print(c)
c.initialize(25)
print(c)
c.add_gate(gate="x", qubit1=0)
c.add_gate(gate="rz", qubit1=0, angle=0.447, time=0.5)
c.add_gate(gate="y", qubit1=1, time=2.5)
c.add_gate(gate="cnot", qubit1=0, qubit2=1, time=3)
print(c)

q.num_qubits = 2
q.create_qubits()
print(q)

for i in range(0, 2):
  q.add_lindblad_emission(i, 1e-4)
  q.add_lindblad_dephasing(i, gamma_2=1e-5)
  q.add_lindblad_thermal_coupling(i, 1e-5)
  for j in range(0, 2):
    if i != j:
      q.add_lindblad_cross_coupling(i, j, 1e-6)

q.create_density_matrix()
q.start_circuit_at(c)

def mon(c1, s, t):
  print("monitor: {0}: step {1:d}, time {2:f}".format(c1, s, t))

q.ts_monitor = mon

q.run(4, dt=0.1)

# Don't call this for now; otherwise getting the error upon exit:
#  Attempting to use an MPI routine after finalizing MPICH
# quac.finalize()

