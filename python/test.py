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

quac.finalize()

