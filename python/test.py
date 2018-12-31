import quac

quac.initialize()
q = quac.Instance()
print("node_id: {0:d}".format(q.node_id))
print("num_nodes: {0:d}".format(q.num_nodes))

quac.finalize()

