#include <Python.h>
#include <structmember.h>

#include <math.h>
#include <stdlib.h>

#include <petsc.h>

#include <quac.h>
#include <operators.h>
#include <error_correction.h>
#include <solver.h>
#include <dm_utilities.h>
#include <quantum_gates.h>
#include <qasm_parser.h>

static int quac_initialized = 0;

static PyObject *
quac_initialize(PyObject *self, PyObject *args) {
  int argc;
  char **argv;
  PyObject* argl;

  argl = PySys_GetObject((char *)("argv"));
  argc = 1;
  if (argl && PyList_Size(argl) > 0)
    argc = (int) PyList_GET_SIZE(argl);

  argv = (char **) malloc(sizeof(char *)*argc);
  argv[0] = (char *) "quac";
  for (int i = 0; i < argc; ++i)
    argv[i] =
#if PY_MAJOR_VERSION < 3
      PyString_AS_STRING(PyList_GET_ITEM(argl, i));
#else
      (char *) PyUnicode_AS_DATA(PyList_GET_ITEM(argl, i));
#endif

  QuaC_initialize(argc, argv);
  quac_initialized = 1;

  free(argv);
  Py_RETURN_NONE;
}

static PyObject *
quac_finalize(PyObject *self, PyObject *args) {
  QuaC_finalize();
  quac_initialized = 0;
  Py_RETURN_NONE;
}

static PyObject *
quac_clear(PyObject *self, PyObject *args) {
  QuaC_clear();
  Py_RETURN_NONE;
}

static PetscErrorCode ts_monitor(TS, PetscInt, PetscReal, Vec, void*);

typedef struct {
  PyObject_HEAD

  int nid, np;

  PetscInt num_qubits;
  int num_levels;
  operator *qubits;
  Vec rho;

  PyObject *ts_monitor_callback;
} QuaCInstance;

typedef struct {
  PyObject_HEAD
  circuit c;
} QuaCCircuit;

static void
QuaCCircuit_dealloc(QuaCCircuit *self) {
  Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
QuaCCircuit_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  QuaCCircuit *self;
  self = (QuaCCircuit *) type->tp_alloc(type, 0);
  if (self == NULL)
    return (PyObject *) self;

  memset(&self->c, 0, sizeof(circuit));

  return (PyObject *) self;
}

static int
QuaCCircuit_init(QuaCCircuit *self, PyObject *args, PyObject *kwds)
{
  static char *kwlist[] = {"start_time", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|d", kwlist,
                                   &self->c.start_time))
    return -1;

  return 0;
}

static PyMemberDef QuaCCircuit_members[] = {
    {"num_gates", T_LONG, offsetof(QuaCCircuit, c.num_gates), READONLY,
     "number of gates"},
    {"start_time", T_DOUBLE, offsetof(QuaCCircuit, c.start_time), 0,
     "start time"},
    {NULL}  /* Sentinel */
};

static PyObject *
QuaCCircuit_repr(QuaCCircuit * self) {
  PyObject *r;
  char *st = PyOS_double_to_string(self->c.start_time, 'g', 0, 0, NULL);
  r = PyUnicode_FromFormat("<QuaC Curcuit{%ld gates starting at t=%s}>",
                           self->c.num_gates, st);
  PyMem_Free(st);
  return r;
}

static PyObject *
QuaCCircuit_read_qasm(QuaCCircuit *self, PyObject *args, PyObject *kwds) {
  char *filename, *format;
  PetscInt num_qubits;

  static char *kwlist[] = {"format", "filename", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "ss", kwlist,
                                   &format, &filename))
    return NULL;

  if (!strcasecmp(format, "quil")) {
    quil_read(filename, &num_qubits, &self->c);
  } else if (!strcasecmp(format, "projectq")) {
    projectq_qasm_read(filename, &num_qubits, &self->c);
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown qasm format!");
    return NULL;
  }

  return PyLong_FromLong(num_qubits);
}

static PyObject *
QuaCCircuit_init2(QuaCCircuit *self, PyObject *args, PyObject *kwds) {
  PetscInt num_gates = 0;

  static char *kwlist[] = {"num_gates", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|l", kwlist,
                                   &num_gates))
    return NULL;

  create_circuit(&self->c, num_gates);

  Py_RETURN_NONE;
}

static PyObject *
QuaCCircuit_add_gate(QuaCCircuit *self, PyObject *args, PyObject *kwds) {
  int qubit1 = -1, qubit2 = -1;
  PetscReal angle = 0, time = 0;
  gate_type gate;
  char *gate_name;

  static char *kwlist[] = {"gate", "qubit1", "qubit2", "angle", "time", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "si|idd", kwlist,
                                   &gate_name, &qubit1, &qubit2, &angle, &time))
    return NULL;

  if (!strcasecmp(gate_name, "CZX")) {
    gate = CZX;
  } else if (!strcasecmp(gate_name, "CmZ")) {
    gate = CmZ;
  } else if (!strcasecmp(gate_name, "CZ")) {
    gate = CZ;
  } else if (!strcasecmp(gate_name, "CXZ")) {
    gate = CXZ;
  } else if (!strcasecmp(gate_name, "CNOT")) {
    gate = CNOT;
  } else if (!strcasecmp(gate_name, "H")) {
    gate = HADAMARD;
  } else if (!strcasecmp(gate_name, "X")) {
    gate = SIGMAX;
  } else if (!strcasecmp(gate_name, "Y")) {
    gate = SIGMAY;
  } else if (!strcasecmp(gate_name, "Z")) {
    gate = SIGMAZ;
  } else if (!strcasecmp(gate_name, "I")) {
    gate = EYE;
  } else if (!strcasecmp(gate_name, "RX")) {
    gate = RX;
  } else if (!strcasecmp(gate_name, "RY")) {
    gate = RY;
  } else if (!strcasecmp(gate_name, "RZ")) {
    gate = RZ;
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown gate type!");
    return NULL;
  }

  if (((int) gate) < 0) {
    if (qubit2 < 0) {
      PyErr_SetString(PyExc_RuntimeError, "qubit2 must be specified for a two-qubit gate!");
      return NULL;
    }

    add_gate_to_circuit(&self->c, time, gate, qubit1, qubit2);
  } else {
    add_gate_to_circuit(&self->c, time, gate, qubit1, angle);
  }

  Py_RETURN_NONE;
}

static PyMethodDef QuaCCircuit_methods[] = {
    {"initialize_and_read_qasm",
     (PyCFunction) QuaCCircuit_read_qasm, METH_VARARGS | METH_KEYWORDS,
     "Initialize and read QASM from the specified file using the specified format."
    },
    {"initialize",
     (PyCFunction) QuaCCircuit_init2, METH_VARARGS | METH_KEYWORDS,
     "Initialize the circuit object."
    },
    {"add_gate",
     (PyCFunction) QuaCCircuit_add_gate, METH_VARARGS | METH_KEYWORDS,
     "Add a gate to the circuit object."
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject QuaCCircuitType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "quac.Circuit",
  .tp_doc = "QuaC Circuit",
  .tp_basicsize = sizeof(QuaCCircuit),
  .tp_itemsize = 0,
  .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  .tp_new = QuaCCircuit_new,
  .tp_init = (initproc) QuaCCircuit_init,
  .tp_dealloc = (destructor) QuaCCircuit_dealloc,
  .tp_members = QuaCCircuit_members,
  .tp_methods = QuaCCircuit_methods,
  .tp_repr = QuaCCircuit_repr,
  .tp_str = QuaCCircuit_repr,
};


static void
QuaCInstance_dealloc(QuaCInstance *self) {
  if (self->qubits) {
    for (int i = 0; i < self->num_qubits; ++i)
      destroy_op(&self->qubits[i]);

    free(self->qubits);
  }

  if (self->rho)
    destroy_dm(self->rho);

  Py_XDECREF(self->ts_monitor_callback);
  Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
QuaCInstance_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  QuaCInstance *self = NULL;

  if (!quac_initialized) {
    PyErr_SetString(PyExc_RuntimeError, "QuaC must be initialized first!");
    return (PyObject *) self;
  }

  self = (QuaCInstance *) type->tp_alloc(type, 0);
  if (self == NULL)
    return (PyObject *) self;

  self->nid = nid;
  self->np = np;

  self->num_qubits = 0;
  self->num_levels = 0;
  self->qubits = NULL;
  self->rho = NULL;

  self->ts_monitor_callback = Py_None;
  Py_INCREF(self->ts_monitor_callback);

  return (PyObject *) self;
}

static int
QuaCInstance_init(QuaCInstance *self, PyObject *args, PyObject *kwds) {
  static char *kwlist[] = {"num_qubits", "ts_monitor", NULL};
  PyObject *tsm = NULL, *tmp;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|lO", kwlist,
                                   &self->num_qubits, &tsm))
    return -1;

  if (tsm) {
    tmp = self->ts_monitor_callback;
    Py_INCREF(tsm);
    self->ts_monitor_callback = tsm;
    Py_XDECREF(tmp);
  }

  return 0;
}

static PyMemberDef QuaCInstance_members[] = {
    {"ts_monitor", T_OBJECT_EX, offsetof(QuaCInstance, ts_monitor_callback), 0,
     "time-step-monitor callback"},
    {"num_qubits", T_LONG, offsetof(QuaCInstance, num_qubits), 0,
     "number of qubits"},
    {"node_id", T_INT, offsetof(QuaCInstance, nid), READONLY,
     "node (rank) identifier"},
    {"num_nodes", T_INT, offsetof(QuaCInstance, np), READONLY,
     "number of nodes"},
    {NULL}  /* Sentinel */
};

static PyObject *
QuaCInstance_repr(QuaCInstance * self) {
  return PyUnicode_FromFormat("<QuaC Instance{%d qubits; %d levels; node %d of %d}>",
                              self->num_qubits, self->num_levels, self->nid, self->np);
}

static int
QuaCInstance_create_qubits(QuaCInstance *self, PyObject *args, PyObject *kwds) {
  self->num_levels = 2;

  // TODO: Can we support different numbers of levels for different qudits?

  static char *kwlist[] = {"num_levels", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist,
                                   &self->num_levels))
    return NULL;

  if (self->qubits) {
    PyErr_SetString(PyExc_RuntimeError, "qbits for this QuaC instance have already been created!");
    return NULL;
  } else if (self->num_qubits) {
    self->qubits = (operator *) malloc(sizeof(operator)*self->num_qubits);
    for (int i = 0; i < self->num_qubits; ++i)
      create_op(self->num_levels, &self->qubits[i]);
  }

  Py_RETURN_NONE;
}

static int
QuaCInstance_add_lindblad_emission(QuaCInstance *self, PyObject *args, PyObject *kwds) {
  int qubit;
  double gamma_1;

  static char *kwlist[] = {"qubit", "gamma_1", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "id", kwlist,
                                   &qubit, &gamma_1))
    return NULL;

  add_lin(gamma_1, self->qubits[qubit]);

  Py_RETURN_NONE;
}

static int
QuaCInstance_add_lindblad_dephasing(QuaCInstance *self, PyObject *args, PyObject *kwds) {
  int qubit;
  double gamma_2;

  static char *kwlist[] = {"qubit", "gamma_2", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "id", kwlist,
                                   &qubit, &gamma_2))
    return NULL;

  add_lin(gamma_2, self->qubits[qubit]->n);

  Py_RETURN_NONE;
}

static int
QuaCInstance_add_lindblad_thermal_coupling(QuaCInstance *self, PyObject *args, PyObject *kwds) {
  int qubit;
  double therm_1, n_therm = 0.5;

  static char *kwlist[] = {"qubit", "therm_1", "n_therm", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "id|d", kwlist,
                                   &qubit, &therm_1, &n_therm))
    return NULL;

  add_lin(therm_1*(n_therm + 1), self->qubits[qubit]);
  add_lin(therm_1*(n_therm), self->qubits[qubit]->dag);

  Py_RETURN_NONE;
}

static int
QuaCInstance_add_lindblad_cross_coupling(QuaCInstance *self, PyObject *args, PyObject *kwds) {
  int qubit1, qubit2;
  double coup_1;

  static char *kwlist[] = {"qubit1", "qubit2", "coup_1", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iid", kwlist,
                                   &qubit1, &qubit2, &coup_1))
    return NULL;

  add_lin_mult2(coup_1, self->qubits[qubit1]->dag, self->qubits[qubit2]);
  add_lin_mult2(coup_1, self->qubits[qubit1], self->qubits[qubit2]->dag);

  Py_RETURN_NONE;
}

static int
QuaCInstance_create_dm(QuaCInstance *self, PyObject *args, PyObject *kwds) {
  if (self->rho) {
    PyErr_SetString(PyExc_RuntimeError, "The density matrix for this QuaC instance has already been created!");
    return NULL;
  }

  create_full_dm(&self->rho);
  add_value_to_dm(self->rho, 0, 0, 1.0);
  assemble_dm(self->rho);

  Py_RETURN_NONE;
}

static int
QuaCInstance_start_circuit_at(QuaCInstance *self, PyObject *args, PyObject *kwds) {
  double time = 0.0;
  PyObject *cir;

  static char *kwlist[] = {"circuit", "time", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|d", kwlist,
                                   &cir, &time))
    return NULL;

  if (!PyObject_TypeCheck(cir, &QuaCCircuitType)) {
    PyErr_SetString(PyExc_TypeError, "Circuit is not a QuaC.Circuit object!");
    return NULL;
  }

  start_circuit_at_time(&((QuaCCircuit *) cir)->c, time);

  Py_RETURN_NONE;
}

static int
QuaCInstance_print_density_matrix(QuaCInstance *self, PyObject *args, PyObject *kwds) {
  char *filename = NULL;
  int num_print_qubits = self->num_qubits;

  static char *kwlist[] = {"filename", "num_qubits", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|sd", kwlist,
                                   &filename, &num_print_qubits))
    return NULL;

  if (filename) {
    print_dm_sparse_to_file(self->rho, pow(self->num_levels, num_print_qubits), filename);
  } else {
    print_dm_sparse(self->rho, pow(self->num_levels, num_print_qubits));
  }

  Py_RETURN_NONE;
}

static int
QuaCInstance_run(QuaCInstance *self, PyObject *args, PyObject *kwds) {
  PetscReal dt = 1.0, start_time = 0.0, end_time;
  PetscInt max_steps = INT_MAX-1;

  static char *kwlist[] = {"end_time", "dt", "start_time", "max_steps", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|ddl", kwlist,
                                   &end_time, &dt, &start_time, &max_steps))
    return NULL;

  set_ts_monitor_ctx(ts_monitor, (void *) self);

  time_step(self->rho, start_time, end_time, dt, max_steps);

  Py_RETURN_NONE;
}

static PyMethodDef QuaCInstance_methods[] = {
    {"create_qubits",
     (PyCFunction) QuaCInstance_create_qubits, METH_VARARGS | METH_KEYWORDS,
     "Create the qubits."
    },
    {"add_lindblad_emission",
     (PyCFunction) QuaCInstance_add_lindblad_emission, METH_VARARGS | METH_KEYWORDS,
     "Add Lindblad spontaneous-emission term(s)."
    },
    {"add_lindblad_dephasing",
     (PyCFunction) QuaCInstance_add_lindblad_dephasing, METH_VARARGS | METH_KEYWORDS,
     "Add Lindblad dephasing term(s)."
    },
    {"add_lindblad_thermal_coupling",
     (PyCFunction) QuaCInstance_add_lindblad_thermal_coupling, METH_VARARGS | METH_KEYWORDS,
     "Add Lindblad thermal-coupling terms."
    },
    {"add_lindblad_cross_coupling",
     (PyCFunction) QuaCInstance_add_lindblad_cross_coupling, METH_VARARGS | METH_KEYWORDS,
     "Add Lindblad cross_coupling terms."
    },
    {"create_density_matrix",
     (PyCFunction) QuaCInstance_create_dm, METH_NOARGS,
     "Create the density matrix."
    },
    {"start_circuit_at",
     (PyCFunction) QuaCInstance_start_circuit_at, METH_VARARGS | METH_KEYWORDS,
     "Registers a circuit to start at the specified time."
    },
    {"run",
     (PyCFunction) QuaCInstance_run, METH_VARARGS | METH_KEYWORDS,
     "Simulate the registered circuits."
    },
    {"print_density_matrix",
     (PyCFunction) QuaCInstance_print_density_matrix, METH_VARARGS | METH_KEYWORDS,
     "Print the density matrix."
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject QuaCInstanceType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "quac.Instance",
  .tp_doc = "QuaC Instance",
  .tp_basicsize = sizeof(QuaCInstance),
  .tp_itemsize = 0,
  .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  .tp_new = QuaCInstance_new,
  .tp_init = (initproc) QuaCInstance_init,
  .tp_dealloc = (destructor) QuaCInstance_dealloc,
  .tp_members = QuaCInstance_members,
  .tp_methods = QuaCInstance_methods,
  .tp_repr = QuaCInstance_repr,
  .tp_str = QuaCInstance_repr,
};

PetscErrorCode ts_monitor(TS ts, PetscInt step, PetscReal time, Vec rho, void *ctx) {
  QuaCInstance *self = (QuaCInstance *) ctx;
  if (self->ts_monitor_callback && self->ts_monitor_callback != Py_None) {
    PyObject *arglist;
    PyObject *result;

    arglist = Py_BuildValue("(Old)", self, step, time);
    result = PyObject_CallObject(self->ts_monitor_callback, arglist);
    Py_DECREF(arglist);

    if (result)
      Py_DECREF(result);
  }

  PetscFunctionReturn(0);
}

static PyMethodDef QuaCMethods[] = {
  {"initialize",  quac_initialize, METH_NOARGS,
   "Initialize QuaC."},
  {"finalize",  quac_finalize, METH_NOARGS,
   "Finalize QuaC."},
  {"clear",  quac_clear, METH_NOARGS,
   "Clear QuaC's internal state."},
  {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef quacmodule = {
  PyModuleDef_HEAD_INIT,
  "quac",   /* name of module */
  NULL,     /* module documentation, may be NULL */
  -1,       /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
  QuaCMethods
};

PyMODINIT_FUNC
PyInit_quac(void) {
  PyObject *m;

  if (PyType_Ready(&QuaCInstanceType) < 0)
    return NULL;

  if (PyType_Ready(&QuaCCircuitType) < 0)
    return NULL;

  m = PyModule_Create(&quacmodule);
  if (m == NULL)
    return NULL;

  Py_INCREF(&QuaCInstanceType);
  PyModule_AddObject(m, "Instance", (PyObject *) &QuaCInstanceType);

  Py_INCREF(&QuaCCircuitType);
  PyModule_AddObject(m, "Circuit", (PyObject *) &QuaCCircuitType);

  return m;
}

