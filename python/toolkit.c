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

  free(argv);
  Py_RETURN_NONE;
}

static PyObject *
quac_finalize(PyObject *self, PyObject *args) {
  QuaC_finalize();
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

  PetscInt num_qubits;
  operator *qubits;
  Vec rho;

  PyObject *ts_monitor_callback;
} QuaCInstance;

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
  QuaCInstance *self;
  self = (QuaCInstance *) type->tp_alloc(type, 0);
  if (self == NULL)
    return (PyObject *) self;

  self->num_qubits = 0;
  self->qubits = NULL;
  self->rho = NULL;

  self->ts_monitor_callback = Py_None;
  Py_INCREF(self->ts_monitor_callback);

  return (PyObject *) self;
}

static int
QuaCInstance_init(QuaCInstance *self, PyObject *args, PyObject *kwds)
{
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
    {"num_qubits", T_INT, offsetof(QuaCInstance, num_qubits), 0,
     "number of qubits"},
    {NULL}  /* Sentinel */
};

static PyMethodDef QuaCInstance_methods[] = {
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
  {"initialize",  quac_initialize, METH_VARARGS,
   "Initialize QuaC."},
  {"finalize",  quac_finalize, METH_VARARGS,
   "Finalize QuaC."},
  {"clear",  quac_clear, METH_VARARGS,
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

  m = PyModule_Create(&quacmodule);
  if (m == NULL)
    return NULL;

  Py_INCREF(&QuaCInstanceType);
  PyModule_AddObject(m, "Instance", (PyObject *) &QuaCInstanceType);
  return m;
}

