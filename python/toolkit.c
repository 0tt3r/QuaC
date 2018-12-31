#include <Python.h>

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

static PyMethodDef QuaCMethods[] = {
  {"initialize",  quac_initialize, METH_VARARGS,
   "Initialize QuaC."},
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
  return PyModule_Create(&quacmodule);
}

