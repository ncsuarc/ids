#include <Python.h>
#include "structmember.h"
#include <ueye.h>

#include "ids.h"

static PyObject *ids_Camera_close(ids_Camera *self, PyObject *args, PyObject *kwds);

PyMethodDef ids_Camera_methods[] = {
    {"close", (PyCFunction) ids_Camera_close, METH_VARARGS, "Closes open camera"},
    {NULL}
};

static PyObject *ids_Camera_close(ids_Camera *self, PyObject *args, PyObject *kwds) {
    if (is_ExitCamera(self->handle) != IS_SUCCESS) {
        Py_INCREF(Py_False);
        return Py_False;
    }

    Py_INCREF(Py_True);
    return Py_True;
}
