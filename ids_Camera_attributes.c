#include <Python.h>
#include <structmember.h>
#include <ueye.h>

#include "ids.h"

static PyObject *ids_Camera_getblah(ids_Camera *self, void *closure);
static int ids_Camera_setblah(ids_Camera *self, PyObject *value, void *closure);

PyGetSetDef ids_Camera_getseters[] = {
    {"blah", (getter) ids_Camera_getblah, (setter) ids_Camera_setblah, "Dummy attribute", NULL},
    {NULL}
};

static PyObject *ids_Camera_getblah(ids_Camera *self, void *closure) {
    Py_INCREF(self->blah);
    return self->blah;
}

static int ids_Camera_setblah(ids_Camera *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'blah'");
        return -1;
    }

    Py_DECREF(self->blah);
    Py_INCREF(value);
    self->blah = value;

    return 0;
}
