#include <Python.h>
#include <structmember.h>
#include <ueye.h>

#include "ids.h"

static PyObject *ids_Camera_getpixelclock(ids_Camera *self, void *closure);
static int ids_Camera_setpixelclock(ids_Camera *self, PyObject *value, void *closure);

PyGetSetDef ids_Camera_getseters[] = {
    {"pixelclock", (getter) ids_Camera_getpixelclock, (setter) ids_Camera_setpixelclock, "Pixel Clock of camera", NULL},
    {NULL}
};

static PyObject *ids_Camera_getpixelclock(ids_Camera *self, void *closure) {
    UINT clock;
    int ret;

    ret = is_PixelClock(self->handle, IS_PIXELCLOCK_CMD_GET, &clock, sizeof(clock));
    switch (ret) {
    case IS_SUCCESS:
        return PyInt_FromLong(clock);
        break;
    default:
        PyErr_SetString(PyExc_IOError, "Failed to retreive pixel clock from camera");
    }

    return NULL;
}

static int ids_Camera_setpixelclock(ids_Camera *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'pixelclock'");
        return -1;
    }

    if (!PyInt_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Attribute must be an int.");
        return -1;
    }

    int clock = (int) PyInt_AsLong(value);
    if (clock < 0) {
        PyErr_SetString(PyExc_ValueError, "Attribute must be positive.");
        return -1;
    }

    Py_DECREF(value);

    int ret;
    ret = is_PixelClock(self->handle, IS_PIXELCLOCK_CMD_SET, (void*) &clock, sizeof(clock));
    switch (ret) {
    case IS_SUCCESS:
        return 0;
        break;
    case IS_INVALID_PARAMETER:
        PyErr_SetString(PyExc_ValueError, "Pixel clock value out of range");
        break;
    default:
        PyErr_SetString(PyExc_IOError, "Failed to set pixel clock.");
    }

    return -1;
}
