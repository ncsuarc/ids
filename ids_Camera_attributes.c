#include <Python.h>
#include <structmember.h>
#include <ueye.h>

#include "ids.h"

static PyObject *ids_Camera_getwidth(ids_Camera *self, void *closure);
static int ids_Camera_setwidth(ids_Camera *self, PyObject *value, void *closure);

static PyObject *ids_Camera_getheight(ids_Camera *self, void *closure);
static int ids_Camera_setheight(ids_Camera *self, PyObject *value, void *closure);

static PyObject *ids_Camera_getpixelclock(ids_Camera *self, void *closure);
static int ids_Camera_setpixelclock(ids_Camera *self, PyObject *value, void *closure);

static PyObject *ids_Camera_getcolor(ids_Camera *self, void *closure);
static int ids_Camera_setcolor(ids_Camera *self, PyObject *value, void *closure);

PyGetSetDef ids_Camera_getseters[] = {
    {"width", (getter) ids_Camera_getwidth, (setter) ids_Camera_setwidth, "Image width", NULL},
    {"height", (getter) ids_Camera_getheight, (setter) ids_Camera_setheight, "Image height", NULL},
    {"pixelclock", (getter) ids_Camera_getpixelclock, (setter) ids_Camera_setpixelclock, "Pixel Clock of camera", NULL},
    {"color", (getter) ids_Camera_getcolor, (setter) ids_Camera_setcolor, "Color mode of images", NULL},
    {NULL}
};

static PyObject *ids_Camera_getwidth(ids_Camera *self, void *closure) {
    return PyInt_FromLong(self->width);
}

static int ids_Camera_setwidth(ids_Camera *self, PyObject *value, void *closure) {
    PyErr_SetString(PyExc_NotImplementedError, "Changing image width not yet supported.");
    return -1;
}

static PyObject *ids_Camera_getheight(ids_Camera *self, void *closure) {
    return PyInt_FromLong(self->height);
}

static int ids_Camera_setheight(ids_Camera *self, PyObject *value, void *closure) {
    PyErr_SetString(PyExc_NotImplementedError, "Changing image height not yet supported.");
    return -1;
}

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
        PyErr_SetString(PyExc_TypeError, "Pixel clock must be an int.");
        return -1;
    }

    int clock = (int) PyInt_AsLong(value);
    if (clock < 0) {
        PyErr_SetString(PyExc_ValueError, "Pixel clock must be positive.");
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

static PyObject *ids_Camera_getcolor(ids_Camera *self, void *closure) {
    int color = is_SetColorMode(self->handle, IS_GET_COLOR_MODE);

    return PyInt_FromLong(color);
}

static int ids_Camera_setcolor(ids_Camera *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'color'");
        return -1;
    }

    if (!PyInt_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Color mode must be an int.");
        return -1;
    }

    int color = (int) PyInt_AsLong(value);
    Py_DECREF(value);

    if (self->bitdepth != color_to_bitdepth(color)) {
        PyErr_SetString(PyExc_NotImplementedError, "Changing color mode to different bitdepth not yet supported.");
        return -1;
    }

    if (!set_color(self, color)) {
        return -1;
    }

    return 0;
}
