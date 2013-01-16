#include <Python.h>
#include <ueye.h>

#include "ids.h"

int color_to_bitdepth(int color) {
    switch (color) {
    case IS_CM_SENSOR_RAW8:
    case IS_CM_MONO8:
        return 8;
    case IS_CM_SENSOR_RAW12:
    case IS_CM_SENSOR_RAW16:
    case IS_CM_MONO12:
    case IS_CM_MONO16:
        return 16;
    case IS_CM_BGR8_PACKED:
    case IS_CM_RGB8_PACKED:
        return 24;
    default:
        return 32;
    }
}

PyObject *set_color_mode(ids_Camera *self, int color) {
    int ret = is_SetColorMode(self->handle, color);
    switch (ret) {
    case IS_SUCCESS:
        break;
    case IS_INVALID_COLOR_FORMAT:
        PyErr_SetString(PyExc_ValueError, "Unsupported color format.");
        return NULL;
    default:
        PyErr_SetString(PyExc_IOError, "Unable to set color mode.");
        return NULL;
    }

    /* Set object bit depth */
    self->bitdepth = color_to_bitdepth(color);

    Py_INCREF(Py_True);
    return Py_True;
}
