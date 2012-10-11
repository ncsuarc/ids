#include <Python.h>
#include <structmember.h>
#include <ueye.h>

#include "ids.h"
#include "intobject.h"

static PyObject *ids_Camera_getwidth(ids_Camera *self, void *closure);
static int ids_Camera_setwidth(ids_Camera *self, PyObject *value, void *closure);

static PyObject *ids_Camera_getheight(ids_Camera *self, void *closure);
static int ids_Camera_setheight(ids_Camera *self, PyObject *value, void *closure);

static PyObject *ids_Camera_getpixelclock(ids_Camera *self, void *closure);
static int ids_Camera_setpixelclock(ids_Camera *self, PyObject *value, void *closure);

static PyObject *ids_Camera_getcolor_mode(ids_Camera *self, void *closure);
static int ids_Camera_setcolor_mode(ids_Camera *self, PyObject *value, void *closure);

static PyObject *ids_Camera_getgain(ids_Camera *self, void *closure);
static int ids_Camera_setgain(ids_Camera *self, PyObject *value, void *closure);

static PyObject *ids_Camera_getexposure(ids_Camera *self, void *closure);
static int ids_Camera_setexposure(ids_Camera *self, PyObject *value, void *closure);

static PyObject *ids_Camera_getauto_exposure(ids_Camera *self, void *closure);
static int ids_Camera_setauto_exposure(ids_Camera *self, PyObject *value, void *closure);

static PyObject *ids_Camera_getauto_speed(ids_Camera *self, void *closure);
static int ids_Camera_setauto_speed(ids_Camera *self, PyObject *value, void *closure);

static PyObject *ids_Camera_getauto_white_balance(ids_Camera *self, void *closure);
static int ids_Camera_setauto_white_balance(ids_Camera *self, PyObject *value, void *closure);

static PyObject *ids_Camera_getcolor_correction(ids_Camera *self, void *closure);
static int ids_Camera_setcolor_correction(ids_Camera *self, PyObject *value, void *closure);

PyGetSetDef ids_Camera_getseters[] = {
    {"width", (getter) ids_Camera_getwidth, (setter) ids_Camera_setwidth, "Image width", NULL},
    {"height", (getter) ids_Camera_getheight, (setter) ids_Camera_setheight, "Image height", NULL},
    {"pixelclock", (getter) ids_Camera_getpixelclock, (setter) ids_Camera_setpixelclock, "Pixel Clock of camera", NULL},
    {"color_mode", (getter) ids_Camera_getcolor_mode, (setter) ids_Camera_setcolor_mode, "Color mode of images", NULL},
    {"gain", (getter) ids_Camera_getgain, (setter) ids_Camera_setgain, "Hardware gain (individual RGB gains not yet supported)", NULL},
    {"exposure", (getter) ids_Camera_getexposure, (setter) ids_Camera_setexposure, "Exposure time", NULL},
    {"auto_exposure", (getter) ids_Camera_getauto_exposure, (setter) ids_Camera_setauto_exposure, "Auto exposure", NULL},
    {"auto_speed", (getter) ids_Camera_getauto_speed, (setter) ids_Camera_setauto_speed, "Auto speed", NULL},
    {"auto_white_balance", (getter) ids_Camera_getauto_white_balance, (setter) ids_Camera_setauto_white_balance, "Auto White Balance", NULL},
    {"color_correction", (getter) ids_Camera_getcolor_correction, (setter) ids_Camera_setcolor_correction, "IR color correction factor", NULL},
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

    int clock;

    if (PyInt_Check(value)) {
        clock = (int) PyInt_AsLong(value);
    }
    else if (PyLong_Check(value)) {
        clock = (int) PyLong_AsLong(value);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Pixel clock must be an int or long.");
        return -1;
    }

    if (clock < 0) {
        PyErr_SetString(PyExc_ValueError, "Pixel clock must be positive.");
        return -1;
    }

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

static PyObject *ids_Camera_getcolor_mode(ids_Camera *self, void *closure) {
    int color = is_SetColorMode(self->handle, IS_GET_COLOR_MODE);

    return PyInt_FromLong(color);
}

static int ids_Camera_setcolor_mode(ids_Camera *self, PyObject *value, void *closure) {
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

    if (!set_color_mode(self, color)) {
        return -1;
    }

    return 0;
}

static PyObject *ids_Camera_getgain(ids_Camera *self, void *closure) {
    int gain = is_SetHardwareGain(self->handle, IS_GET_MASTER_GAIN, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER);

    return PyInt_FromLong(gain);
}

static int ids_Camera_setgain(ids_Camera *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'gain'");
        return -1;
    }

    if (!PyInt_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Gain must be an int.");
        return -1;
    }

    int gain = (int) PyInt_AsLong(value);
    Py_DECREF(value);

    int ret = is_SetHardwareGain(self->handle, gain, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER);
    switch (ret) {
    case IS_SUCCESS:
        return 0;
    case IS_INVALID_PARAMETER:
        PyErr_SetString(PyExc_ValueError, "Gain out of range.");
        return -1;
    default:
        PyErr_SetString(PyExc_IOError, "Unable to set gain.");
        return -1;
    }

    return -1;
}

static PyObject *ids_Camera_getexposure(ids_Camera *self, void *closure) {
    double exposure;
    int ret;
    ret = is_Exposure(self->handle, IS_EXPOSURE_CMD_GET_EXPOSURE, &exposure, sizeof(exposure));
    switch (ret) {
    case IS_SUCCESS:
        return PyFloat_FromDouble(exposure);
        break;
    default:
        PyErr_Format(PyExc_IOError, "Failed to retrieve exposure time from camera. Returned: %d", ret);
    }

    return NULL;
}

static int ids_Camera_setexposure(ids_Camera *self, PyObject *value, void *closure) {
    double exposure;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'exposure' (that would be silly)");
        return -1;
    }

    PyObject *flt = PyNumber_Float(value);
    if (flt == NULL) {
        PyErr_SetString(PyExc_TypeError, "Could not convert your crappy arg to a float.");
        Py_DECREF(value);
        return -1;
    }

    exposure = PyFloat_AsDouble(flt); 

    Py_DECREF(flt);

    int ret;
    ret = is_Exposure(self->handle, IS_EXPOSURE_CMD_SET_EXPOSURE, (void*) &exposure, sizeof(exposure));
    switch (ret) {
    case IS_SUCCESS:
        return 0;
        break;
    case IS_INVALID_PARAMETER:
        PyErr_SetString(PyExc_ValueError, "Exposure out of range");
        break;
    default:
        PyErr_SetString(PyExc_IOError, "Failed to set exposure time");
    }

    return -1;
}

static PyObject *ids_Camera_getauto_exposure(ids_Camera *self, void *closure) {
    double val;
    int ret;
    
    ret = is_SetAutoParameter(self->handle, IS_GET_ENABLE_AUTO_SHUTTER, &val, NULL);
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        PyErr_SetString(PyExc_IOError, "Failed to get auto exposure setting.");
        return NULL;
    }
        
    if (val) {
        Py_INCREF(Py_True);
        return Py_True;
    }
    else {
        Py_INCREF(Py_False);
        return Py_False;
    }
}

static int ids_Camera_setauto_exposure(ids_Camera *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'auto_exposure'");
        return -1;
    }

    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Auto exposure must be a bool.");
        return -1;
    }

    double val = (value == Py_True) ? 1 : 0;
    Py_DECREF(value);

    int ret = is_SetAutoParameter(self->handle, IS_SET_ENABLE_AUTO_SHUTTER, &val, NULL);
    switch (ret) {
    case IS_SUCCESS:
        if (val > 0) {
            self->autofeatures++;
        }
        else {
            self->autofeatures--;
        }
        return 0;
    default:
        PyErr_SetString(PyExc_IOError, "Unable to set auto exposure.");
        return -1;
    }

    return -1;
}

static PyObject *ids_Camera_getauto_speed(ids_Camera *self, void *closure) {
    double val;
    int ret;
    
    ret = is_SetAutoParameter(self->handle, IS_GET_AUTO_SPEED, &val, NULL);
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        PyErr_SetString(PyExc_IOError, "Failed to get auto speed setting.");
        return NULL;
    }
        
    return PyFloat_FromDouble(val);
}

static int ids_Camera_setauto_speed(ids_Camera *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'auto_exposure'");
        return -1;
    }

    if (!PyFloat_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Auto speed must be a float.");
        return -1;
    }

    double val = PyFloat_AsDouble(value);
    Py_DECREF(value);

    int ret = is_SetAutoParameter(self->handle, IS_SET_AUTO_SPEED, &val, NULL);
    switch (ret) {
    case IS_SUCCESS:
        return 0;
    case IS_INVALID_PARAMETER:
        PyErr_SetString(PyExc_ValueError, "Auto speed out of range");
        break;
    default:
        PyErr_SetString(PyExc_IOError, "Unable to set auto speed.");
        return -1;
    }

    return -1;
}

static PyObject *ids_Camera_getauto_white_balance(ids_Camera *self, void *closure) {
    double val;
    UINT val2;
    int ret;
    
    ret = is_SetAutoParameter(self->handle, IS_GET_ENABLE_AUTO_WHITEBALANCE, &val, NULL);
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        PyErr_SetString(PyExc_IOError, "Failed to get auto white balance setting.");
        return NULL;
    }

    ret = is_AutoParameter(self->handle, IS_AWB_CMD_GET_ENABLE, &val2, sizeof(val2));
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        PyErr_SetString(PyExc_IOError, "Failed to get auto white balance setting.");
        return NULL;
    }
        
    if (val && val2) {
        Py_INCREF(Py_True);
        return Py_True;
    }
    else if (val || val2) {
        PyErr_SetString(PyExc_StandardError, "Only one white balance setting is enabled.  It may or may not work.");
        return NULL;
    }
    else {
        Py_INCREF(Py_False);
        return Py_False;
    }
}

static int ids_Camera_setauto_white_balance(ids_Camera *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'auto_white_balance'");
        return -1;
    }

    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Auto white balance must be a bool.");
        return -1;
    }

    double val = (value == Py_True) ? 1 : 0;
    Py_DECREF(value);

    int ret = is_SetAutoParameter(self->handle, IS_SET_ENABLE_AUTO_WHITEBALANCE, &val, NULL);
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        PyErr_SetString(PyExc_IOError, "Unable to set auto white balance.");
        return -1;
    }

    UINT val2 = val/1;

    ret = is_AutoParameter(self->handle, IS_AWB_CMD_SET_ENABLE, &val2, sizeof(val2));
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        val = 0;
        is_SetAutoParameter(self->handle, IS_SET_ENABLE_AUTO_WHITEBALANCE, &val, NULL);
        PyErr_SetString(PyExc_IOError, "Unable to set auto white balance.");
        return -1;
    }

    if (val2) {
        UINT nType = IS_AWB_COLOR_TEMPERATURE;
        ret = is_AutoParameter(self->handle, IS_AWB_CMD_SET_TYPE, (void*)&nType, sizeof(nType));
        switch (ret) {
        case IS_SUCCESS:
            break;
        default:
            val = 0;
            is_SetAutoParameter(self->handle, IS_SET_ENABLE_AUTO_WHITEBALANCE, &val, NULL);
            val2 = 0;
            is_AutoParameter(self->handle, IS_AWB_CMD_SET_ENABLE, &val2, sizeof(val2));
            PyErr_SetString(PyExc_IOError, "Unable to set auto white balance.");
            return -1;
        }
    }

    if (val > 0) {
        self->autofeatures++;
    }
    else {
        self->autofeatures--;
    }

    return 0;
}

static PyObject *ids_Camera_getcolor_correction(ids_Camera *self, void *closure) {
    double factor;
    int ret;
    
    ret = is_SetColorCorrection(self->handle, IS_GET_CCOR_MODE, &factor);
    switch (ret) {
    case IS_CCOR_ENABLE_NORMAL:
    case IS_CCOR_ENABLE_BG40_ENHANCED:
    case IS_CCOR_ENABLE_HQ_ENHANCED:
        Py_INCREF(Py_True);
        return Py_True;
    case IS_CCOR_DISABLE:
        Py_INCREF(Py_False);
        return Py_False;
    default:
        PyErr_SetString(PyExc_IOError, "Failed to get color correction setting.");
        return NULL;
    }
}

static int ids_Camera_setcolor_correction(ids_Camera *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'color_correction'");
        return -1;
    }

    /* Disable color correction */
    if (value == Py_False) {
        double factor;
        Py_DECREF(value);

        int ret = is_SetColorCorrection(self->handle, IS_CCOR_DISABLE, &factor);
        switch (ret) {
        case IS_SUCCESS:
            return 0;
        default:
            PyErr_SetString(PyExc_IOError, "Unable to disable color correction.");
            return -1;
        }
    }

    PyObject *num = PyNumber_Float(value);
    Py_DECREF(value);
    if (!num) {
        PyErr_SetString(PyExc_TypeError, "Color correction factor must be a float(ish).");
        return -1;
    }

    double factor = PyFloat_AsDouble(num);
    Py_DECREF(num);

    int ret = is_SetColorCorrection(self->handle, IS_CCOR_SET_IR_AUTOMATIC, &factor);
    switch (ret) {
    case IS_SUCCESS:
        return 0;
    case IS_INVALID_PARAMETER:
        PyErr_SetString(PyExc_ValueError, "Color correction factor out of range (0 to 1)");
        break;
    default:
        PyErr_SetString(PyExc_IOError, "Unable to set color correction factor.");
        return -1;
    }

    return -1;
}
