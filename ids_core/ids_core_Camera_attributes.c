/*
 * Copyright (c) 2012, 2013, North Carolina State University Aerial Robotics Club
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the North Carolina State University Aerial Robotics Club
 *       nor the names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <Python.h>
#include <structmember.h>
#include <ueye.h>

#include "ids_core.h"

PyObject *ids_core_Camera_getinfo(ids_core_Camera *self, void *closure) {
    CAMINFO cam_info;
    SENSORINFO sensor_info;

    int ret = is_GetCameraInfo(self->handle, &cam_info);
    if (ret != IS_SUCCESS) {
        raise_general_error(self, ret);
        return NULL;
    }

    ret = is_GetSensorInfo(self->handle, &sensor_info);
    if (ret != IS_SUCCESS) {
        raise_general_error(self, ret);
        return NULL;
    }

    PyObject *dict = PyDict_New();

    PyObject *serial_num = PyBytes_FromString(cam_info.SerNo);
    PyObject *manufacturer = PyBytes_FromString(cam_info.ID);
    PyObject *hw_version = PyBytes_FromString(cam_info.Version);
    PyObject *manufacture_date = PyBytes_FromString(cam_info.Date);
    PyObject *id = Py_BuildValue("B", cam_info.Select);
    PyObject *sensor_id = Py_BuildValue("H", sensor_info.SensorID);
    PyObject *sensor_name = PyBytes_FromString(sensor_info.strSensorName);
    PyObject *max_width = Py_BuildValue("I", sensor_info.nMaxWidth);
    PyObject *max_height = Py_BuildValue("I", sensor_info.nMaxHeight);
    PyObject *pixel_size = Py_BuildValue("d", sensor_info.wPixelSize/100.0);

    PyObject *type;
    switch (cam_info.Type) {
    case IS_CAMERA_TYPE_UEYE_USB_SE:
        type = PyBytes_FromString("USB uEye SE or RE");
        break;
    case IS_CAMERA_TYPE_UEYE_USB_ME:
        type = PyBytes_FromString("USB uEye ME");
        break;
    case IS_CAMERA_TYPE_UEYE_USB_LE:
        type = PyBytes_FromString("USB uEye LE");
        break;
    case IS_CAMERA_TYPE_UEYE_USB3_CP:
        type = PyBytes_FromString("USB 3 uEye CP");
        break;
    case IS_CAMERA_TYPE_UEYE_ETH_HE:
        type = PyBytes_FromString("GigE uEye HE");
        break;
    case IS_CAMERA_TYPE_UEYE_ETH_SE:
        type = PyBytes_FromString("GigE uEye SE or RE");
        break;
    case IS_CAMERA_TYPE_UEYE_ETH_LE:
        type = PyBytes_FromString("GigE uEye LE");
        break;
    case IS_CAMERA_TYPE_UEYE_ETH_CP:
        type = PyBytes_FromString("GigE uEye CP");
        break;
    default:
        type = PyBytes_FromString("Unknown");
    }

    PyObject *color_mode;
    switch (sensor_info.nColorMode) {
    case IS_COLORMODE_BAYER:
        color_mode = PyBytes_FromString("Bayer");
        break;
    case IS_COLORMODE_MONOCHROME:
        color_mode = PyBytes_FromString("Monochrome");
        break;
    case IS_COLORMODE_CBYCRY:
        color_mode = PyBytes_FromString("CBYCRY");
        break;
    default:
        color_mode = PyBytes_FromString("Unknown");
    }

    PyDict_SetItemString(dict, "serial_num", serial_num);
    PyDict_SetItemString(dict, "manufacturer", manufacturer);
    PyDict_SetItemString(dict, "hw_version", hw_version);
    PyDict_SetItemString(dict, "manufacture_date", manufacture_date);
    PyDict_SetItemString(dict, "id", id);
    PyDict_SetItemString(dict, "sensor_id", sensor_id);
    PyDict_SetItemString(dict, "sensor_name", sensor_name);
    PyDict_SetItemString(dict, "max_width", max_width);
    PyDict_SetItemString(dict, "max_height", max_height);
    PyDict_SetItemString(dict, "type", type);
    PyDict_SetItemString(dict, "color_mode", color_mode);
    PyDict_SetItemString(dict, "pixel_size", pixel_size);   /* in um */

    /* Gains */
    if (sensor_info.bMasterGain) {
        PyDict_SetItemString(dict, "master_gain", Py_True);
    }
    else {
        PyDict_SetItemString(dict, "master_gain", Py_False);
    }

    if (sensor_info.bRGain) {
        PyDict_SetItemString(dict, "red_gain", Py_True);
    }
    else {
        PyDict_SetItemString(dict, "red_gain", Py_False);
    }

    if (sensor_info.bGGain) {
        PyDict_SetItemString(dict, "green_gain", Py_True);
    }
    else {
        PyDict_SetItemString(dict, "green_gain", Py_False);
    }

    if (sensor_info.bBGain) {
        PyDict_SetItemString(dict, "blue_gain", Py_True);
    }
    else {
        PyDict_SetItemString(dict, "blue_gain", Py_False);
    }

    /* Global shutter, rolling if false */
    if (sensor_info.bGlobShutter) {
        PyDict_SetItemString(dict, "global_shutter", Py_True);
    }
    else {
        PyDict_SetItemString(dict, "global_shutter", Py_False);
    }

    Py_DECREF(serial_num);
    Py_DECREF(manufacturer);
    Py_DECREF(hw_version);
    Py_DECREF(manufacture_date);
    Py_DECREF(id);
    Py_DECREF(sensor_id);
    Py_DECREF(sensor_name);
    Py_DECREF(max_width);
    Py_DECREF(max_height);
    Py_DECREF(type);
    Py_DECREF(color_mode);
    Py_DECREF(pixel_size);

    return dict;
}

static int ids_core_Camera_setinfo(ids_core_Camera *self, PyObject *value, void *closure) {
    PyErr_SetString(PyExc_TypeError, "Cannot modify attribute 'info'");
    return -1;
}

PyObject *ids_core_Camera_getname(ids_core_Camera *self, void *closure) {
    Py_INCREF(self->name);
    return self->name;
}

static int ids_core_Camera_setname(ids_core_Camera *self, PyObject *value, void *closure) {
    PyErr_SetString(PyExc_TypeError, "Cannot modify attribute 'name'");
    return -1;
}

static PyObject *ids_core_Camera_getwidth(ids_core_Camera *self, void *closure) {
    return PyLong_FromLong(self->width);
}

static int ids_core_Camera_setwidth(ids_core_Camera *self, PyObject *value, void *closure) {
    PyErr_SetString(PyExc_NotImplementedError, "Changing image width not yet supported.");
    return -1;
}

static PyObject *ids_core_Camera_getheight(ids_core_Camera *self, void *closure) {
    return PyLong_FromLong(self->height);
}

static int ids_core_Camera_setheight(ids_core_Camera *self, PyObject *value, void *closure) {
    PyErr_SetString(PyExc_NotImplementedError, "Changing image height not yet supported.");
    return -1;
}

static PyObject *ids_core_Camera_getpixelclock(ids_core_Camera *self, void *closure) {
    UINT clock;
    int ret;

    ret = is_PixelClock(self->handle, IS_PIXELCLOCK_CMD_GET, &clock, sizeof(clock));
    switch (ret) {
    case IS_SUCCESS:
        return PyLong_FromLong(clock);
        break;
    default:
        raise_general_error(self, ret);
    }

    return NULL;
}

static int ids_core_Camera_setpixelclock(ids_core_Camera *self, PyObject *value, void *closure) {
    int ret, clock;
    PyObject *exception;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'pixelclock'");
        return -1;
    }

    clock = (int) PyLong_AsLong(value);
    exception = PyErr_Occurred();
    if (exception) {
        PyErr_SetString(exception, "Pixel clock value must be an int or long");
        return -1;
    }

    if (clock < 0) {
        PyErr_SetString(PyExc_ValueError, "Pixel clock must be positive.");
        return -1;
    }

    ret = is_PixelClock(self->handle, IS_PIXELCLOCK_CMD_SET, (void*) &clock, sizeof(clock));
    switch (ret) {
    case IS_SUCCESS:
        return 0;
        break;
    case IS_INVALID_PARAMETER:
        PyErr_SetString(PyExc_ValueError, "Pixel clock value out of range");
        break;
    default:
        raise_general_error(self, ret);
    }

    return -1;
}

static PyObject *ids_core_Camera_getcolor_mode(ids_core_Camera *self, void *closure) {
    int color = is_SetColorMode(self->handle, IS_GET_COLOR_MODE);

    return PyLong_FromLong(color);
}

static int ids_core_Camera_setcolor_mode(ids_core_Camera *self, PyObject *value, void *closure) {
    int color;
    PyObject *exception;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'color'");
        return -1;
    }

    color = (int) PyLong_AsLong(value);
    exception = PyErr_Occurred();
    if (exception) {
        PyErr_SetString(exception, "Color mode must be an int or long");
        return -1;
    }

    if (!set_color_mode(self, color)) {
        return -1;
    }

    return 0;
}

static PyObject *ids_core_Camera_getgain(ids_core_Camera *self, void *closure) {
    int gain = is_SetHardwareGain(self->handle, IS_GET_MASTER_GAIN,
            IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER);

    return PyLong_FromLong(gain);
}

static int ids_core_Camera_setgain(ids_core_Camera *self, PyObject *value, void *closure) {
    int ret, gain;
    PyObject *exception;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'gain'");
        return -1;
    }

    gain = (int) PyLong_AsLong(value);
    exception = PyErr_Occurred();
    if (exception) {
        PyErr_SetString(exception, "Gain must be an int or long");
        return -1;
    }

    ret = is_SetHardwareGain(self->handle, gain, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER);
    switch (ret) {
    case IS_SUCCESS:
        return 0;
    case IS_INVALID_PARAMETER:
        PyErr_SetString(PyExc_ValueError, "Gain out of range.");
        return -1;
    default:
        raise_general_error(self, ret);
        return -1;
    }

    return -1;
}

static PyObject *ids_core_Camera_getexposure(ids_core_Camera *self, void *closure) {
    double exposure;
    int ret;

    ret = is_Exposure(self->handle, IS_EXPOSURE_CMD_GET_EXPOSURE, &exposure, sizeof(exposure));
    switch (ret) {
    case IS_SUCCESS:
        return PyFloat_FromDouble(exposure);
        break;
    default:
        raise_general_error(self, ret);
    }

    return NULL;
}

static int ids_core_Camera_setexposure(ids_core_Camera *self, PyObject *value, void *closure) {
    int ret;
    double exposure;
    PyObject *exception;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'exposure'");
        return -1;
    }

    exposure = PyFloat_AsDouble(value);
    exception = PyErr_Occurred();
    if (exception) {
        PyErr_SetString(exception, "Exposure must be a number");
        return -1;
    }

    ret = is_Exposure(self->handle, IS_EXPOSURE_CMD_SET_EXPOSURE, (void*) &exposure, sizeof(exposure));
    switch (ret) {
    case IS_SUCCESS:
        return 0;
        break;
    case IS_INVALID_PARAMETER:
        PyErr_SetString(PyExc_ValueError, "Exposure out of range");
        break;
    default:
        raise_general_error(self, ret);
    }

    return -1;
}

static PyObject *ids_core_Camera_getauto_exposure(ids_core_Camera *self, void *closure) {
    double val;
    int ret;
    
    ret = is_SetAutoParameter(self->handle, IS_GET_ENABLE_AUTO_SHUTTER, &val, NULL);
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        raise_general_error(self, ret);
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

static int ids_core_Camera_setauto_exposure(ids_core_Camera *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'auto_exposure'");
        return -1;
    }

    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Auto exposure must be a bool.");
        return -1;
    }

    double val = (value == Py_True) ? 1 : 0;

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
        raise_general_error(self, ret);
        return -1;
    }

    return -1;
}

static PyObject *ids_core_Camera_getauto_exposure_brightness(ids_core_Camera *self, void *closure) {
    double val;
    int ret;

    ret = is_SetAutoParameter(self->handle, IS_GET_AUTO_REFERENCE, &val, NULL);
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        raise_general_error(self, ret);
        return NULL;
    }

    /* Value is 0..255, return proportion */
    return PyFloat_FromDouble(val/255);
}

static int ids_core_Camera_setauto_exposure_brightness(ids_core_Camera *self, PyObject *value, void *closure) {
    int ret;
    double val;
    PyObject *exception;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'auto_exposure_brightness'");
        return -1;
    }

    val = PyFloat_AsDouble(value);
    exception = PyErr_Occurred();
    if (exception) {
        PyErr_SetString(exception, "Exposure brightness must be a number");
        return -1;
    }

    if (val < 0 || val > 1) {
        PyErr_SetString(PyExc_ValueError, "Exposure brightness must be between 0 and 1");
        return -1;
    }

    /* Value is 0..255, extract from proportion */
    val = val * 255;

    ret = is_SetAutoParameter(self->handle, IS_SET_AUTO_REFERENCE, &val, NULL);
    switch (ret) {
    case IS_SUCCESS:
        return 0;
    case IS_INVALID_PARAMETER:
        PyErr_SetString(PyExc_ValueError, "Auto exposure brightness out of range");
        break;
    default:
        raise_general_error(self, ret);
        return -1;
    }

    return -1;
}

static PyObject *ids_core_Camera_getauto_speed(ids_core_Camera *self, void *closure) {
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

static int ids_core_Camera_setauto_speed(ids_core_Camera *self, PyObject *value, void *closure) {
    int ret;
    double val;
    PyObject *exception;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'auto_speed'");
        return -1;
    }

    val = PyFloat_AsDouble(value);
    exception = PyErr_Occurred();
    if (exception) {
        PyErr_SetString(exception, "Auto speed must be a number");
        return -1;
    }

    /* is_SetAutoParameter() returns IS_NO_SUCCESS for out-of-range values ... */
    if (val < 0 || val > 100) {
        PyErr_SetString(PyExc_ValueError, "Auto speed out of range (0...100)");
        return -1;
    }

    ret = is_SetAutoParameter(self->handle, IS_SET_AUTO_SPEED, &val, NULL);
    switch (ret) {
    case IS_SUCCESS:
        return 0;
    case IS_INVALID_PARAMETER:
        PyErr_SetString(PyExc_ValueError, "Auto speed out of range (0..100)");
        break;
    default:
        raise_general_error(self, ret);
        return -1;
    }

    return -1;
}

static PyObject *ids_core_Camera_getauto_white_balance(ids_core_Camera *self, void *closure) {
    double val;
    UINT val2;
    int ret;
    
    ret = is_SetAutoParameter(self->handle, IS_GET_ENABLE_AUTO_WHITEBALANCE, &val, NULL);
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        raise_general_error(self, ret);
        return NULL;
    }

    ret = is_AutoParameter(self->handle, IS_AWB_CMD_GET_ENABLE, &val2, sizeof(val2));
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        raise_general_error(self, ret);
        return NULL;
    }
        
    if (val && val2) {
        Py_INCREF(Py_True);
        return Py_True;
    }
    else if (val || val2) {
        PyErr_SetString(PyExc_RuntimeError, "Only one white balance setting is enabled.  It may or may not work.");
        return NULL;
    }
    else {
        Py_INCREF(Py_False);
        return Py_False;
    }
}

static int ids_core_Camera_setauto_white_balance(ids_core_Camera *self, PyObject *value, void *closure) {
    int ret;
    double val;
    UINT val2;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'auto_white_balance'");
        return -1;
    }

    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Auto white balance must be a bool.");
        return -1;
    }

    val = (value == Py_True) ? 1 : 0;
    val2 = (UINT) val;

    ret = is_SetAutoParameter(self->handle, IS_SET_ENABLE_AUTO_WHITEBALANCE, &val, NULL);
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        goto err;
    }

    ret = is_AutoParameter(self->handle, IS_AWB_CMD_SET_ENABLE, &val2, sizeof(val2));
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        goto err_reset_set_auto_param;
    }

    if (val2) {
        UINT nType = IS_AWB_COLOR_TEMPERATURE;
        ret = is_AutoParameter(self->handle, IS_AWB_CMD_SET_TYPE, (void*)&nType, sizeof(nType));
        switch (ret) {
        case IS_SUCCESS:
            break;
        default:
            goto err_reset_auto_param;
        }
    }

    if (val > 0) {
        self->autofeatures++;
    }
    else {
        self->autofeatures--;
    }

    return 0;

err_reset_auto_param:
    val2 = 0;
    is_AutoParameter(self->handle, IS_AWB_CMD_SET_ENABLE, &val2, sizeof(val2));
err_reset_set_auto_param:
    val = 0;
    is_SetAutoParameter(self->handle, IS_SET_ENABLE_AUTO_WHITEBALANCE, &val, NULL);
err:
    raise_general_error(self, ret);
    return -1;
}

static PyObject *ids_core_Camera_getcolor_correction(ids_core_Camera *self, void *closure) {
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
        raise_general_error(self, ret);
        return NULL;
    }
}

static int ids_core_Camera_setcolor_correction(ids_core_Camera *self, PyObject *value, void *closure) {
    int ret;
    double factor;
    PyObject *exception;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'color_correction'");
        return -1;
    }

    /* Disable color correction */
    if (value == Py_False) {
        ret = is_SetColorCorrection(self->handle, IS_CCOR_DISABLE, &factor);
        switch (ret) {
        case IS_SUCCESS:
            return 0;
        default:
            raise_general_error(self, ret);
            return -1;
        }
    }

    factor = PyFloat_AsDouble(value);
    exception = PyErr_Occurred();
    if (exception) {
        PyErr_SetString(exception, "Color correction factor must be a number, or False to disable");
        return -1;
    }

    ret = is_SetColorCorrection(self->handle, IS_CCOR_SET_IR_AUTOMATIC, &factor);
    switch (ret) {
    case IS_SUCCESS:
        return 0;
    case IS_INVALID_PARAMETER:
        PyErr_SetString(PyExc_ValueError, "Color correction factor out of range (0 to 1)");
        break;
    default:
        raise_general_error(self, ret);
        return -1;
    }

    return -1;
}

static PyObject *ids_core_Camera_getcontinuous_capture(ids_core_Camera *self,
                                                       void *closure) {
    int ret;

    ret = is_CaptureVideo(self->handle, IS_GET_LIVE);
    if (ret == TRUE) {
        Py_INCREF(Py_True);
        return Py_True;
    }

    Py_INCREF(Py_False);
    return Py_False;
}

static int ids_core_Camera_setcontinuous_capture(ids_core_Camera *self,
                                                 PyObject *value, void *closure) {
    int ret;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete attribute 'continuous_capture'");
        return -1;
    }

    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Attribute 'continuous_capture' must be boolean");
        return -1;
    }

    /* Enable continuous capture */
    if (value == Py_True) {
        ret = is_CaptureVideo(self->handle, IS_DONT_WAIT);
        switch (ret) {
        case IS_SUCCESS:
            break;
        case IS_TIMED_OUT:
            PyErr_SetString(PyExc_IOError, "Continuous capture start timed out.");
            return -1;
        case IS_NO_ACTIVE_IMG_MEM:
            PyErr_SetString(PyExc_IOError, "No image memory available.");
            return -1;
        default:
            raise_general_error(self, ret);
            return -1;
        }
    }
    /* Disable continuous capture */
    else if (value == Py_False) {
        ret = is_StopLiveVideo(self->handle, IS_FORCE_VIDEO_STOP);
        switch (ret) {
        case IS_SUCCESS:
            break;
        default:
            raise_general_error(self, ret);
            return -1;
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError, "Unknown boolean value");
        return -1;
    }

    return 0;
}

PyGetSetDef ids_core_Camera_getseters[] = {
    {"info", (getter) ids_core_Camera_getinfo, (setter) ids_core_Camera_setinfo, "Camera info", NULL},
    {"name", (getter) ids_core_Camera_getname, (setter) ids_core_Camera_setname, "Camera manufacturer and name", NULL},
    {"width", (getter) ids_core_Camera_getwidth, (setter) ids_core_Camera_setwidth, "Image width", NULL},
    {"height", (getter) ids_core_Camera_getheight, (setter) ids_core_Camera_setheight, "Image height", NULL},
    {"pixelclock", (getter) ids_core_Camera_getpixelclock, (setter) ids_core_Camera_setpixelclock, "Pixel Clock of camera", NULL},
    {"color_mode", (getter) ids_core_Camera_getcolor_mode, (setter) ids_core_Camera_setcolor_mode,
        "Color mode of images.\n\n"
        "It is recommended to change color mode only when not\n"
        "capturing images, and to free and reallocate memory\n"
        "after changing, as the new color mode may have a different\n"
        "bit depth.", NULL},
    {"gain", (getter) ids_core_Camera_getgain, (setter) ids_core_Camera_setgain, "Hardware gain (individual RGB gains not yet supported)", NULL},
    {"exposure", (getter) ids_core_Camera_getexposure, (setter) ids_core_Camera_setexposure, "Exposure time", NULL},
    {"auto_exposure", (getter) ids_core_Camera_getauto_exposure, (setter) ids_core_Camera_setauto_exposure, "Auto exposure", NULL},
    {"auto_exposure_brightness", (getter) ids_core_Camera_getauto_exposure_brightness, (setter) ids_core_Camera_setauto_exposure_brightness, "Auto exposure reference brightness (0 to 1)", NULL},
    {"auto_speed", (getter) ids_core_Camera_getauto_speed, (setter) ids_core_Camera_setauto_speed, "Auto speed", NULL},
    {"auto_white_balance", (getter) ids_core_Camera_getauto_white_balance, (setter) ids_core_Camera_setauto_white_balance, "Auto White Balance", NULL},
    {"color_correction", (getter) ids_core_Camera_getcolor_correction, (setter) ids_core_Camera_setcolor_correction, "IR color correction factor", NULL},
    {"continuous_capture", (getter) ids_core_Camera_getcontinuous_capture, (setter) ids_core_Camera_setcontinuous_capture,
        "Enable or disable camera continuous capture (free-run) mode.\n\n"
        "Once set to True, continuous capture is enabled, and methods\n"
        "to retrieve images can be called.", NULL},
    {NULL}
};
