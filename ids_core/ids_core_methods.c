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

static PyObject *ids_core_number_cameras(PyObject *self, PyObject *args) {
    int num_cams, ret;

    ret = is_GetNumberOfCameras(&num_cams);
    if (ret != IS_SUCCESS) {
        PyErr_Format(IDSError, "uEye SDK error %d", ret);
        return NULL;
    }

    return Py_BuildValue("i", num_cams);
}

static PyObject *ids_core_camera_list(PyObject *self, PyObject *args) {
    int num_cams, ret, i;
    UEYE_CAMERA_LIST *cameras;
    PyObject *list = PyList_New(0);
    void* cam_data = NULL;

    ret = is_GetNumberOfCameras(&num_cams);
    if (ret != IS_SUCCESS) {
        Py_DECREF(list);
        PyErr_Format(IDSError, "uEye SDK error %d", ret);
        return NULL;
    }

    if (!num_cams) {
        return list;
    }

    /*
     * This is insane.
     *
     * IDS expects us to dynamically resize UEYE_CAMERA_LIST for the
     * appropriate number of cameras.  Thus, we build a structure of the
     * appropriate size on the stack, and then cast it to UEYE_CAMERA_LIST
     */
    cam_data = malloc(sizeof(ULONG) + num_cams * sizeof(UEYE_CAMERA_INFO));
    if (!cam_data) {
        Py_DECREF(list);
        PyErr_Format(IDSError, "malloc() failed");
        return NULL;
    }
    cameras = (UEYE_CAMERA_LIST *) cam_data;
    cameras->dwCount = num_cams;

    ret = is_GetCameraList(cameras);
    if (ret != IS_SUCCESS) {
        free(cam_data);
        return NULL;
    }
    

    for (i = 0; i < cameras->dwCount; i++) {
        PyObject *camera_info = PyDict_New();

        PyObject *camera_id = Py_BuildValue("I", cameras->uci[i].dwCameraID);
        PyObject *device_id = Py_BuildValue("I", cameras->uci[i].dwDeviceID);
        PyObject *sensor_id = Py_BuildValue("I", cameras->uci[i].dwSensorID);
        PyObject *in_use = Py_BuildValue("I", cameras->uci[i].dwInUse);
        PyObject *serial_number = Py_BuildValue("s", cameras->uci[i].SerNo);
        PyObject *model = Py_BuildValue("s", cameras->uci[i].Model);
        PyObject *status = Py_BuildValue("I", cameras->uci[i].dwStatus);

        PyDict_SetItemString(camera_info, "camera_id", camera_id);
        PyDict_SetItemString(camera_info, "device_id", device_id);
        PyDict_SetItemString(camera_info, "sensor_id", sensor_id);
        PyDict_SetItemString(camera_info, "in_use", in_use);
        PyDict_SetItemString(camera_info, "serial_number", serial_number);
        PyDict_SetItemString(camera_info, "model", model);
        PyDict_SetItemString(camera_info, "status", status);

        Py_DECREF(camera_id);
        Py_DECREF(device_id);
        Py_DECREF(sensor_id);
        Py_DECREF(in_use);
        Py_DECREF(serial_number);
        Py_DECREF(model);
        Py_DECREF(status);

        PyList_Append(list, camera_info);

        Py_DECREF(camera_info);
    }

    free(cam_data);
    return list;
}

PyMethodDef ids_coreMethods[] = {
    {"number_cameras", ids_core_number_cameras, METH_VARARGS,
        "number_cameras() -> number of cameras connected\n\n"
        "Determines total number of cameras available.\n\n"
        "Returns:\n"
        "    Total number of cameras available.\n\n"
        "Raises:\n"
        "    IDSError: An unknown error occured in the uEye SDK."
    },
    {"camera_list", ids_core_camera_list, METH_VARARGS,
        "camera_list() -> list of cameras available\n\n"
        "Gets information on all available cameras, including camera handle,\n"
        "which can be used to select a camera to open.\n\n"
        "Returns:\n"
        "    List of dictionaries with information for each available camera.\n\n"
        "Raises:\n"
        "    IDSError: An unknown error occured in the uEye SDK."
    },
    {NULL, NULL, 0, NULL}
};
