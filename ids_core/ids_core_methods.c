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
    int num_cams, ret;
    UEYE_CAMERA_LIST    *cameras;
    PyObject *list = PyList_New(0);

    ret = is_GetNumberOfCameras(&num_cams);
    if (ret != IS_SUCCESS) {
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
    uint8_t cam_data[sizeof(cameras->dwCount) + num_cams * sizeof(cameras->uci)];
    cameras = (UEYE_CAMERA_LIST *) &cam_data;
    cameras->dwCount = num_cams;

    ret = is_GetCameraList(cameras);
    if (ret != IS_SUCCESS) {
        PyErr_Format(IDSError, "uEye SDK error %d", ret);
        return NULL;
    }

    for (int i = 0; i < cameras->dwCount; i++) {
        PyObject *camera_info = PyDict_New();

        PyDict_SetItemString(camera_info, "camera_id", Py_BuildValue("I", cameras->uci[i].dwCameraID));
        PyDict_SetItemString(camera_info, "device_id", Py_BuildValue("I", cameras->uci[i].dwDeviceID));
        PyDict_SetItemString(camera_info, "sensor_id", Py_BuildValue("I", cameras->uci[i].dwSensorID));
        PyDict_SetItemString(camera_info, "in_use", Py_BuildValue("I", cameras->uci[i].dwInUse));
        PyDict_SetItemString(camera_info, "serial_number", Py_BuildValue("s", cameras->uci[i].SerNo));
        PyDict_SetItemString(camera_info, "model", Py_BuildValue("s", cameras->uci[i].Model));
        PyDict_SetItemString(camera_info, "status", Py_BuildValue("I", cameras->uci[i].dwStatus));

        PyList_Append(list, camera_info);
    }

    return list;
}

PyMethodDef ids_coreMethods[] = {
    {"number_cameras", ids_core_number_cameras, METH_VARARGS, "number_cameras() -> number of cameras connected."},
    {"camera_list", ids_core_camera_list, METH_VARARGS, "camera_list() -> list of cameras with metadata."},
    {NULL, NULL, 0, NULL}
};
