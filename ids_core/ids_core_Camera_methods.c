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
#include <wchar.h>
#include <stdio.h>
#include <sys/queue.h>

#define PY_ARRAY_UNIQUE_SYMBOL  ids_core_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "ids_core.h"

#define IMG_TIMEOUT 500
#define NUM_TRIES 5

static PyObject *create_matrix(ids_core_Camera *self, char *mem);

static PyObject *ids_core_Camera_capture_status(ids_core_Camera *self, PyObject *args, PyObject *kwds) {
    int ret;
    UEYE_CAPTURE_STATUS_INFO capture_status;

    ret = is_CaptureStatus(self->handle, IS_CAPTURE_STATUS_INFO_CMD_GET,
                           (void *)&capture_status, sizeof(capture_status));
    if (ret != IS_SUCCESS) {
        raise_general_error(self, ret);
    }

    PyObject *dict = PyDict_New();
    PyObject *total = Py_BuildValue("I", capture_status.dwCapStatusCnt_Total);
    PyObject *no_destination_mem = Py_BuildValue("I",
            capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_API_NO_DEST_MEM]);
    PyObject *conversion_failed = Py_BuildValue("I",
            capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_API_CONVERSION_FAILED]);
    PyObject *image_locked = Py_BuildValue("I",
            capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_API_IMAGE_LOCKED]);
    PyObject *no_driver_mem = Py_BuildValue("I",
            capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_DRV_OUT_OF_BUFFERS]);
    PyObject *device_not_available = Py_BuildValue("I",
            capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_DRV_DEVICE_NOT_READY]);
    PyObject *usb_transfer_failed = Py_BuildValue("I",
            capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_USB_TRANSFER_FAILED]);
    PyObject *device_timeout = Py_BuildValue("I",
            capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_DEV_TIMEOUT]);
    PyObject *eth_buffer_overrun = Py_BuildValue("I",
            capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_ETH_BUFFER_OVERRUN]);
    PyObject *eth_missed_images = Py_BuildValue("I",
            capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_ETH_MISSED_IMAGES]);

    PyDict_SetItemString(dict, "total", total);
    PyDict_SetItemString(dict, "no_destination_mem", no_destination_mem);
    PyDict_SetItemString(dict, "conversion_failed", conversion_failed);
    PyDict_SetItemString(dict, "image_locked", image_locked);
    PyDict_SetItemString(dict, "no_driver_mem", no_driver_mem);
    PyDict_SetItemString(dict, "device_not_available", device_not_available);
    PyDict_SetItemString(dict, "usb_transfer_failed", usb_transfer_failed);
    PyDict_SetItemString(dict, "device_timeout", device_timeout);
    PyDict_SetItemString(dict, "eth_buffer_overrun", eth_buffer_overrun);
    PyDict_SetItemString(dict, "eth_missed_images", eth_missed_images);

    Py_DECREF(total);
    Py_DECREF(no_destination_mem);
    Py_DECREF(conversion_failed);
    Py_DECREF(image_locked);
    Py_DECREF(no_driver_mem);
    Py_DECREF(device_not_available);
    Py_DECREF(usb_transfer_failed);
    Py_DECREF(device_timeout);
    Py_DECREF(eth_buffer_overrun);
    Py_DECREF(eth_missed_images);

    /* Reset errors */
    ret = is_CaptureStatus(self->handle, IS_CAPTURE_STATUS_INFO_CMD_RESET, NULL, 0);
    if (ret != IS_SUCCESS) {
        Py_DECREF(dict);
        raise_general_error(self, ret);
        return NULL;
    }

    return dict;
}

static int add_mem(ids_core_Camera *self, char *mem, int id) {
    struct allocated_mem *node = malloc(sizeof(struct allocated_mem));
    if (node == NULL) {
        return -1;
    }

    node->mem = mem;
    node->id = id;
    LIST_INSERT_HEAD(&self->mem_list, node, list);

    return 0;
}

static PyObject *ids_core_Camera_alloc(ids_core_Camera *self, PyObject *args, PyObject *kwds) {
    char *mem;
    int id, ret;

    ret = is_AllocImageMem(self->handle, self->width, self->height,
                           self->bitdepth, &mem, &id);
    if (ret != IS_SUCCESS) {
        goto err;
    }

    ret = is_AddToSequence(self->handle, mem, id);
    if (ret != IS_SUCCESS) {
        goto err_free;
    }

    if (add_mem(self, mem, id) != 0) {
        goto err_free;
    }

    Py_INCREF(Py_None);
    return Py_None;

err_free:
    is_FreeImageMem(self->handle, mem, id);
err:
    PyErr_SetString(PyExc_MemoryError, "Unable to allocate image memory.");
    return NULL;
}

PyObject *ids_core_Camera_free_all(ids_core_Camera *self, PyObject *args, PyObject *kwds) {
    is_ClearSequence(self->handle);

    while (!LIST_EMPTY(&self->mem_list)) {
        struct allocated_mem *mem = LIST_FIRST(&self->mem_list);
        is_FreeImageMem(self->handle, mem->mem, mem->id);
        LIST_REMOVE(mem, list);
        free(mem);
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *ids_core_Camera_close(ids_core_Camera *self, PyObject *args, PyObject *kwds) {
    int ret = is_ExitCamera(self->handle);
    if (ret != IS_SUCCESS) {
        raise_general_error(self, ret);
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

/* Gets next image with is_WaitForNextImage().
 * Returns zero on success, non-zero on failure,
 * with exception set. */
static int get_next_image(ids_core_Camera *self, char **mem, INT *image_id, int timeout) {
    int ret;

	ret = is_CaptureVideo(self->handle, IS_GET_LIVE);

	if (ret == FALSE)
    	is_FreezeVideo(self->handle, IS_DONT_WAIT);

    ret = is_WaitForNextImage(self->handle, timeout, mem, image_id);

    switch (ret) {
    case IS_SUCCESS:
        break;
    case IS_TIMED_OUT:
        PyErr_Format(IDSTimeoutError, "Timeout of %dms exceeded", timeout);
        return 1;
    case IS_CAPTURE_STATUS:
        PyErr_SetString(IDSCaptureStatus, "Transfer error.  Check capture status.");
        return 1;
    default:
        raise_general_error(self, ret);
        return 1;
    }

    return 0;
}

static PyObject *ids_core_Camera_next_save(ids_core_Camera *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"filename", "filetype", "quality", NULL};
    char *filename;
    wchar_t fancy_filename[256];
    int filetype = IS_IMG_JPG;
    unsigned int quality = 100;
	int timeout = IMG_TIMEOUT;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|iI", kwlist, &filename, &filetype, &quality)) {
        return NULL;
    }

    swprintf(fancy_filename, 256, L"%hs", filename);

    if (filetype != IS_IMG_JPG && filetype != IS_IMG_PNG && filetype != IS_IMG_BMP) {
        PyErr_SetString(PyExc_ValueError, "Invalid image filetype");
    }

    int ret;
    char *mem;
    INT image_id;

    ret = get_next_image(self, &mem, &image_id, timeout);
    if (ret) {
        /* Exception set, return */
        return NULL;
    }

    IMAGE_FILE_PARAMS ImageFileParams;
    ImageFileParams.pwchFileName = fancy_filename;
    ImageFileParams.nFileType = filetype;
    ImageFileParams.nQuality = quality;
    ImageFileParams.ppcImageMem = &mem;
    ImageFileParams.pnImageID = (UINT*) &image_id;
    ret = is_ImageFile(self->handle, IS_IMAGE_FILE_CMD_SAVE, (void*)&ImageFileParams, sizeof(ImageFileParams));
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        raise_general_error(self, ret);
        return NULL;
    }

    PyObject *info = image_info(self, image_id);
    if (!info) {
        return NULL;
    }
    
    ret = is_UnlockSeqBuf(self->handle, image_id, mem);
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        Py_DECREF(info);
        raise_general_error(self, ret);
        return NULL;
    }

    return info;
}

static PyObject *ids_core_Camera_next(ids_core_Camera *self, PyObject *args, PyObject *kwds) {
	static char *kwlist[] = {"timeout"};    
	int ret;
    char *mem;
    INT image_id;

	int timeout;


	printf(args);

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", kwlist, &timeout))
		timeout = IMG_TIMEOUT;

    ret = get_next_image(self, &mem, &image_id, timeout);
    if (ret) {
        /* Exception set, return */
        return NULL;
    }

    PyObject *image = create_matrix(self, mem);
    if (!image) {
        return NULL;
    }

    PyObject *info = image_info(self, image_id);
    if (!info) {
        Py_DECREF(image);
        return NULL;
    }
    
    ret = is_UnlockSeqBuf(self->handle, image_id, mem);
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        Py_DECREF(image);
        Py_DECREF(info);
        raise_general_error(self, ret);
        return NULL;
    }

    PyObject *tuple = Py_BuildValue("(OO)", image, info);

    /* BuildValue INCREF's these objects, but we don't need them anymore */
    Py_DECREF(image);
    Py_DECREF(info);

    return tuple;
}

/* Create NumPy array for image in mem, and copy image data into it */
static PyObject *create_matrix(ids_core_Camera *self, char *mem) {
    int color = is_SetColorMode(self->handle, IS_GET_COLOR_MODE);
    PyArrayObject* matrix;

    switch (color) {
    case IS_CM_SENSOR_MONO8:
    case IS_CM_SENSOR_RAW8: {
        npy_intp dims[2];
        dims[0] = self->height;
        dims[1] = self->width;

        matrix = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_UINT8);
        memcpy(PyArray_DATA(matrix), mem, self->bitdepth/8 * dims[0] * dims[1]);
        break; 
    }
    case IS_CM_SENSOR_RAW12: /* You need to left shift the output by 4 bits */
    case IS_CM_SENSOR_RAW16: {
        npy_intp dims[2];
        dims[0] = self->height;
        dims[1] = self->width;

        matrix = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_UINT16);
        memcpy(PyArray_DATA(matrix), mem, self->bitdepth/8 * dims[0] * dims[1]);
        break;
    }
    case IS_CM_BGRA8_PACKED:
    case IS_CM_BGRY8_PACKED:
    case IS_CM_RGBA8_PACKED:
    case IS_CM_RGBY8_PACKED: 
    case IS_CM_BGR8_PACKED:
    case IS_CM_RGB8_PACKED: {
        npy_intp dims[3];
        dims[0] = self->height;
        dims[1] = self->width;
        dims[2] = self->bitdepth/8;

        matrix = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_UINT8);
        memcpy(PyArray_DATA(matrix), mem, self->bitdepth/8 * dims[0] * dims[1]);
        break;
    }
    default:
        PyErr_SetString(PyExc_NotImplementedError, "Unsupported color format for conversion to array.");
        return NULL;
    }

    return (PyObject*)matrix;
}

PyMethodDef ids_core_Camera_methods[] = {
    {"capture_status", (PyCFunction) ids_core_Camera_capture_status, METH_NOARGS,
        "capture_status() -> status\n\n"
        "Get internal camera and driver errors\n\n"
        "Dictionary contains counts of occurrences of various internal errors,\n"
        "which are documented in the IDS SDK is_CaptureStatus() documentation.\n\n"
        "Error counts reset every time this function is called.\n\n"
        "Returns:\n"
        "    Dictionary of internal camera and driver errors.\n\n"
        "Raises:\n"
        "    IDSError: An unknown error occured in the uEye SDK."
    },
    {"alloc", (PyCFunction) ids_core_Camera_alloc, METH_NOARGS,
        "alloc()\n\n"
        "Allocates a single memory location for storing images.\n"
        "Memory locations must be allocated before capturing images.\n\n"
        "Raises:\n"
        "    MemoryError: Unable to allocate memory."
    },
    {"free_all", (PyCFunction) ids_core_Camera_free_all, METH_NOARGS,
        "free_all()\n\n"
        "Frees all allocated memory for storing images."
    },
    {"close", (PyCFunction) ids_core_Camera_close, METH_NOARGS,
        "close()\n\n"
        "Closes open camera.\n\n"
        "Raises:\n"
        "    IDSError: An unknown error occured in the uEye SDK."
    },
    {"next_save", (PyCFunction) ids_core_Camera_next_save, METH_VARARGS | METH_KEYWORDS,
        "next_save(filename [, filetype=ids_core.FILETYPE_JPG, quality=100]) -> metadata\n\n"
        "Saves next available image.\n\n"
        "Using the uEye SDK image saving functions to save the next available\n"
        "image to disk.  Blocks until image is available, or timeout occurs.\n\n"
        "Arguments:\n"
        "    filename: File to save image to.\n"
        "    filetype: Filetype to save as, one of ids_core.FILETYPE_*\n"
        "    quality: Image quality for JPEG and PNG, with 100 as maximum quality\n\n"
        "Returns:\n"
        "    Dictionary containing image metadata.  Timestamp is provided in UTC.\n\n"
        "Raises:\n"
        "    ValueError: Invalid filetype.\n"
        "    IDSTimeoutError: An image was not available within the timeout.\n"
        "    IDSError: An unknown error occured in the uEye SDK."
    },
    {"next", (PyCFunction) ids_core_Camera_next, METH_VARARGS | METH_KEYWORDS,
        "next() -> image, metadata\n\n"
        "Gets next available image.\n\n"
        "Gets the next available image from the camera as a Numpy array\n"
        "Blocks until image is available, or timeout occurs.\n\n"
        "Returns:\n"
        "    (image, metadata) tuple, where image is a Numpy array containing\n"
        "    the image, and metadata is a dictionary containing image metadata.\n"
        "    Timestamp is provided as a UTC datetime object\n\n"
        "Raises:\n"
        "    IDSTimeoutError: An image was not available within the timeout.\n"
        "    IDSError: An unknown error occured in the uEye SDK."
        "    NotImplementedError: The current color format cannot be converted\n"
        "        to a numpy array."
    },
    {NULL}
};
