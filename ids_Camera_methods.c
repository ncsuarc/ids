#include <Python.h>
#include <structmember.h>
#include <ueye.h>
#include <wchar.h>

#define PY_ARRAY_UNIQUE_SYMBOL  ids_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "ids.h"

static PyObject *ids_Camera_close(ids_Camera *self, PyObject *args, PyObject *kwds);
static PyObject *ids_Camera_start_queue(ids_Camera *self, PyObject *args, PyObject *kwds);
static PyObject *ids_Camera_freeze_save(ids_Camera *self, PyObject *args, PyObject *kwds);
static PyObject *ids_Camera_freeze(ids_Camera *self, PyObject *args, PyObject *kwds);

static PyObject *create_matrix(ids_Camera *self, uint8_t *mem);

PyMethodDef ids_Camera_methods[] = {
    {"close", (PyCFunction) ids_Camera_close, METH_VARARGS, "Closes open camera"},
    {"start_queue", (PyCFunction) ids_Camera_start_queue, METH_VARARGS, "Initializes image buffer queue mode."},
    {"freeze_save", (PyCFunction) ids_Camera_freeze_save, METH_VARARGS, "Capture an image and save it."},
    {"freeze", (PyCFunction) ids_Camera_freeze, METH_VARARGS, "Capture an image."},
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

static PyObject *ids_Camera_start_queue(ids_Camera *self, PyObject *args, PyObject *kwds) {
    int ret = is_InitImageQueue(self->handle, 0);
    if (ret != IS_SUCCESS) {
        PyErr_SetString(PyExc_IOError, "Unable to start image queue.");
        Py_INCREF(Py_False);
        return Py_False;
    }

    Py_INCREF(Py_True);
    return Py_True;
}

static PyObject *ids_Camera_freeze_save(ids_Camera *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"filename", NULL};
    char *filename;
    wchar_t fancy_filename[256];

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filename)) {
        PyErr_SetString(PyExc_TypeError, "Filename must be a string.");
        return NULL;
    }

    swprintf(fancy_filename, 256, L"%hs", filename);

    int ret;
    ret = is_FreezeVideo(self->handle, IS_WAIT);
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        PyErr_SetString(PyExc_IOError, "Failed to capture image.");
        return NULL;
    }

    IMAGE_FILE_PARAMS ImageFileParams;
    ImageFileParams.pwchFileName = fancy_filename;
    ImageFileParams.nFileType = IS_IMG_BMP;
    ImageFileParams.ppcImageMem = NULL;
    ImageFileParams.pnImageID = NULL;
    ret = is_ImageFile(self->handle, IS_IMAGE_FILE_CMD_SAVE, (void*)&ImageFileParams, sizeof(ImageFileParams));
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        PyErr_SetString(PyExc_IOError, "Failed to save image.");
        return NULL;
    }

    Py_INCREF(Py_True);
    return Py_True;
}

static PyObject *ids_Camera_freeze(ids_Camera *self, PyObject *args, PyObject *kwds) {
    int ret;
    ret = is_FreezeVideo(self->handle, IS_WAIT);
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        PyErr_SetString(PyExc_IOError, "Failed to capture image.");
        return NULL;
    }

    uint8_t *mem;
    ret = is_GetImageMem(self->handle, (void *) &mem);
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        PyErr_SetString(PyExc_IOError, "Failed to capture image.");
        return NULL;
    }

    return create_matrix(self, mem);
}

static PyObject *create_matrix(ids_Camera *self, uint8_t *mem) {
    int color = is_SetColorMode(self->handle, IS_GET_COLOR_MODE);
    PyArrayObject* matrix;

    switch (color) {
    case IS_CM_BAYER_RG8: {
        npy_intp dims[2];
        dims[0] = self->height;
        dims[1] = self->width;

        matrix = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_UINT8);
        memcpy(PyArray_DATA(matrix), mem, self->bitdepth/8 * dims[0] * dims[1]);
        break; 
    }
    case IS_CM_BAYER_RG12: /* You need to left shift the output by 4 bits */
    case IS_CM_BAYER_RG16: {
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
    case IS_CM_RGBY8_PACKED: {
        npy_intp dims[3];
        dims[0] = self->height;
        dims[1] = self->width;
        dims[2] = 4;

        matrix = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_UINT8);
        memcpy(PyArray_DATA(matrix), mem, self->bitdepth/8 * dims[0] * dims[1]);
        break;
    }
    default:
        PyErr_SetString(PyExc_NotImplementedError, "Unsupport color format for conversion to array.");
        return NULL;
    }

    return (PyObject*)matrix;
}
