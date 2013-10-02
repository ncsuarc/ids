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

#define PY_ARRAY_UNIQUE_SYMBOL  ids_core_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "tiffio.h"

#include "ids_core.h"

#define IMG_TIMEOUT 3000
#define NUM_TRIES 5

static PyObject *create_matrix(ids_core_Camera *self, char *mem);

static PyObject *ids_core_Camera_close(ids_core_Camera *self, PyObject *args, PyObject *kwds) {
    if (is_ExitCamera(self->handle) != IS_SUCCESS) {
        Py_INCREF(Py_False);
        return Py_False;
    }

    Py_INCREF(Py_True);
    return Py_True;
}

static PyObject *ids_core_Camera_start_continuous(ids_core_Camera *self, PyObject *args, PyObject *kwds) {
    int ret = is_CaptureVideo(self->handle, IS_DONT_WAIT);
    switch (ret) {
    case IS_SUCCESS:
        break;
    case IS_TIMED_OUT:
        PyErr_SetString(PyExc_IOError, "Continuous capture start timed out.");
        return NULL;
    default:
        PyErr_SetString(PyExc_IOError, "Unable to start continuous capture.");
        return NULL;
    }

    Py_INCREF(Py_True);
    return Py_True;
}

static void warn_capture_status(ids_core_Camera *self) {
    UEYE_CAPTURE_STATUS_INFO capture_status;
    int r = is_CaptureStatus(self->handle, IS_CAPTURE_STATUS_INFO_CMD_GET, (void *) &capture_status, sizeof(capture_status));
    if (r == IS_SUCCESS) {
        if (capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_API_NO_DEST_MEM]) {
            printf("Warning: out of memory locations for images, retrying. ");
        }
        else if (capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_API_CONVERSION_FAILED]) {
            printf("Warning: image conversion failed, retrying. ");
        }
        else if (capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_API_IMAGE_LOCKED]) {
            printf("Warning: destination buffer locked, retrying. ");
        }
        else if (capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_DRV_OUT_OF_BUFFERS]) {
            printf("Warning: no internal memory available, image lost, retrying. ");
        }
        else if (capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_DRV_DEVICE_NOT_READY]) {
            printf("Warning: camera not available, retrying. ");
        }
        else if (capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_USB_TRANSFER_FAILED]) {
            printf("Warning: transfer failed, retrying. ");
        }
        else if (capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_DEV_TIMEOUT]) {
            printf("Warning: camera timed out, retrying. ");
        }
        else if (capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_ETH_BUFFER_OVERRUN]) {
            printf("Warning: camera buffer overflow, retrying. ");
        }
        else if (capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_ETH_MISSED_IMAGES]) {
            printf("Warning: missed %d image(s), retrying. ", capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_ETH_MISSED_IMAGES]);
        }
        else {
            printf("Warning: Capture Status, total: %d. ", capture_status.dwCapStatusCnt_Total);
        }

        is_CaptureStatus(self->handle, IS_CAPTURE_STATUS_INFO_CMD_RESET, NULL, 0);
    }
    else {
        printf("Warning: Capture Status failed. ");
    }
    fflush(stdout);
}

/* Gets next image with is_WaitForNextImage().
 * Returns zero on success, non-zero on failure,
 * with exception set. */
static int get_next_image(ids_core_Camera *self, char **mem, INT *image_id) {
    int ret;
    int tries = 0;

retry:
    tries++;
    ret = is_WaitForNextImage(self->handle, IMG_TIMEOUT, mem, image_id); 

    switch (ret) {
    case IS_SUCCESS:
        break;
    case IS_TIMED_OUT:
        printf("Warning: Capture timed out, retrying. ");
        fflush(stdout);
        if (tries < NUM_TRIES)
            goto retry;
        else {
            PyErr_SetString(PyExc_IOError, "Too many timeout retries.");
            return 1;
        }
    case IS_CAPTURE_STATUS: {
        warn_capture_status(self);
        goto retry;
    }
    default:
        PyErr_Format(PyExc_IOError,  "Failed to capture image on WaitForNextImage.  ret = %d", ret);
        return 1;
    }

    return 0;
}

static PyObject *ids_core_Camera_next_save(ids_core_Camera *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"filename", "filetype", NULL};
    char *filename;
    wchar_t fancy_filename[256];
    int filetype = IS_IMG_JPG;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|i", kwlist, &filename, &filetype)) {
        return NULL;
    }

    swprintf(fancy_filename, 256, L"%hs", filename);

    if (filetype != IS_IMG_JPG && filetype != IS_IMG_PNG && filetype != IS_IMG_BMP) {
        PyErr_SetString(PyExc_ValueError, "Invalid image filetype");
    }

    int ret;
    char *mem;
    INT image_id;

    ret = get_next_image(self, &mem, &image_id);
    if (ret) {
        /* Exception set, return */
        return NULL;
    }

    IMAGE_FILE_PARAMS ImageFileParams;
    ImageFileParams.pwchFileName = fancy_filename;
    ImageFileParams.nFileType = filetype;
    ImageFileParams.nQuality = 100;
    ImageFileParams.ppcImageMem = &mem;
    ImageFileParams.pnImageID = (UINT*) &image_id;
    ret = is_ImageFile(self->handle, IS_IMAGE_FILE_CMD_SAVE, (void*)&ImageFileParams, sizeof(ImageFileParams));
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        PyErr_Format(PyExc_IOError, "Failed to save image. ret = %d", ret);
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
        PyErr_SetString(PyExc_IOError, "Failed to unlock image memory.");
        return NULL;
    }

    return info;
}

static PyObject *ids_core_Camera_next(ids_core_Camera *self, PyObject *args, PyObject *kwds) {
    int ret;
    char *mem;
    INT image_id;

    ret = get_next_image(self, &mem, &image_id);
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
        return NULL;
    }
    
    ret = is_UnlockSeqBuf(self->handle, image_id, mem);
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        PyErr_SetString(PyExc_IOError, "Failed to unlock image memory.");
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
        PyErr_SetString(PyExc_NotImplementedError, "Unsupport color format for conversion to array.");
        return NULL;
    }

    return (PyObject*)matrix;
}

static PyObject *ids_core_Camera_save_tiff(ids_core_Camera *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"image", "filename", NULL};
    PyArrayObject* matrix;
    char *filename;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Os", kwlist, &matrix, &filename)) {
        PyErr_SetString(PyExc_TypeError, "Object must be nparray, filename must be string.");
        return NULL;
    }

    if (!PyArray_Check(matrix)) {
        PyErr_SetString(PyExc_TypeError, "nparray required");
        return NULL;
    }

    if (!PyArray_ISCONTIGUOUS(matrix)) {
        PyErr_SetString(PyExc_TypeError, "nparray must be contiguous.");
        return NULL;
    }

    int dng = 0;
    int samples_per_pixel = 1;
    int photometric = PHOTOMETRIC_CFA;

    switch (self->color) {
    case IS_CM_SENSOR_RAW8:
    case IS_CM_SENSOR_RAW12:
    case IS_CM_SENSOR_RAW16:
        dng = 1;
        samples_per_pixel = 1;
        photometric = PHOTOMETRIC_CFA;
        break;
    case IS_CM_MONO8:
    case IS_CM_MONO12:
    case IS_CM_MONO16:
        dng = 0;
        samples_per_pixel = 1;
        photometric = PHOTOMETRIC_MINISBLACK;
        break;
    case IS_CM_RGB8_PACKED:
        dng = 0;
        samples_per_pixel = 3;
        photometric = PHOTOMETRIC_RGB;
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "Unsupported color format for tiff conversion.");
        return NULL;
    }

    char *mem = PyArray_BYTES(matrix);

    TIFF *file = NULL;

    file = TIFFOpen(filename, "w");

    if (file == NULL) {
        PyErr_SetString(PyExc_IOError, "libtiff failed to open file for writing.");
        return NULL;
    }

    TIFFSetField(file, TIFFTAG_IMAGEWIDTH, self->width);
    TIFFSetField(file, TIFFTAG_IMAGELENGTH, self->height);
    TIFFSetField(file, TIFFTAG_UNIQUECAMERAMODEL, "IDS UI549xSE-C");

    TIFFSetField(file, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(file, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(file, TIFFTAG_SUBFILETYPE, 0);

    TIFFSetField(file, TIFFTAG_BITSPERSAMPLE, self->bitdepth/samples_per_pixel);
    TIFFSetField(file, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel);
    TIFFSetField(file, TIFFTAG_PHOTOMETRIC, photometric);

    /* If we are saving bayer data, this will be a DNG */
    if (dng) {
        short cfapatterndim[] = {2,2};
        char  cfapattern[] = {0,1,1,2}; /* RGGB */
        const float cam_xyz[9] = /* Placeholder! Need to computer real values */
        { 2.005,-0.771,-0.269,-0.752,1.688,0.064,-0.149,0.283,0.745 };

        TIFFSetField(file, TIFFTAG_CFAREPEATPATTERNDIM, cfapatterndim);
        TIFFSetField(file, TIFFTAG_CFAPATTERN, cfapattern);
        TIFFSetField(file, TIFFTAG_COLORMATRIX1, 9, cam_xyz);
        TIFFSetField(file, TIFFTAG_DNGVERSION, "\001\001\0\0");
        TIFFSetField(file, TIFFTAG_DNGBACKWARDVERSION, "\001\0\0\0");
    }

    for (int row = 0; row < self->height; row++) {
        if (TIFFWriteScanline(file, mem, row, 0) < 0) {
            TIFFClose(file);
            PyErr_SetString(PyExc_IOError, "libtiff failed to write row.");
            return NULL;
        }
        else {
            mem += self->width * self->bitdepth/8;
        }
    }

    TIFFWriteDirectory(file);
    TIFFClose(file);

    Py_INCREF(Py_True);
    return Py_True;
}

PyMethodDef ids_core_Camera_methods[] = {
    {"close", (PyCFunction) ids_core_Camera_close, METH_VARARGS, "close()\n\nCloses open camera"},
    {"start_continuous", (PyCFunction) ids_core_Camera_start_continuous, METH_VARARGS, "start_continuous()\n\nInitializes continuous image capture."},
    {"next_save", (PyCFunction) ids_core_Camera_next_save, METH_VARARGS | METH_KEYWORDS, "next_save(filename [, filetype=ids_core.FILETYPE_JPG]) -> metadata\n\nSaves next image in buffer and returns metadata from camera."},
    {"next", (PyCFunction) ids_core_Camera_next, METH_VARARGS, "next() -> image, metadata\n\nReturns next image in buffer as a numpy array and metadata from camera."},
    {"save_tiff", (PyCFunction) ids_core_Camera_save_tiff, METH_VARARGS, "save_tiff(image, filename)\n\nSave a captured image as a tiff.  Image must be a numpy array,\nand is expected to be one returned from next().\nIf the color mode is currently bayer, the image will be saved as a RAW DNG\n, a subset of TIFF.  Otherwise, it will be saved as a standard TIFF.\n Non bayer images must be ids_core.COLOR_RGB8 or ids_core.COLOR_MONO*."},
    {NULL}
};
