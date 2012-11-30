#include <Python.h>
#include <structmember.h>
#include <ueye.h>
#include <wchar.h>
#include <stdio.h>

#define PY_ARRAY_UNIQUE_SYMBOL  ids_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "tiffio.h"

#include "ids.h"

#define IMG_TIMEOUT 5000

static PyObject *ids_Camera_close(ids_Camera *self, PyObject *args, PyObject *kwds);
static PyObject *ids_Camera_start_continuous(ids_Camera *self, PyObject *args, PyObject *kwds);
static PyObject *ids_Camera_next_save(ids_Camera *self, PyObject *args, PyObject *kwds);
static PyObject *ids_Camera_next(ids_Camera *self, PyObject *args, PyObject *kwds);
static PyObject *ids_Camera_save_dng(ids_Camera *self, PyObject *args, PyObject *kwds);

static PyObject *create_matrix(ids_Camera *self, char *mem);

PyMethodDef ids_Camera_methods[] = {
    {"close", (PyCFunction) ids_Camera_close, METH_VARARGS, "close()\n\nCloses open camera"},
    {"start_continuous", (PyCFunction) ids_Camera_start_continuous, METH_VARARGS, "start_continuous()\n\nInitializes continuous image capture."},
    {"next_save", (PyCFunction) ids_Camera_next_save, METH_VARARGS | METH_KEYWORDS, "next_save(filename [, filetype=ids.FILETYPE_JPG]) -> metadata\n\nSaves next image in buffer and returns metadata from camera."},
    {"next", (PyCFunction) ids_Camera_next, METH_VARARGS, "next() -> image, metadata\n\nReturns next image in buffer as a numpy array and metadata from camera."},
    {"save_dng", (PyCFunction) ids_Camera_save_dng, METH_VARARGS, "save_dng(image, filename)\n\nSave a captured image as a DNG.  Image must be a numpy array,\nand is expected to be one returned from next().\nCurrently only supports bayer images."},
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

static PyObject *ids_Camera_start_continuous(ids_Camera *self, PyObject *args, PyObject *kwds) {
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

static PyObject *ids_Camera_next_save(ids_Camera *self, PyObject *args, PyObject *kwds) {
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

retry:
    ret = is_WaitForNextImage(self->handle, IMG_TIMEOUT, &mem, &image_id); 

    switch (ret) {
    case IS_SUCCESS:
        break;
    case IS_TIMED_OUT:
        PyErr_SetString(PyExc_IOError, "Capture timed out on WaitForNextImage.");
        return NULL;
    case IS_CAPTURE_STATUS: {
        UEYE_CAPTURE_STATUS_INFO capture_status;
        int r = is_CaptureStatus(self->handle, IS_CAPTURE_STATUS_INFO_CMD_GET, (void *) &capture_status, sizeof(capture_status));
        if (r == IS_SUCCESS) {
            if (capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_API_NO_DEST_MEM]) {
                printf("Warning: out of memory locations for images, retrying. ");
            }
            else {
                printf("Warning: Capture Status, total: %d. ", capture_status.dwCapStatusCnt_Total);
                printf("NO_DEST_MEM: %d, CONVT_FAILED: %d, IMAGE_LOCKED: %d, OUT_OF_BUFFERS: %d, DEVICE_NOT_RDY: %d, TRANSF_FAILED: %d, TIMEOUT: %d, BUFFER_OVRRUN: %d, MISSED_IMGS: %d. ", capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_API_NO_DEST_MEM], capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_API_CONVERSION_FAILED], capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_API_IMAGE_LOCKED], capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_DRV_OUT_OF_BUFFERS], capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_DRV_DEVICE_NOT_READY], capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_USB_TRANSFER_FAILED], capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_DEV_TIMEOUT], capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_ETH_BUFFER_OVERRUN], capture_status.adwCapStatusCnt_Detail[IS_CAP_STATUS_ETH_MISSED_IMAGES]);
            }
        }
        else {
            printf("Warning: Capture Status failed. ");
        }

        goto retry;
    }
    default:
        PyErr_Format(PyExc_IOError,  "Failed to capture image on WaitForNextImage.  ret = %d", ret);
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

static PyObject *ids_Camera_next(ids_Camera *self, PyObject *args, PyObject *kwds) {
    int ret;
    char *mem;
    INT image_id;

    ret = is_WaitForNextImage(self->handle, IMG_TIMEOUT, &mem, &image_id); 

    switch (ret) {
    case IS_SUCCESS:
        break;
    case IS_TIMED_OUT:
        PyErr_SetString(PyExc_IOError, "Capture timed out.");
        return NULL;
    default:
        PyErr_SetString(PyExc_IOError, "Failed to capture image.");
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

    return Py_BuildValue("(OO)", image, info);
}

static PyObject *create_matrix(ids_Camera *self, char *mem) {
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

static PyObject *ids_Camera_save_dng(ids_Camera *self, PyObject *args, PyObject *kwds) {
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

    char *mem = PyArray_BYTES(matrix);

    short cfapatterndim[] = {2,2};
    char  cfapattern[] = {0,1,1,2};

    /* Not for our camera! */
    static const float cam_xyz[] =
    { 2.005,-0.771,-0.269, -0.752,1.688,0.064, -0.149,0.283,0.745 };

    TIFF *file = NULL;

    file = TIFFOpen(filename, "w");

    if (file == NULL) {
        PyErr_SetString(PyExc_IOError, "libtiff failed to open file for writing.");
        return NULL;
    }

    TIFFSetField(file, TIFFTAG_IMAGEWIDTH, self->width);
    TIFFSetField(file, TIFFTAG_IMAGELENGTH, self->height);
    TIFFSetField(file, TIFFTAG_BITSPERSAMPLE, self->bitdepth);
    TIFFSetField(file, TIFFTAG_CFAREPEATPATTERNDIM, cfapatterndim);
    TIFFSetField(file, TIFFTAG_CFAPATTERN, cfapattern);
    TIFFSetField(file, TIFFTAG_COLORMATRIX1, 9, cam_xyz);
    TIFFSetField(file, TIFFTAG_UNIQUECAMERAMODEL, "IDS UI549xSE-C");

    TIFFSetField(file, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_CFA);
    TIFFSetField(file, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(file, TIFFTAG_SUBFILETYPE, 0);
    TIFFSetField(file, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(file, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(file, TIFFTAG_DNGVERSION, "\001\001\0\0");
    TIFFSetField(file, TIFFTAG_DNGBACKWARDVERSION, "\001\0\0\0");

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
