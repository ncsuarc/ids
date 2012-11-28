#include <Python.h>
#include <ueye.h>

#include "ids.h"

void add_constants(PyObject *m) {
    /* Color */
    PyModule_AddIntConstant(m, "COLOR_BAYER_16", IS_CM_BAYER_RG16);
    PyModule_AddIntConstant(m, "COLOR_BAYER_12", IS_CM_BAYER_RG12);
    PyModule_AddIntConstant(m, "COLOR_BAYER_8", IS_CM_BAYER_RG8);
    PyModule_AddIntConstant(m, "COLOR_MONO_16", IS_CM_MONO16);
    PyModule_AddIntConstant(m, "COLOR_MONO_12", IS_CM_MONO12);
    PyModule_AddIntConstant(m, "COLOR_MONO_8", IS_CM_MONO8);
    PyModule_AddIntConstant(m, "COLOR_RGB10V2", IS_CM_RGB10V2_PACKED);
    PyModule_AddIntConstant(m, "COLOR_RGBA8", IS_CM_RGBA8_PACKED);
    PyModule_AddIntConstant(m, "COLOR_RGBY8", IS_CM_RGBY8_PACKED);
    PyModule_AddIntConstant(m, "COLOR_RGB8", IS_CM_RGB8_PACKED);
    PyModule_AddIntConstant(m, "COLOR_BGR10V2", IS_CM_BGR10V2_PACKED);
    PyModule_AddIntConstant(m, "COLOR_BGRA8", IS_CM_BGRA8_PACKED);
    PyModule_AddIntConstant(m, "COLOR_BGRY8", IS_CM_BGRY8_PACKED);
    PyModule_AddIntConstant(m, "COLOR_BGR8", IS_CM_BGR8_PACKED);
    PyModule_AddIntConstant(m, "COLOR_BGR565", IS_CM_BGR565_PACKED);
    PyModule_AddIntConstant(m, "COLOR_BGR555", IS_CM_BGR555_PACKED);
    PyModule_AddIntConstant(m, "COLOR_UYVY", IS_CM_UYVY_PACKED);
    PyModule_AddIntConstant(m, "COLOR_CBYCRY", IS_CM_CBYCRY_PACKED);

    PyModule_AddIntConstant(m, "FILETYPE_JPG", IS_IMG_JPG);
    PyModule_AddIntConstant(m, "FILETYPE_BMP", IS_IMG_BMP);
    PyModule_AddIntConstant(m, "FILETYPE_PNG", IS_IMG_PNG);
}
