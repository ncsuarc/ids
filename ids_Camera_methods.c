#include <Python.h>
#include <structmember.h>
#include <ueye.h>

#include "ids.h"

static PyObject *ids_Camera_close(ids_Camera *self, PyObject *args, PyObject *kwds);
static PyObject *ids_Camera_start_queue(ids_Camera *self, PyObject *args, PyObject *kwds);

static int add_mem(ids_Camera *self, char *mem, int id);

PyMethodDef ids_Camera_methods[] = {
    {"close", (PyCFunction) ids_Camera_close, METH_VARARGS, "Closes open camera"},
    {"start_queue", (PyCFunction) ids_Camera_start_queue, METH_VARARGS, "Initializes image buffer queue mode."},
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

PyObject *alloc_ids_mem(ids_Camera *self, int width, int height, int bitdepth, uint32_t num) {
    char *mem;
    int id;

    for (int i = 0; i < num; i++) {
        int ret;
        ret = is_AllocImageMem(self->handle, width, height, bitdepth, &mem, &id);
        if (ret != IS_SUCCESS) {
            free_all_ids_mem(self);
            PyErr_SetString(PyExc_MemoryError, "Unable to allocate image memory.");
            return NULL;
        }

        ret = is_AddToSequence(self->handle, mem, id);
        if (ret != IS_SUCCESS) {
            is_FreeImageMem(self->handle, mem, id);
            free_all_ids_mem(self);
            PyErr_SetString(PyExc_MemoryError, "Unable to allocate image memory.");
            return NULL;
        }

        if (add_mem(self, mem, id) != 0) {
            free_all_ids_mem(self);
            PyErr_SetString(PyExc_MemoryError, "Unable to allocate image memory.");
            return NULL;
        }
    }

    Py_INCREF(Py_True);
    return Py_True;
}

void free_all_ids_mem(ids_Camera *self) {
    struct allocated_mem *prev = self->mem;
    struct allocated_mem *curr = self->mem;

    while (curr) {
        prev = curr;
        curr = curr->next;
        is_FreeImageMem(self->handle, prev->mem, prev->id);
        free(prev);
    }
}

static int add_mem(ids_Camera *self, char *mem, int id) {
    struct allocated_mem *node = malloc(sizeof(struct allocated_mem));
    if (node == NULL) {
        return -1;
    }

    node->mem = mem;
    node->id  = id;
    node->next= NULL;

    if (self->mem) {
        struct allocated_mem *curr = self->mem;
        struct allocated_mem *prev = self->mem;

        /* Could be faster if tail was saved */
        while (curr) {
            prev = curr;
            curr = curr->next;
        }

        prev->next = node;
    }
    else {
        self->mem = node;
    }

    return 0;
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
