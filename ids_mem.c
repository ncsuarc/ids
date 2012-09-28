#include <Python.h>
#include <ueye.h>

#include "ids.h"

static int add_mem(ids_Camera *self, char *mem, int id);

PyObject *alloc_ids_mem(ids_Camera *self, int width, int height, uint32_t num) {
    char *mem;
    int id;

    for (int i = 0; i < num; i++) {
        int ret;
        ret = is_AllocImageMem(self->handle, width, height, self->bitdepth, &mem, &id);
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
