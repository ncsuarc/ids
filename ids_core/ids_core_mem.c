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

#include <sys/queue.h>
#include <Python.h>
#include <ueye.h>

#include "ids_core.h"

static int add_mem(ids_core_Camera *self, char *mem, int id);

PyObject *alloc_ids_core_mem(ids_core_Camera *self, int width, int height, uint32_t num) {
    char *mem;
    int id;

    for (int i = 0; i < num; i++) {
        int ret;
        ret = is_AllocImageMem(self->handle, width, height, self->bitdepth, &mem, &id);
        if (ret != IS_SUCCESS) {
            goto err_free_all;
        }

        ret = is_AddToSequence(self->handle, mem, id);
        if (ret != IS_SUCCESS) {
            goto err_free_current;
        }

        if (add_mem(self, mem, id) != 0) {
            goto err_free_current;
        }
    }

    Py_INCREF(Py_True);
    return Py_True;

err_free_current:
    is_FreeImageMem(self->handle, mem, id);
err_free_all:
    free_all_ids_core_mem(self);
    PyErr_SetString(PyExc_MemoryError, "Unable to allocate image memory.");
    return NULL;
}

void free_all_ids_core_mem(ids_core_Camera *self) {
    is_ClearSequence(self->handle);

    while (!LIST_EMPTY(&self->mem_list)) {
        struct allocated_mem *mem = LIST_FIRST(&self->mem_list);
        is_FreeImageMem(self->handle, mem->mem, mem->id);
        LIST_REMOVE(mem, list);
        free(mem);
    }
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
