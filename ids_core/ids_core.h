#ifndef IDS_H_INCLUDED
#define IDS_H_INCLUDED

#include <sys/queue.h>

/* Module methods */
extern PyMethodDef ids_coreMethods[];

/* IDS data structures */
struct allocated_mem {
    char *mem;
    int id;
    LIST_ENTRY(allocated_mem) list;
};

/* Camera class */
typedef struct {
    PyObject_HEAD;
    /* Externally available elements (in Python) */
    HIDS                    handle;
    uint32_t                width;
    uint32_t                height;
    /* Internal structures */
    LIST_HEAD(allocated_mem_head, allocated_mem) mem_list;
    int                     bitdepth;
    int                     color;
    int                     autofeatures;
    int                     ready;
} ids_core_Camera;   /* Be sure to update ids_core_Camera_members with new entries */

enum ready {
    NOT_READY,
    CONNECTED,
    ALLOCATED_MEM,
    READY,
};

void add_constants(PyObject *m);
PyObject *set_color_mode(ids_core_Camera *self, int color);
int color_to_bitdepth(int color);
PyObject *alloc_ids_core_mem(ids_core_Camera *self, int width, int height, uint32_t num);
void free_all_ids_core_mem(ids_core_Camera *self);
PyObject *image_info(ids_core_Camera *self, int image_id);

extern PyTypeObject ids_core_CameraType;
extern PyMethodDef ids_core_Camera_methods[];
extern PyGetSetDef ids_core_Camera_getseters[];

#endif
