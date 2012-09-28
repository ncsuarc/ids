#ifndef IDS_H_INCLUDED
#define IDS_H_INCLUDED

/* Module methods */
extern PyMethodDef idsMethods[];

/* IDS data structures */
struct allocated_mem {
    char *mem;
    int id;
    struct allocated_mem *next;
};

/* Camera class */
typedef struct {
    PyObject_HEAD;
    /* Externally available elements (in Python) */
    HIDS                    handle;
    uint32_t                width;
    uint32_t                height;
    /* Internal structures */
    struct allocated_mem    *mem;
    int                     bitdepth;
} ids_Camera;   /* Be sure to update ids_Camera_members with new entries */

void add_constants(PyObject *m);
PyObject *set_color_mode(ids_Camera *self, int color);
int color_to_bitdepth(int color);
PyObject *alloc_ids_mem(ids_Camera *self, int width, int height, uint32_t num);
void free_all_ids_mem(ids_Camera *self);

extern PyTypeObject ids_CameraType;
extern PyMethodDef ids_Camera_methods[];
extern PyGetSetDef ids_Camera_getseters[];

#endif
