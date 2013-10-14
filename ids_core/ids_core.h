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
    /*
     * Lazily initialized name, ensure it is initialized before using!
     * ids_core_Camera_getname() will initialize it.
     */
    PyObject                *name;
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
    READY,
};

void add_constants(PyObject *m);
PyObject *set_color_mode(ids_core_Camera *self, int color);
int color_to_bitdepth(int color);
PyObject *image_info(ids_core_Camera *self, int image_id);

/* Exported method for freeing all memory */
PyObject *ids_core_Camera_free_all(ids_core_Camera *self, PyObject *args, PyObject *kwds);

/*
 * Exported method to initialize camera name
 *
 * Initializes self->name and returns a new reference to the name
 */
PyObject *ids_core_Camera_getname(ids_core_Camera *self, void *closure);

/*
 * Raise an exception for an unknown IDS error code.
 *
 * Attempts to lookup error message with is_GetError(),
 * then raises an IDSError with the error code and message.
 *
 * @param self  Camera object
 * @param error Error code returned from IDS function
 */
void raise_general_error(ids_core_Camera *self, int error);

extern PyTypeObject ids_core_CameraType;
extern PyMethodDef ids_core_Camera_methods[];
extern PyGetSetDef ids_core_Camera_getseters[];

/* IDS Exceptions */
extern PyObject *IDSError;
extern PyObject *IDSTimeoutError;
extern PyObject *IDSCaptureStatus;

#endif
