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
    PyObject                *blah;
    /* Internal structures */
    struct allocated_mem    *mem;
} ids_Camera;   /* Be sure to update ids_Camera_members with new entries */

extern PyTypeObject ids_CameraType;
extern PyMethodDef ids_Camera_methods[];
extern PyGetSetDef ids_Camera_getseters[];

#endif
