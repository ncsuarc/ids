#ifndef IDS_H_INCLUDED
#define IDS_H_INCLUDED

/* Module methods */
extern PyMethodDef idsMethods[];

/* Camera class */
typedef struct {
    PyObject_HEAD;
    HIDS handle;
    /* Type fields here */
} ids_Camera;   /* Be sure to update ids_Camera_members with new entries */

extern PyTypeObject ids_CameraType;
extern PyMethodDef ids_Camera_methods[];

#endif
