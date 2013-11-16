IDS Python Module
==================

A module for interfacing with IDS Imaging machine vision cameras.
This module wraps the IDS uEye SDK, providing a convient Python interface
to much of the SDK.  The ids.Camera object provides attributes for easily
controlling camera settings and capturing images.

## Requirements

The ids module is written in Python and C, using the Python C API, and supports
both Python 2 and Python 3.  It has been tested using Python 2.7 and Python 3.2.

The module has only been tested on Linux, and likely does not support Windows.

Build requirements:

* IDS uEye SDK
    * SDK version 4.20 or higher is supported
    * The SDK can be acquired from the
        [IDS website](http://en.ids-imaging.com/download-ueye.html)
* libtiff-dev
    * Tested on libtiff-4.0.3
    * A version newer than 4.0.3 or a patched version is required for DNG
        support.  See "DNG Support" below.

## Building and installing

Once all dependencies are met, it is simple to build and install the module:

    $ python setup.py install

Or, for Python 3:

    $ python3 setup.py install

Of course, if the installation location requires root permissions, `sudo` may
be necessary.

### DNG Support

The module supports saving raw bayer images in the DNG file format, which is a
superset of the TIFF file format.  In order to describe the bayer format,
several TIFF tags are required that are not included in libtiff-4.0.3.  In
order to maintain compatibility with most installations of libtiff, DNG
support is disabled by default.

A patch for libtiff-4.0.3 is included in `libtiff-4.0.3.patch` to add support
for these tags.  This patch has been merged into the libtiff trunk, and should
be included in the release following 4.0.3.

To patch libtiff, first grab the latest version from the libtiff
[website](http://www.remotesensing.org/libtiff/).  If this is > 4.0.3,
the DNG patch should be included, and you can simply build and install this
version.

Otherwise, first patch libtiff, then configure, build, and install it.

    $ cd /path/to/libtiff
    $ patch -p1 < /path/to/ids/libtiff-4.0.3.patch

Once the patched libtiff in installed, the ids module can be rebuilt it with
DNG support.

    $ python setup.py clean --all
    $ python setup.py build_ext --define=DNG_SUPPORT
    $ python setup.py install

## Usage

The ids module makes it easy to control a camera.  Just initialize the Camera
object and set the attributes of interest, then start image capture.

    >>> import ids
    >>> cam = ids.Camera()
    >>> cam.color_mode = ids.ids_core.COLOR_RGB8    # Get images in RGB format
    >>> cam.exposure = 5                            # Set initial exposure to 5ms
    >>> cam.auto_exposure = True
    >>> cam.continuous_capture = True               # Start image capture

You can get images from the camera as a Numpy array

    >>> img, meta = cam.next()                      # Get image as a Numpy array

A built in method will save the images as TIFFs.  This is particularly useful
for saving raw Bayer data as a DNG.

    >>> cam.save_tiff(img, "image.tif")

PIL also provides a wide range of formats to save in.

    >>> Import Image
    >>> pil_img = Image.fromarray(img)
    >>> pil_img.save("pil.jpg", quality=95)

OpenCV also allows you to save images.

    >>> import cv2
    >>> bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    >>> cv2.imwrite('cv2.jpg', bgr_img) # cv2.imwrite takes a BGR image

Alternatively, the IDS uEye SDK provides a function for saving images.  The
camera must be in a BGR mode for JPEGs.

    >>> cam.continuous_capture = False  # Stop capture to change color mode
    >>> cam.color_mode = ids.ids_core.COLOR_BGR8
    >>> cam.continuous_capture = True
    >>> meta = cam.next_save("ids.jpg")

### Color Modes

It is important to take the color mode images are captured in into account,
particularly for saving images.  Different formats and image libraries have
different expectations of the format images are stored in, so capturing images
in the wrong format may result in images with swapped color channels.

The color mode constants are provided in the ids module as
`ids.ids_core.COLOR_*`.  The color mode can be passed into the `color`
keyword argument of the `ids.Camera` object, or set while image capture
is not running with the `ids.Camera.color_mode` attribute.

For example, standard TIFFs are saved in an RGB format, so RGB format arrays
should be passed into `Camera.save_tiff()`, unless raw bayer images are being
saved.

The IDS uEye SDK imaging saving functions, used by `Camera.next_save()` copy
image data directly, so the appropriate mode for the format must be used.
JPEG images are stored in a BGR format, while BMP and PNG are RGB.

Different libraries expect images in different format.  PIL expects images to
be in an RGB format.  OpenCV supports many formats, but generally expects BGR
data.  However, `cv2.cvtColor()` supports conversions between many formats.

## Additional information

The IDS uEye [documentation](http://en.ids-imaging.com/manuals/uEye_SDK/EN/uEye_Manual/index.html)
is a useful source of information about the features provided by the IDS
cameras and uEye SDK, and by extension, this module.
