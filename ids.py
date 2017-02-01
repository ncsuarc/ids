# Copyright (c) 2012, 2013, North Carolina State University Aerial Robotics Club
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the North Carolina State University Aerial Robotics Club
#       nor the names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import logging
import ids_core
from ids_core import number_cameras, camera_list

class Camera(ids_core.Camera):
    """
    IDS Camera object

    Provides access to, and control of, IDS machine vision cameras.  This
    class provides as attributes many of the camera settings.  It handles
    image capture internally, and provides methods to get images from the
    camera.

    Arguments:
        logger (optional): logging object to use for log output.
        nummem (optional): Number of memory locations to allocate for storing
            images.
        handle (optional): Image handle to connect to.
        color (optional): Default color mode.  One of ids_core.COLOR_*
            constants.
    """

    def __init__(self, *args, **kwargs):
        logging.basicConfig()   # Configure logging, if it isn't already
        self.logger = kwargs.pop('logger', None) or logging.getLogger(__name__)

        self.nummem = kwargs.pop('nummem', 5)

        super(Camera, self).__init__(*args, **kwargs)

        self._allocate_memory()

    def _allocate_memory(self):
        for i in range(self.nummem):
            self.alloc()

    def _check_capture_status(self):
        """
        Check camera capture status, logging any warnings or errors detected.
        """
        messages = {'total': None,
                    'no_destination_mem': "Out of memory locations for images",
                    'conversion_failed': "Image conversion failed",
                    'image_locked': "Destination image memory locked",
                    'no_driver_mem': "Out of internal memory",
                    'device_not_available': "Camera not available",
                    'usb_transfer_failed': "USB transfer failed",
                    'device_timeout': "Camera timed out capturing image",
                    'eth_buffer_overrun': "Camera internal buffers overrun",
                    'eth_missed_images': "Image missed due to lack of bandwidth or processing power",
                   }

        status = self.capture_status()

        self.logger.debug("%d total capture status warnings or errors" % status['total'])

        for key, value in status.items():
            if value and messages[key]:
                self.logger.warning("%s (%d instances)" % (messages[key], value))

    def next(self):
        """
        Get the next available image from the camera.

        Waits for the next image to be available from the camera, and returns
        it as a numpy array.  Blocks until image is available, or timeout is
        reached.

        Returns:
            (image, metadata) tuple, where image is a numpy array containing
            the image in the camera color format, and metadata is a dictionary
            with image metadata.  Timestamp is provided as a datetime object in
            UTC.

        Raises:
            IDSTimeoutError: An image was not available within the timeout.
            IDSError: An unknown error occured in the uEye SDK.
            NotImplementedError: The current color format cannot be converted
                to a numpy array.
        """
        while True:
            try:
                return super(Camera, self).next()
            except ids_core.IDSCaptureStatus:
                self._check_capture_status()

    def next_save(self, *args, **kwargs):
        """
        Save the next available image to a file.

        This function behaves similarly to Camera.next(), however instead
        of returning the image, it uses the IDS functions to save the image
        to a file.  The appropriate color mode for the filetype should be
        used (eg. BGR for JPEG).

        Arguments:
            filename: File to save image to.
            filetype (optional): Filetype to save as, defaults to
                ids_core.FILETYPE_JPG.
            quality (optional): Image quality for JPEG and PNG,
                with 100 as maximum quality.

        Returns:
            Dictonary containing image metadata.  Timestamp is provided as
            a datetime object in UTC.

        Raises:
            ValueError: An invalid filetype was passed in.
            IDSTimeoutError: An image was not available within the timeout.
            IDSError: An unknown error occured in the uEye SDK.
        """
        while True:
            try:
                return super(Camera, self).next_save(*args, **kwargs)
            except ids_core.IDSCaptureStatus:
                self._check_capture_status()

    # Override color_mode to reallocate memory when changed
    @property
    def color_mode(self):
        """
        Color mode used for capturing images.

        One of ids_core.COLOR_* constants.  Image memory will be reallocated
        after changing color mode, as the memory requirements may change.

        Raises:
            IOError: Color mode cannot be changed while capturing images.
        """
        return ids_core.Camera.color_mode.__get__(self)

    @color_mode.setter
    def color_mode(self, val):
        if self.continuous_capture:
            raise IOError("Color cannot be changed while capturing images")

        ids_core.Camera.color_mode.__set__(self, val)

        # Free all memory and reallocate, as bitdepth may have changed
        self.free_all()
        self._allocate_memory()

    # Override aoi to reallocate memory when changed
    @property
    def aoi(self):
        """
        AOI settings used for capturing images.
        Reset memory as color_mode does.
        """
        return ids_core.Camera.aoi.__get__(self)

    @aoi.setter
    def aoi(self, val):
        if self.continuous_capture:
            raise IOError("IO Error")

        ids_core.Camera.aoi.__set__(self, val)

        # Free all memory and reallocate, as bitdepth may have changed
        self.free_all()
        self._allocate_memory()
