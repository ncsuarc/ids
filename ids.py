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

import ids_core
import logging

class Camera(ids_core.Camera):
    """
    IDS Camera object

    Arguments:
        nummem: Number of memory locations to allocate for storing images
    """

    def __init__(self, nummem=5, color=ids_core.COLOR_BGRA8, logger=None):
        logging.basicConfig()   # Configure logging, if it isn't already
        self.logger = logger or logging.getLogger(__name__)

        self.nummem = nummem

        super(Camera, self).__init__(color=color)

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

        self.logger.info("%d total capture status warnings or errors" % status['total'])

        for key, value in status.items():
            if value and messages[key]:
                self.logger.warning("%s (%d instances)" % (messages[key], value))

    def next(self):
        while True:
            try:
                return super(Camera, self).next()
            except ids_core.IDSCaptureStatus:
                self._check_capture_status()

    def next_save(self, filename, filetype=ids_core.FILETYPE_JPG):
        while True:
            try:
                return super(Camera, self).next_save(filename, filetype=filetype)
            except ids_core.IDSCaptureStatus:
                self._check_capture_status()

    # Override color_mode to reallocate memory when changed
    @property
    def color_mode(self):
        return ids_core.Camera.color_mode.__get__(self)

    @color_mode.setter
    def color_mode(self, val):
        if self.continuous_capture:
            raise IOError("Color cannot be changed while capturing images")

        ids_core.Camera.color_mode.__set__(self, val)

        # Free all memory and reallocate, as bitdepth may have changed
        self.free_all()
        self._allocate_memory()

def number_cameras():
    return ids_core.number_cameras()

def camera_list():
    return ids_core.camera_list()
