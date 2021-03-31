# Copyright 2021 Toyota Research Institute.  All rights reserved.

import numpy as np
from OpenGL.GL import \
    glGenBuffers, glBindBuffer, glBufferData, glBufferSubData, GL_ARRAY_BUFFER, GL_STATIC_DRAW

from display.camviz.utils.utils import numpyf, cmapJET
from packnet_sfm.utils.types import is_tuple, is_list, is_tensor


class Buffer:
    """
    Initialize a data buffer

    Parameters
    ----------
    data : np.array [N,D] or tuple (n,d)
        Data to be added to the buffer
        If it's a tuple, create a data buffer of that size
    dtype : numpy type (e.g. np.float32)
        Numpy data type
    gltype : OpenGL type (e.g. GL_FLOAT32)
        OpenGL data type
    """
    def __init__(self, data, dtype, gltype):
        # Initialize buffer ID and max size
        self.id, self.max = glGenBuffers(1), 0
        # Store data types
        self.dtype, self.gltype = dtype, gltype
        if is_tuple(data):
            # If data is a tuple, store dimensions
            data, (self.n, self.d) = None, data
        else:
            # Process data and store dimensions
            data = self.process(data)
            self.n, self.d = data.shape[:2]
        # If size is larger than available, recreate buffer
        if self.n > self.max:
            self._create(data)

    @property
    def size(self):
        """Get buffer size"""
        return self.n * self.d * np.dtype(self.dtype).itemsize

    def process(self, data):
        """
        Process data buffer to get relevant information

        Parameters
        ----------
        data : list or np.array or torch.Tensor
            Data to be processed

        Returns
        -------
        data : np.array
            Processed data
        """
        # If it's a list
        if is_list(data):
            data = numpyf(data)
        # If it's a tensor
        if is_tensor(data):
            data = data.detach().cpu().numpy()
        # If it's not the correct type, convert
        if data.dtype != self.dtype:
            data = data.astype(self.dtype)
        # Expand if necessary
        if len(data.shape) == 1:
            data = np.expand_dims(data, 1)
        # Return data
        return data

    def _create(self, data):
        """Create a new data buffer"""
        self.max = self.n
        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        glBufferData(GL_ARRAY_BUFFER, self.size, data, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def update(self, data):
        """Update data buffer"""
        # Process data
        data = self.process(data)
        # Get dimensions or initialize as zero
        self.n = 0 if data.size == 0 else data.shape[0]
        # If dimensions are larger than available, recreate
        if self.n > self.max:
            self._create(data)
        # Otherwise
        else:
            # Bind buffer and copy data
            glBindBuffer(GL_ARRAY_BUFFER, self.id)
            glBufferSubData(GL_ARRAY_BUFFER, 0, self.size, data.astype(self.dtype))
            glBindBuffer(GL_ARRAY_BUFFER, 0)

    def clear(self):
        """Clear buffer"""
        self.n = 0

    def updateJET(self, data):
        """Update buffer using a JET colormap"""
        self.update(cmapJET(data))