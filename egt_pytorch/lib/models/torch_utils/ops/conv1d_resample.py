# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""1D convolution with optional up/downsampling."""

import torch

from .. import misc
from . import conv1d_gradfix
from . import upfirdn1d
from .upfirdn1d import _parse_padding
from .upfirdn1d import _get_filter_size

#----------------------------------------------------------------------------

def _get_weight_shape(w):
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        shape = [int(sz) for sz in w.shape]
    misc.assert_shape(w, shape)
    return shape

#----------------------------------------------------------------------------

def _conv1d_wrapper(x, w, stride=1, padding=0, groups=1, transpose=False, flip_weight=True):
    """Wrapper for the underlying `conv1d()` and `conv_transpose1d()` implementations.
    """
    out_channels, in_channels_per_group, kd = _get_weight_shape(w)

    # Flip weight if requested.
    if not flip_weight: # conv1d() actually performs correlation (flip_weight=True) not convolution (flip_weight=False).
        w = w.flip([2, 3])

    # Workaround performance pitfall in cuDNN 8.0.5, triggered when using
    # 1x1 kernel + memory_format=channels_last + less than 64 channels.
    if kd == 1 and stride == 1 and padding in [0, [0, 0]] and not transpose:
        if x.stride()[1] == 1 and min(out_channels, in_channels_per_group) < 64:
            if out_channels <= 4 and groups == 1:
                in_shape = x.shape
                x = w.squeeze(2) @ x.reshape([in_shape[0], in_channels_per_group, -1])
                x = x.reshape([in_shape[0], out_channels, in_shape[2]])
            else:
                x = x.to(memory_format=torch.contiguous_format)
                w = w.to(memory_format=torch.contiguous_format)
                x = conv1d_gradfix.conv1d(x, w, groups=groups)
            return x.to(memory_format=torch.channels_last)

    # Otherwise => execute using conv1d_gradfix.
    op = conv1d_gradfix.conv_transpose1d if transpose else conv1d_gradfix.conv1d
    return op(x, w, stride=stride, padding=padding, groups=groups)

#----------------------------------------------------------------------------

@misc.profiled_function
def conv1d_resample(x, w, f=None, up=1, down=1, padding=0, groups=1, flip_weight=True, flip_filter=False):
    r"""1D convolution with optional up/downsampling.

    Padding is performed only once at the beginning, not between the operations.

    Args:
        x:              Input tensor of shape
                        `[batch_size, in_channels, in_dimension]`.
        w:              Weight tensor of shape
                        `[out_channels, in_channels//groups, kernel_dimension]`.
        f:              Low-pass filter for up/downsampling. Must be prepared beforehand by
                        calling upfirdn1d.setup_filter(). None = identity (default).
        up:             Integer upsampling factor (default: 1).
        down:           Integer downsampling factor (default: 1).
        padding:        Padding with respect to the upsampled image. Can be a single number
                        or a list/tuple `[x, y]` or `[x_before, x_after]`
                        (default: 0).
        groups:         Split input channels into N groups (default: 1).
        flip_weight:    False = convolution, True = correlation (default: True).
        flip_filter:    False = convolution, True = correlation (default: False).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_dimension]`.
    """
    # Validate arguments.
    assert isinstance(x, torch.Tensor) and (x.ndim == 3)
    assert isinstance(w, torch.Tensor) and (w.ndim == 3) and (w.dtype == x.dtype)
    assert f is None or (isinstance(f, torch.Tensor) and f.ndim in [1, 2] and f.dtype == torch.float32)
    assert isinstance(up, int) and (up >= 1)
    assert isinstance(down, int) and (down >= 1)
    assert isinstance(groups, int) and (groups >= 1)
    out_channels, in_channels_per_group, kd = _get_weight_shape(w)
    fd = _get_filter_size(f)
    px0, px1 = _parse_padding(padding)

    # Adjust padding to account for up/downsampling.
    if up > 1:
        px0 += (fd + up - 1) // 2
        px1 += (fd - up) // 2
    if down > 1:
        px0 += (fd - down + 1) // 2
        px1 += (fd - down) // 2

    # Fast path: 1x1 convolution with downsampling only => downsample first, then convolve.
    if kd == 1 and (down > 1 and up == 1):
        x = upfirdn1d.upfirdn1d(x=x, f=f, down=down, padding=[px0,px1], flip_filter=flip_filter)
        x = _conv1d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        return x

    # Fast path: 1x1 convolution with upsampling only => convolve first, then upsample.
    if kd == 1 and (up > 1 and down == 1):
        x = _conv1d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        x = upfirdn1d.upfirdn1d(x=x, f=f, up=up, padding=[px0,px1], gain=up**2, flip_filter=flip_filter)
        return x

    # Fast path: downsampling only => use strided convolution.
    if down > 1 and up == 1:
        x = upfirdn1d.upfirdn1d(x=x, f=f, padding=[px0,px1], flip_filter=flip_filter)
        x = _conv1d_wrapper(x=x, w=w, stride=down, groups=groups, flip_weight=flip_weight)
        return x

    # Fast path: upsampling with optional downsampling => use transpose strided convolution.
    if up > 1:
        if groups == 1:
            w = w.transpose(0, 1)
        else:
            w = w.reshape(groups, out_channels // groups, in_channels_per_group, kd)
            w = w.transpose(1, 2)
            w = w.reshape(groups * in_channels_per_group, out_channels // groups, kd)
        px0 -= kd - 1
        px1 -= kd - up
        pxt = max(min(-px0, -px1), 0)
        x = _conv1d_wrapper(x=x, w=w, stride=up, padding=[pxt], groups=groups, transpose=True, flip_weight=(not flip_weight))
        x = upfirdn1d.upfirdn1d(x=x, f=f, padding=[px0+pxt,px1+pxt], gain=up**2, flip_filter=flip_filter)
        if down > 1:
            x = upfirdn1d.upfirdn1d(x=x, f=f, down=down, flip_filter=flip_filter)
        return x

    # Fast path: no up/downsampling, padding supported by the underlying implementation => use plain conv1d.
    if up == 1 and down == 1:
        if px0 == px1 and px0 >= 0:
            return _conv1d_wrapper(x=x, w=w, padding=[px0], groups=groups, flip_weight=flip_weight)

    # Fallback: Generic reference implementation.
    x = upfirdn1d.upfirdn1d(x=x, f=(f if up > 1 else None), up=up, padding=[px0,px1], gain=up**2, flip_filter=flip_filter)
    x = _conv1d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
    if down > 1:
        x = upfirdn1d.upfirdn1d(x=x, f=f, down=down, flip_filter=flip_filter)
    return x

#----------------------------------------------------------------------------
