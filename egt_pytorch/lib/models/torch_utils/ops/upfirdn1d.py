# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom PyTorch ops for efficient resampling of 1d vectors."""

import os
import warnings
import numpy as np
import torch
import traceback

from .. import custom_ops
from .. import misc
from . import conv1d_gradfix

#----------------------------------------------------------------------------

_inited = False
_plugin = None

def _init():
    global _inited, _plugin
    if not _inited:
        sources = ['upfirdn2d.cpp', 'upfirdn2d.cu']
        sources = [os.path.join(os.path.dirname(__file__), s) for s in sources]
        try:
            _plugin = custom_ops.get_plugin('upfirdn2d_plugin', sources=sources, extra_cuda_cflags=['--use_fast_math'])
        except:
            warnings.warn('Failed to build CUDA kernels for upfirdn2d. Falling back to slow reference implementation. Details:\n\n' + traceback.format_exc())
    return _plugin is not None

def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, _ = scaling
    assert sx >= 1
    return sx

def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx0, padx1 = padding
    padx0, padx1 = padding
    return padx0, padx1

def _get_filter_size(f):
    if f is None:
        return 1
    assert isinstance(f, torch.Tensor) and f.ndim in [1,2]
    fd = f.shape[0]
    with misc.suppress_tracer_warnings():
        fd = int(fd)
    # misc.assert_shape(f, [fd][:f.ndim])
    assert fd >= 1
    return fd

#----------------------------------------------------------------------------

def setup_filter(f, device=torch.device('cpu'), normalize=True, flip_filter=False, gain=1, separable=None):
    r"""Convenience function to setup 1d FIR filter for `upfirdn1d()`.

    Args:
        f:           Torch tensor, numpy array, or python list of the shape
                     `[filter_dimension]` (non-separable),
                     `[filter_taps]` (separable),
                     `[]` (impulse), or
                     `None` (identity).
        device:      Result device (default: cpu).
        normalize:   Normalize the filter so that it retains the magnitude
                     for constant input signal (DC)? (default: True).
        flip_filter: Flip the filter? (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        separable:   Return a separable filter? (default: select automatically).

    Returns:
        Float32 tensor of the shape
        `[filter_dimension]` (non-separable) or
        `[filter_taps]` (separable).
    """
    # Validate.
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]
    # Separable?
    if separable is None:
        separable = (f.ndim == 1 and f.numel() >= 8)
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    assert f.ndim == (1 if separable else 2)

    # Apply normalize, flip, gain, and device.
    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * (gain ** (f.ndim / 2))
    f = f.to(device=device)
    return f

#----------------------------------------------------------------------------

def upfirdn1d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Pad, upsample, filter, and downsample a batch of 1d images.

    Performs the following sequence of operations for each channel:

    1. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    2. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.

    3. Convolve the image with the specified 1d FIR filter (`f`), shrinking it
       so that the footprint of all output pixels lies within the input image.

    4. Downsample the image by keeping every Nth pixel (`down`).

    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_dimension]`.
        f:           Float32 FIR filter of the shape
                     `[filter_dimension]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_dimension]`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    # if impl == 'cuda' and x.device.type == 'cuda' and _init():
    #     return _upfirdn1d_cuda(up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain).apply(x, f)
    return _upfirdn1d_ref(x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain)

#----------------------------------------------------------------------------

@misc.profiled_function
def _upfirdn1d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Slow reference implementation of `upfirdn1d()` using standard PyTorch ops.
    """
    # Validate arguments.
    assert isinstance(x, torch.Tensor) and x.ndim == 3
    if f is None:
        f = torch.ones([1], dtype=torch.float32, device=x.device)
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad
    batch_size, num_channels, in_dimension = x.shape
    upx = _parse_scaling(up)
    downx = _parse_scaling(down)
    padx0, padx1 = _parse_padding(padding)

    # Upsample by inserting zeros.
    x = x.reshape([batch_size, num_channels, in_dimension, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1])
    x = x.reshape([batch_size, num_channels, in_dimension * upx])

    # Pad or crop.
    x = torch.nn.functional.pad(x, [max(padx0, 0), max(padx1, 0)])
    x = x[:, :, max(-padx0, 0) : x.shape[2] - max(-padx1, 0)]

    # Setup filter.
    f = f * (gain ** (f.ndim / 2))
    f = f.to(x.dtype)
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))
    # Convolve with the filter.
    f = f[0]
    f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    x = conv1d_gradfix.conv1d(input=x, weight=f, groups=num_channels)

    # Downsample by throwing away pixels.
    x = x[:, :, ::downx]
    return x

#----------------------------------------------------------------------------

_upfirdn1d_cuda_cache = dict()

def _upfirdn1d_cuda(up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Fast CUDA implementation of `upfirdn1d()` using custom ops.
    """
    # Parse arguments.
    upx = _parse_scaling(up)
    downx = _parse_scaling(down)
    padx0, padx1 = _parse_padding(padding)

    # Lookup from cache.
    key = (upx, downx, padx0, padx1, flip_filter, gain)
    if key in _upfirdn1d_cuda_cache:
        return _upfirdn1d_cuda_cache[key]

    # Forward op.
    class upfirdn1dCuda(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, f): # pylint: disable=arguments-differ
            assert isinstance(x, torch.Tensor) and x.ndim == 3
            if f is None:
                f = torch.ones([1], dtype=torch.float32, device=x.device)
            assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
            y = x
            if f.ndim == 1:
                y = _plugin.upfirdn2d(y, f, upx, downx, padx0, padx1, flip_filter, gain)
            else:
                y = _plugin.upfirdn2d(y, f.unsqueeze(0), upx, 1, downx, 1, padx0, padx1, 0, 0, flip_filter, np.sqrt(gain))
            ctx.save_for_backward(f)
            ctx.x_shape = x.shape
            return y

        @staticmethod
        def backward(ctx, dy): # pylint: disable=arguments-differ
            f, = ctx.saved_tensors
            _, _, id = ctx.x_shape
            _, _, od = dy.shape
            fd = _get_filter_size(f)
            p = [
                fd - padx0 - 1,
                id * upx - od * downx + padx0 - upx + 1,
            ]
            dx = None

            if ctx.needs_input_grad[0]:
                dx = _upfirdn1d_cuda(up=down, down=up, padding=p, flip_filter=(not flip_filter), gain=gain).apply(dy, f)

            assert not ctx.needs_input_grad[1]
            return dx

    # Add to cache.
    _upfirdn1d_cuda_cache[key] = upfirdn1dCuda
    return upfirdn1dCuda

#----------------------------------------------------------------------------

def filter1d(x, f, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Filter a batch of 1d images using the given 1d FIR filter.

    By default, the result is padded so that its shape matches the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_dimension]`.
        f:           Float32 FIR filter of the shape
                     `[filter_dimension]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        padding:     Padding with respect to the output. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    padx0, padx1 = _parse_padding(padding)
    fd = _get_filter_size(f)
    p = [
        padx0 + fd // 2,
        padx1 + (fd - 1) // 2
    ]
    return upfirdn1d(x, f, padding=p, flip_filter=flip_filter, gain=gain, impl=impl)

#----------------------------------------------------------------------------

def upsample1d(x, f, up=1, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Upsample a batch of 1d images using the given 1d FIR filter.

    By default, the result is padded so that its shape is a multiple of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_dimension]`.
        f:           Float32 FIR filter of the shape
                     `[filter_dimension]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x]` (default: 1).
        padding:     Padding with respect to the output. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    upx = _parse_scaling(up)
    padx0, padx1 = _parse_padding(padding)
    fd = _get_filter_size(f)
    p = [
        padx0 + (fd + upx - 1) // 2,
        padx1 + (fd - upx) // 2
    ]
    return upfirdn1d(x, f, up=up, padding=p, flip_filter=flip_filter, gain=gain*upx, impl=impl)

#----------------------------------------------------------------------------

def downsample1d(x, f, down=2, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Downsample a batch of 1d images using the given 1d FIR filter.

    By default, the result is padded so that its shape is a fraction of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the input. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    downx = _parse_scaling(down)
    padx0, padx1 = _parse_padding(padding)
    fd = _get_filter_size(f)
    p = [
        padx0 + (fd - downx + 1) // 2,
        padx1 + (fd - downx) // 2
    ]
    return upfirdn1d(x, f, down=down, padding=p, flip_filter=flip_filter, gain=gain, impl=impl)

#----------------------------------------------------------------------------
