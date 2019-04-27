# File content is auto-generated. Do not modify.
# pylint: skip-file
from ._internal import NDArrayBase
from ..base import _Null

def adjust_lighting(data=None, alpha=_Null, out=None, name=None, **kwargs):
    r"""Adjust the lighting level of the input. Follow the AlexNet style.

    Defined in src/operator/image/image_random.cc:L118

    Parameters
    ----------
    data : NDArray
        The input.
    alpha : tuple of <float>, required
        The lighting alphas for the R, G, B channels.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def flip_left_right(data=None, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L68

    Parameters
    ----------
    data : NDArray
        The input.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def flip_top_bottom(data=None, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L76

    Parameters
    ----------
    data : NDArray
        The input.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def normalize(data=None, mean=_Null, std=_Null, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L52

    Parameters
    ----------
    data : NDArray
        The input.
    mean : tuple of <float>, required
        Sequence of mean for each channel.
    std : tuple of <float>, required
        Sequence of standard deviations for each channel.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def random_brightness(data=None, min_factor=_Null, max_factor=_Null, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L84

    Parameters
    ----------
    data : NDArray
        The input.
    min_factor : float, required
        Minimum factor.
    max_factor : float, required
        Maximum factor.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def random_color_jitter(data=None, brightness=_Null, contrast=_Null, saturation=_Null, hue=_Null, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L111

    Parameters
    ----------
    data : NDArray
        The input.
    brightness : float, required
        How much to jitter brightness.
    contrast : float, required
        How much to jitter contrast.
    saturation : float, required
        How much to jitter saturation.
    hue : float, required
        How much to jitter hue.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def random_contrast(data=None, min_factor=_Null, max_factor=_Null, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L90

    Parameters
    ----------
    data : NDArray
        The input.
    min_factor : float, required
        Minimum factor.
    max_factor : float, required
        Maximum factor.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def random_flip_left_right(data=None, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L72

    Parameters
    ----------
    data : NDArray
        The input.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def random_flip_top_bottom(data=None, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L80

    Parameters
    ----------
    data : NDArray
        The input.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def random_hue(data=None, min_factor=_Null, max_factor=_Null, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L104

    Parameters
    ----------
    data : NDArray
        The input.
    min_factor : float, required
        Minimum factor.
    max_factor : float, required
        Maximum factor.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def random_lighting(data=None, alpha_std=_Null, out=None, name=None, **kwargs):
    r"""Randomly add PCA noise. Follow the AlexNet style.

    Defined in src/operator/image/image_random.cc:L125

    Parameters
    ----------
    data : NDArray
        The input.
    alpha_std : float, optional, default=0.05
        Level of the lighting noise.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def random_saturation(data=None, min_factor=_Null, max_factor=_Null, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L97

    Parameters
    ----------
    data : NDArray
        The input.
    min_factor : float, required
        Minimum factor.
    max_factor : float, required
        Maximum factor.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def to_tensor(data=None, out=None, name=None, **kwargs):
    r"""

    Defined in src/operator/image/image_random.cc:L42

    Parameters
    ----------
    data : NDArray
        The input.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

__all__ = ['adjust_lighting', 'flip_left_right', 'flip_top_bottom', 'normalize', 'random_brightness', 'random_color_jitter', 'random_contrast', 'random_flip_left_right', 'random_flip_top_bottom', 'random_hue', 'random_lighting', 'random_saturation', 'to_tensor']