# File content is auto-generated. Do not modify.
# pylint: skip-file
from ._internal import SymbolBase
from ..base import _Null

def _CachedOp(*data, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol[]
        input data list

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _CrossDeviceCopy(name=None, attr=None, out=None, **kwargs):
    r"""Special op to copy data cross device

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _CustomFunction(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _Div(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Divides arguments element-wise.

    The storage type of ``elemwise_div`` output is always dense



    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _DivScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Divide an array with a scalar.

    ``_div_scalar`` only operates on data array of input if input is sparse.

    For example, if input of shape (100, 100) has only 2 non zero elements,
    i.e. input.data = [5, 6], scalar = nan,
    it will result output.data = [nan, nan] instead of 10000 nans.



    Defined in src/operator/tensor/elemwise_binary_scalar_op_basic.cc:L171

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _Equal(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _EqualScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _Greater(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _GreaterEqualScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _GreaterScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _Greater_Equal(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _Hypot(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Given the "legs" of a right triangle, return its hypotenuse.



    Defined in src/operator/tensor/elemwise_binary_op_extended.cc:L79

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _HypotScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _Lesser(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _LesserEqualScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _LesserScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _Lesser_Equal(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _LogicalAndScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _LogicalOrScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _LogicalXorScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _Logical_And(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _Logical_Or(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _Logical_Xor(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _Maximum(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _MaximumScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _Minimum(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _MinimumScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _Minus(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Subtracts arguments element-wise.

    The storage type of ``elemwise_sub`` output depends on storage types of inputs

       - elemwise_sub(row_sparse, row_sparse) = row_sparse
       - elemwise_sub(csr, csr) = csr
       - elemwise_sub(default, csr) = default
       - elemwise_sub(csr, default) = default
       - elemwise_sub(default, rsp) = default
       - elemwise_sub(rsp, default) = default
       - otherwise, ``elemwise_sub`` generates output with default storage



    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _MinusScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _Mod(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _ModScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _Mul(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Multiplies arguments element-wise.

    The storage type of ``elemwise_mul`` output depends on storage types of inputs

       - elemwise_mul(default, default) = default
       - elemwise_mul(row_sparse, row_sparse) = row_sparse
       - elemwise_mul(default, row_sparse) = row_sparse
       - elemwise_mul(row_sparse, default) = row_sparse
       - elemwise_mul(csr, csr) = csr
       - otherwise, ``elemwise_mul`` generates output with default storage



    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _MulScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Multiply an array with a scalar.

    ``_mul_scalar`` only operates on data array of input if input is sparse.

    For example, if input of shape (100, 100) has only 2 non zero elements,
    i.e. input.data = [5, 6], scalar = nan,
    it will result output.data = [nan, nan] instead of 10000 nans.



    Defined in src/operator/tensor/elemwise_binary_scalar_op_basic.cc:L149

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _NDArray(*data, **kwargs):
    r"""Stub for implementing an operator implemented in native frontend language with ndarray.

    Parameters
    ----------
    data : Symbol[]
        Input data for the custom operator.
    info : ptr, required

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _Native(*data, **kwargs):
    r"""Stub for implementing an operator implemented in native frontend language.

    Parameters
    ----------
    data : Symbol[]
        Input data for the custom operator.
    info : ptr, required
    need_top_grad : boolean, optional, default=1
        Whether this layer needs out grad for backward. Should be false for loss layers.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _NoGradient(name=None, attr=None, out=None, **kwargs):
    r"""Place holder for variable who cannot perform gradient

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _NotEqualScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _Not_Equal(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _Plus(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Adds arguments element-wise.

    The storage type of ``elemwise_add`` output depends on storage types of inputs

       - elemwise_add(row_sparse, row_sparse) = row_sparse
       - elemwise_add(csr, csr) = csr
       - elemwise_add(default, csr) = default
       - elemwise_add(csr, default) = default
       - elemwise_add(default, rsp) = default
       - elemwise_add(rsp, default) = default
       - otherwise, ``elemwise_add`` generates output with default storage



    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _PlusScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _Power(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _PowerScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _RDivScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _RMinusScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _RModScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _RPowerScalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _add(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Adds arguments element-wise.

    The storage type of ``elemwise_add`` output depends on storage types of inputs

       - elemwise_add(row_sparse, row_sparse) = row_sparse
       - elemwise_add(csr, csr) = csr
       - elemwise_add(default, csr) = default
       - elemwise_add(csr, default) = default
       - elemwise_add(default, rsp) = default
       - elemwise_add(rsp, default) = default
       - otherwise, ``elemwise_add`` generates output with default storage



    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _arange(start=_Null, stop=_Null, step=_Null, repeat=_Null, infer_range=_Null, ctx=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Return evenly spaced values within a given interval. Similar to Numpy

    Parameters
    ----------
    start : double, required
        Start of interval. The interval includes this value. The default start value is 0.
    stop : double or None, optional, default=None
        End of interval. The interval does not include this value, except in some cases where step is not an integer and floating point round-off affects the length of out.
    step : double, optional, default=1
        Spacing between values.
    repeat : int, optional, default='1'
        The repeating time of all elements. E.g repeat=3, the element a will be repeated three times --> a, a, a.
    infer_range : boolean, optional, default=0
        When set to True, infer the stop position from the start, step, repeat, and output tensor size.
    ctx : string, optional, default=''
        Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.
    dtype : {'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'},optional, default='float32'
        Target data type.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_Activation(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_BatchNorm(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_BatchNorm_v1(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_BilinearSampler(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_CachedOp(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_Concat(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_Convolution(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_Convolution_v1(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_Correlation(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_Crop(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_Custom(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_CustomFunction(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_Deconvolution(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_Dropout(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_Embedding(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_FullyConnected(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_GridGenerator(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_IdentityAttachKLSparseReg(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_InstanceNorm(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_L2Normalization(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_LRN(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_LayerNorm(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_LeakyReLU(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_MakeLoss(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_Pad(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_Pooling(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_Pooling_v1(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_RNN(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_ROIAlign(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_ROIPooling(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_SVMOutput(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_SequenceLast(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_SequenceMask(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_SequenceReverse(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_SliceChannel(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_Softmax(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_SoftmaxActivation(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_SoftmaxOutput(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_SparseEmbedding(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_SpatialTransformer(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_SwapAxis(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_UpSampling(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward__CrossDeviceCopy(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward__NDArray(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward__Native(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward__contrib_DeformableConvolution(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward__contrib_DeformablePSROIPooling(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward__contrib_MultiBoxDetection(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward__contrib_MultiBoxPrior(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward__contrib_MultiBoxTarget(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward__contrib_MultiProposal(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward__contrib_PSROIPooling(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward__contrib_Proposal(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward__contrib_SyncBatchNorm(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward__contrib_count_sketch(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward__contrib_fft(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward__contrib_ifft(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_abs(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_add(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_arccos(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_arccosh(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_arcsin(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_arcsinh(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_arctan(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_arctanh(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_batch_dot(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_broadcast_add(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_broadcast_div(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_broadcast_hypot(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_broadcast_maximum(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_broadcast_minimum(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_broadcast_mod(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_broadcast_mul(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_broadcast_power(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_broadcast_sub(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_cast(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_cbrt(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_clip(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_cond(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_contrib_AdaptiveAvgPooling2D(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_contrib_BilinearResize2D(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_contrib_bipartite_matching(is_ascend=_Null, threshold=_Null, topk=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    is_ascend : boolean, optional, default=0
        Use ascend order for scores instead of descending. Please set threshold accordingly.
    threshold : float, required
        Ignore matching when score < thresh, if is_ascend=false, or ignore score > thresh, if is_ascend=true.
    topk : int, optional, default='-1'
        Limit the number of matches to topk, set -1 for no limit

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_contrib_boolean_mask(axis=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    axis : int, optional, default='0'
        An integer that represents the axis in NDArray to mask from.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_contrib_box_iou(format=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    format : {'center', 'corner'},optional, default='corner'
        The box encoding type. 
     "corner" means boxes are encoded as [xmin, ymin, xmax, ymax], "center" means boxes are encodes as [x, y, width, height].

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_contrib_box_nms(overlap_thresh=_Null, valid_thresh=_Null, topk=_Null, coord_start=_Null, score_index=_Null, id_index=_Null, force_suppress=_Null, in_format=_Null, out_format=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    overlap_thresh : float, optional, default=0.5
        Overlapping(IoU) threshold to suppress object with smaller score.
    valid_thresh : float, optional, default=0
        Filter input boxes to those whose scores greater than valid_thresh.
    topk : int, optional, default='-1'
        Apply nms to topk boxes with descending scores, -1 to no restriction.
    coord_start : int, optional, default='2'
        Start index of the consecutive 4 coordinates.
    score_index : int, optional, default='1'
        Index of the scores/confidence of boxes.
    id_index : int, optional, default='-1'
        Optional, index of the class categories, -1 to disable.
    force_suppress : boolean, optional, default=0
        Optional, if set false and id_index is provided, nms will only apply to boxes belongs to the same category
    in_format : {'center', 'corner'},optional, default='corner'
        The input box encoding type. 
     "corner" means boxes are encoded as [xmin, ymin, xmax, ymax], "center" means boxes are encodes as [x, y, width, height].
    out_format : {'center', 'corner'},optional, default='corner'
        The output box encoding type. 
     "corner" means boxes are encoded as [xmin, ymin, xmax, ymax], "center" means boxes are encodes as [x, y, width, height].

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_copy(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_cos(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_cosh(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_ctc_loss(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_degrees(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_diag(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_div(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_div_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_dot(transpose_a=_Null, transpose_b=_Null, forward_stype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    transpose_a : boolean, optional, default=0
        If true then transpose the first input before dot.
    transpose_b : boolean, optional, default=0
        If true then transpose the second input before dot.
    forward_stype : {None, 'csr', 'default', 'row_sparse'},optional, default='None'
        The desired storage type of the forward output given by user, if thecombination of input storage types and this hint does not matchany implemented ones, the dot operator will perform fallback operationand still produce an output of the desired storage type.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_erf(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_expm1(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_foreach(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_gamma(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_gammaln(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_gather_nd(data=None, indices=None, shape=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Accumulates data according to indices and get the result. It's the backward of
    `gather_nd`.

    Given `data` with shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})` and indices with shape
    `(M, Y_0, ..., Y_{K-1})`, the output will have shape `(X_0, X_1, ..., X_{N-1})`,
    where `M <= N`. If `M == N`, data shape should simply be `(Y_0, ..., Y_{K-1})`.

    The elements in output is defined as follows::

      output[indices[0, y_0, ..., y_{K-1}],
             ...,
             indices[M-1, y_0, ..., y_{K-1}],
             x_M, ..., x_{N-1}] += data[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}]

    all other entries in output are 0 or the original value if AddTo is triggered.

    Examples::

      data = [2, 3, 0]
      indices = [[1, 1, 0], [0, 1, 0]]
      shape = (2, 2)
      _backward_gather_nd(data, indices, shape) = [[0, 0], [2, 3]] # Same as scatter_nd

      # The difference between scatter_nd and scatter_nd_acc is the latter will accumulate
      #  the values that point to the same index.

      data = [2, 3, 0]
      indices = [[1, 1, 0], [1, 1, 0]]
      shape = (2, 2)
      _backward_gather_nd(data, indices, shape) = [[0, 0], [0, 5]]



    Parameters
    ----------
    data : Symbol
        data
    indices : Symbol
        indices
    shape : Shape(tuple), required
        Shape of output.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_hard_sigmoid(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_hypot(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_hypot_scalar(lhs=None, rhs=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input
    scalar : float
        scalar value

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_linalg_gelqf(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_linalg_gemm(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_linalg_gemm2(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_linalg_potrf(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_linalg_potri(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_linalg_sumlogdiag(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_linalg_syevd(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_linalg_syrk(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_linalg_trmm(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_linalg_trsm(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_linear_reg_out(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_log(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_log10(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_log1p(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_log2(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_log_softmax(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_logistic_reg_out(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_mae_reg_out(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_max(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_maximum(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_maximum_scalar(lhs=None, rhs=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input
    scalar : float
        scalar value

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_mean(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_min(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_minimum(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_minimum_scalar(lhs=None, rhs=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input
    scalar : float
        scalar value

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_mod(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_mod_scalar(lhs=None, rhs=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input
    scalar : float
        scalar value

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_mul(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_mul_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_nanprod(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_nansum(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_norm(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_pick(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_power(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_power_scalar(lhs=None, rhs=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input
    scalar : float
        scalar value

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_prod(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_radians(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_rcbrt(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_rdiv_scalar(lhs=None, rhs=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input
    scalar : float
        scalar value

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_reciprocal(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_relu(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_repeat(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_reverse(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_rmod_scalar(lhs=None, rhs=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input
    scalar : float
        scalar value

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_rpower_scalar(lhs=None, rhs=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input
    scalar : float
        scalar value

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_rsqrt(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_sample_multinomial(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_sigmoid(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_sign(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_sin(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_sinh(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_slice(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_slice_axis(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_slice_like(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_smooth_l1(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_softmax(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_softmax_cross_entropy(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_softmin(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_softsign(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_sparse_retain(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_sqrt(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_square(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_square_sum(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_squeeze(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_stack(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_sub(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_sum(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_take(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_tan(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_tanh(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_tile(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_topk(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_where(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _backward_while_loop(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _broadcast_backward(name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------


    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _cond(*data, **kwargs):
    r"""Run a if-then-else using user-defined condition and computation

    From:src/operator/control_flow.cc:1379
    This function support variable length of positional input.

    Parameters
    ----------
    cond : Symbol
        Input graph for the condition.
    then_branch : Symbol
        Input graph for the then branch.
    else_branch : Symbol
        Input graph for the else branch.
    data : Symbol[]
        The input arrays that include data arrays and states.
    num_outputs : int, required
        The number of outputs of the subgraph.
    cond_input_locs : tuple of <>, required
        The locations of cond's inputs in the given inputs.
    then_input_locs : tuple of <>, required
        The locations of then's inputs in the given inputs.
    else_input_locs : tuple of <>, required
        The locations of else's inputs in the given inputs.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _copy(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns a copy of the input.

    From:src/operator/tensor/elemwise_unary_op_basic.cc:200

    Parameters
    ----------
    data : Symbol
        The input array.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _copyto(data=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : NDArray
        input data

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _crop_assign(lhs=None, rhs=None, begin=_Null, end=_Null, step=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Assign the rhs to a cropped subset of lhs.

    Requirements
    ------------
    - output should be explicitly given and be the same as lhs.
    - lhs and rhs are of the same data type, and on the same device.


    From:src/operator/tensor/matrix_op.cc:440

    Parameters
    ----------
    lhs : Symbol
        Source input
    rhs : Symbol
        value to assign
    begin : Shape(tuple), required
        starting indices for the slice operation, supports negative indices.
    end : Shape(tuple), required
        ending indices for the slice operation, supports negative indices.
    step : Shape(tuple), optional, default=[]
        step for the slice operation, supports negative values.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _crop_assign_scalar(data=None, scalar=_Null, begin=_Null, end=_Null, step=_Null, name=None, attr=None, out=None, **kwargs):
    r"""(Assign the scalar to a cropped subset of the input.

    Requirements
    ------------
    - output should be explicitly given and be the same as input
    )

    From:src/operator/tensor/matrix_op.cc:465

    Parameters
    ----------
    data : Symbol
        Source input
    scalar : double, optional, default=0
        The scalar value for assignment.
    begin : Shape(tuple), required
        starting indices for the slice operation, supports negative indices.
    end : Shape(tuple), required
        ending indices for the slice operation, supports negative indices.
    step : Shape(tuple), optional, default=[]
        step for the slice operation, supports negative values.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _cvcopyMakeBorder(src=None, top=_Null, bot=_Null, left=_Null, right=_Null, type=_Null, value=_Null, values=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Pad image border with OpenCV. 


    Parameters
    ----------
    src : NDArray
        source image
    top : int, required
        Top margin.
    bot : int, required
        Bottom margin.
    left : int, required
        Left margin.
    right : int, required
        Right margin.
    type : int, optional, default='0'
        Filling type (default=cv2.BORDER_CONSTANT).
    value : double, optional, default=0
        (Deprecated! Use ``values`` instead.) Fill with single value.
    values : tuple of <double>, optional, default=[]
        Fill with value(RGB[A] or gray), up to 4 channels.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _cvimdecode(buf=None, flag=_Null, to_rgb=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Decode image with OpenCV. 
    Note: return image in RGB by default, instead of OpenCV's default BGR.

    Parameters
    ----------
    buf : NDArray
        Buffer containing binary encoded image
    flag : int, optional, default='1'
        Convert decoded image to grayscale (0) or color (1).
    to_rgb : boolean, optional, default=1
        Whether to convert decoded image to mxnet's default RGB format (instead of opencv's default BGR).

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _cvimread(filename=_Null, flag=_Null, to_rgb=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Read and decode image with OpenCV. 
    Note: return image in RGB by default, instead of OpenCV's default BGR.

    Parameters
    ----------
    filename : string, required
        Name of the image file to be loaded.
    flag : int, optional, default='1'
        Convert decoded image to grayscale (0) or color (1).
    to_rgb : boolean, optional, default=1
        Whether to convert decoded image to mxnet's default RGB format (instead of opencv's default BGR).

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _cvimresize(src=None, w=_Null, h=_Null, interp=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Resize image with OpenCV. 


    Parameters
    ----------
    src : NDArray
        source image
    w : int, required
        Width of resized image.
    h : int, required
        Height of resized image.
    interp : int, optional, default='1'
        Interpolation method (default=cv2.INTER_LINEAR).

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _div(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Divides arguments element-wise.

    The storage type of ``elemwise_div`` output is always dense



    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _div_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Divide an array with a scalar.

    ``_div_scalar`` only operates on data array of input if input is sparse.

    For example, if input of shape (100, 100) has only 2 non zero elements,
    i.e. input.data = [5, 6], scalar = nan,
    it will result output.data = [nan, nan] instead of 10000 nans.



    Defined in src/operator/tensor/elemwise_binary_scalar_op_basic.cc:L171

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _equal(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _equal_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _eye(N=_Null, M=_Null, k=_Null, ctx=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : , required
        Number of rows in the output.
    M : , optional, default=0
        Number of columns in the output. If 0, defaults to N
    k : , optional, default=0
        Index of the diagonal. 0 (the default) refers to the main diagonal.A positive value refers to an upper diagonal.A negative value to a lower diagonal.
    ctx : string, optional, default=''
        Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.
    dtype : {'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'},optional, default='float32'
        Target data type.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _foreach(*data, **kwargs):
    r"""Run a for loop over an NDArray with user-defined computation

    From:src/operator/control_flow.cc:1256
    This function support variable length of positional input.

    Parameters
    ----------
    fn : Symbol
        Input graph.
    data : Symbol[]
        The input arrays that include data arrays and states.
    num_outputs : int, required
        The number of outputs of the subgraph.
    num_out_data : int, required
        The number of output data of the subgraph.
    in_state_locs : tuple of <>, required
        The locations of loop states among the inputs.
    in_data_locs : tuple of <>, required
        The locations of input data among the inputs.
    remain_locs : tuple of <>, required
        The locations of remaining data among the inputs.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _full(shape=_Null, ctx=_Null, dtype=_Null, value=_Null, name=None, attr=None, out=None, **kwargs):
    r"""fill target with a scalar value

    Parameters
    ----------
    shape : Shape(tuple), optional, default=[]
        The shape of the output
    ctx : string, optional, default=''
        Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.
    dtype : {'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'},optional, default='float32'
        Target data type.
    value : double, required
        Value with which to fill newly created tensor

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _grad_add(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _greater(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _greater_equal(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _greater_equal_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _greater_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _histogram(data=None, bins=None, bin_cnt=_Null, range=_Null, name=None, attr=None, out=None, **kwargs):
    r"""This operators implements the histogram function.

    Example::
      x = [[0, 1], [2, 2], [3, 4]]
      histo, bin_edges = histogram(data=x, bin_bounds=[], bin_cnt=5, range=(0,5))
      histo = [1, 1, 2, 1, 1]
      bin_edges = [0., 1., 2., 3., 4.]
      histo, bin_edges = histogram(data=x, bin_bounds=[0., 2.1, 3.])
      histo = [4, 1]



    Defined in src/operator/tensor/histogram.cc:L136

    Parameters
    ----------
    data : Symbol
        Input ndarray
    bins : Symbol
        Input ndarray
    bin_cnt : int or None, optional, default='None'
        Number of bins for uniform case
    range : , optional, default=None
        The lower and upper range of the bins. if not provided, range is simply (a.min(), a.max()). values outside the range are ignored. the first element of the range must be less than or equal to the second. range affects the automatic bin computation as well. while bin width is computed to be optimal based on the actual data within range, the bin count will fill the entire range including portions containing no data.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _hypot(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Given the "legs" of a right triangle, return its hypotenuse.



    Defined in src/operator/tensor/elemwise_binary_op_extended.cc:L79

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _hypot_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _identity_with_attr_like_rhs(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        First input.
    rhs : Symbol
        Second input.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _imdecode(mean=None, index=_Null, x0=_Null, y0=_Null, x1=_Null, y1=_Null, c=_Null, size=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Decode an image, clip to (x0, y0, x1, y1), subtract mean, and write to buffer

    Parameters
    ----------
    mean : Symbol
        image mean
    index : int
        buffer position for output
    x0 : int
        x0
    y0 : int
        y0
    x1 : int
        x1
    y1 : int
        y1
    c : int
        channel
    size : int
        length of str_img

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _lesser(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _lesser_equal(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _lesser_equal_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _lesser_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _logical_and(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _logical_and_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _logical_or(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _logical_or_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _logical_xor(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _logical_xor_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _maximum(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _maximum_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _minimum(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _minimum_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _minus(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Subtracts arguments element-wise.

    The storage type of ``elemwise_sub`` output depends on storage types of inputs

       - elemwise_sub(row_sparse, row_sparse) = row_sparse
       - elemwise_sub(csr, csr) = csr
       - elemwise_sub(default, csr) = default
       - elemwise_sub(csr, default) = default
       - elemwise_sub(default, rsp) = default
       - elemwise_sub(rsp, default) = default
       - otherwise, ``elemwise_sub`` generates output with default storage



    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _minus_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _mod(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _mod_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _mul(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Multiplies arguments element-wise.

    The storage type of ``elemwise_mul`` output depends on storage types of inputs

       - elemwise_mul(default, default) = default
       - elemwise_mul(row_sparse, row_sparse) = row_sparse
       - elemwise_mul(default, row_sparse) = row_sparse
       - elemwise_mul(row_sparse, default) = row_sparse
       - elemwise_mul(csr, csr) = csr
       - otherwise, ``elemwise_mul`` generates output with default storage



    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _mul_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Multiply an array with a scalar.

    ``_mul_scalar`` only operates on data array of input if input is sparse.

    For example, if input of shape (100, 100) has only 2 non zero elements,
    i.e. input.data = [5, 6], scalar = nan,
    it will result output.data = [nan, nan] instead of 10000 nans.



    Defined in src/operator/tensor/elemwise_binary_scalar_op_basic.cc:L149

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _not_equal(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _not_equal_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _onehot_encode(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : NDArray
        Left operand to the function.
    rhs : NDArray
        Right operand to the function.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _ones(shape=_Null, ctx=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""fill target with ones

    Parameters
    ----------
    shape : Shape(tuple), optional, default=[]
        The shape of the output
    ctx : string, optional, default=''
        Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.
    dtype : {'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'},optional, default='float32'
        Target data type.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _plus(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Adds arguments element-wise.

    The storage type of ``elemwise_add`` output depends on storage types of inputs

       - elemwise_add(row_sparse, row_sparse) = row_sparse
       - elemwise_add(csr, csr) = csr
       - elemwise_add(default, csr) = default
       - elemwise_add(csr, default) = default
       - elemwise_add(default, rsp) = default
       - elemwise_add(rsp, default) = default
       - otherwise, ``elemwise_add`` generates output with default storage



    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _plus_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _power(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _power_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _ravel_multi_index(data=None, shape=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Converts a batch of index arrays into an array of flat indices. The operator follows numpy conventions so a single multi index is given by a column of the input matrix. 

    Examples::
   
       A = [[3,6,6],[4,5,1]]
       ravel(A, shape=(7,6)) = [22,41,37]



    Defined in src/operator/tensor/ravel.cc:L41

    Parameters
    ----------
    data : Symbol
        Batch of multi-indices
    shape : Shape(tuple), optional, default=[]
        Shape of the array into which the multi-indices apply.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _rdiv_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _rminus_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _rmod_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _rnn_param_concat(*data, **kwargs):
    r"""
    This function support variable length of positional input.

    Parameters
    ----------
    data : Symbol[]
        List of arrays to concatenate
    dim : int, optional, default='1'
        the dimension to be concated.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _rpower_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _sample_exponential(lam=None, shape=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Concurrent sampling from multiple
    exponential distributions with parameters lambda (rate).

    The parameters of the distributions are provided as an input array.
    Let *[s]* be the shape of the input array, *n* be the dimension of *[s]*, *[t]*
    be the shape specified as the parameter of the operator, and *m* be the dimension
    of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.

    For any valid *n*-dimensional index *i* with respect to the input array, *output[i]*
    will be an *m*-dimensional array that holds randomly drawn samples from the distribution
    which is parameterized by the input value at index *i*. If the shape parameter of the
    operator is not set, then one sample will be drawn per distribution and the output array
    has the same shape as the input array.

    Examples::

       lam = [ 1.0, 8.5 ]

       // Draw a single sample for each distribution
       sample_exponential(lam) = [ 0.51837951,  0.09994757]

       // Draw a vector containing two samples for each distribution
       sample_exponential(lam, shape=(2)) = [[ 0.51837951,  0.19866663],
                                             [ 0.09994757,  0.50447971]]


    Defined in src/operator/random/multisample_op.cc:L284

    Parameters
    ----------
    lam : Symbol
        Lambda (rate) parameters of the distributions.
    shape : Shape(tuple), optional, default=[]
        Shape to be sampled from each random distribution.
    dtype : {'None', 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _sample_gamma(alpha=None, beta=None, shape=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Concurrent sampling from multiple
    gamma distributions with parameters *alpha* (shape) and *beta* (scale).

    The parameters of the distributions are provided as input arrays.
    Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
    be the shape specified as the parameter of the operator, and *m* be the dimension
    of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.

    For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
    will be an *m*-dimensional array that holds randomly drawn samples from the distribution
    which is parameterized by the input values at index *i*. If the shape parameter of the
    operator is not set, then one sample will be drawn per distribution and the output array
    has the same shape as the input arrays.

    Examples::

       alpha = [ 0.0, 2.5 ]
       beta = [ 1.0, 0.7 ]

       // Draw a single sample for each distribution
       sample_gamma(alpha, beta) = [ 0.        ,  2.25797319]

       // Draw a vector containing two samples for each distribution
       sample_gamma(alpha, beta, shape=(2)) = [[ 0.        ,  0.        ],
                                               [ 2.25797319,  1.70734084]]


    Defined in src/operator/random/multisample_op.cc:L282

    Parameters
    ----------
    alpha : Symbol
        Alpha (shape) parameters of the distributions.
    shape : Shape(tuple), optional, default=[]
        Shape to be sampled from each random distribution.
    dtype : {'None', 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
    beta : Symbol
        Beta (scale) parameters of the distributions.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _sample_generalized_negative_binomial(mu=None, alpha=None, shape=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Concurrent sampling from multiple
    generalized negative binomial distributions with parameters *mu* (mean) and *alpha* (dispersion).

    The parameters of the distributions are provided as input arrays.
    Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
    be the shape specified as the parameter of the operator, and *m* be the dimension
    of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.

    For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
    will be an *m*-dimensional array that holds randomly drawn samples from the distribution
    which is parameterized by the input values at index *i*. If the shape parameter of the
    operator is not set, then one sample will be drawn per distribution and the output array
    has the same shape as the input arrays.

    Samples will always be returned as a floating point data type.

    Examples::

       mu = [ 2.0, 2.5 ]
       alpha = [ 1.0, 0.1 ]

       // Draw a single sample for each distribution
       sample_generalized_negative_binomial(mu, alpha) = [ 0.,  3.]

       // Draw a vector containing two samples for each distribution
       sample_generalized_negative_binomial(mu, alpha, shape=(2)) = [[ 0.,  3.],
                                                                     [ 3.,  1.]]


    Defined in src/operator/random/multisample_op.cc:L293

    Parameters
    ----------
    mu : Symbol
        Means of the distributions.
    shape : Shape(tuple), optional, default=[]
        Shape to be sampled from each random distribution.
    dtype : {'None', 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
    alpha : Symbol
        Alpha (dispersion) parameters of the distributions.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _sample_multinomial(data=None, shape=_Null, get_prob=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Concurrent sampling from multiple multinomial distributions.

    *data* is an *n* dimensional array whose last dimension has length *k*, where
    *k* is the number of possible outcomes of each multinomial distribution. This
    operator will draw *shape* samples from each distribution. If shape is empty
    one sample will be drawn from each distribution.

    If *get_prob* is true, a second array containing log likelihood of the drawn
    samples will also be returned. This is usually used for reinforcement learning
    where you can provide reward as head gradient for this array to estimate
    gradient.

    Note that the input distribution must be normalized, i.e. *data* must sum to
    1 along its last axis.

    Examples::

       probs = [[0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0]]

       // Draw a single sample for each distribution
       sample_multinomial(probs) = [3, 0]

       // Draw a vector containing two samples for each distribution
       sample_multinomial(probs, shape=(2)) = [[4, 2],
                                               [0, 0]]

       // requests log likelihood
       sample_multinomial(probs, get_prob=True) = [2, 1], [0.2, 0.3]


    Parameters
    ----------
    data : Symbol
        Distribution probabilities. Must sum to one on the last axis.
    shape : Shape(tuple), optional, default=[]
        Shape to be sampled from each random distribution.
    get_prob : boolean, optional, default=0
        Whether to also return the log probability of sampled result. This is usually used for differentiating through stochastic variables, e.g. in reinforcement learning.
    dtype : {'float16', 'float32', 'float64', 'int32', 'uint8'},optional, default='int32'
        DType of the output in case this can't be inferred.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _sample_negative_binomial(k=None, p=None, shape=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Concurrent sampling from multiple
    negative binomial distributions with parameters *k* (failure limit) and *p* (failure probability).

    The parameters of the distributions are provided as input arrays.
    Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
    be the shape specified as the parameter of the operator, and *m* be the dimension
    of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.

    For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
    will be an *m*-dimensional array that holds randomly drawn samples from the distribution
    which is parameterized by the input values at index *i*. If the shape parameter of the
    operator is not set, then one sample will be drawn per distribution and the output array
    has the same shape as the input arrays.

    Samples will always be returned as a floating point data type.

    Examples::

       k = [ 20, 49 ]
       p = [ 0.4 , 0.77 ]

       // Draw a single sample for each distribution
       sample_negative_binomial(k, p) = [ 15.,  16.]

       // Draw a vector containing two samples for each distribution
       sample_negative_binomial(k, p, shape=(2)) = [[ 15.,  50.],
                                                    [ 16.,  12.]]


    Defined in src/operator/random/multisample_op.cc:L289

    Parameters
    ----------
    k : Symbol
        Limits of unsuccessful experiments.
    shape : Shape(tuple), optional, default=[]
        Shape to be sampled from each random distribution.
    dtype : {'None', 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
    p : Symbol
        Failure probabilities in each experiment.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _sample_normal(mu=None, sigma=None, shape=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Concurrent sampling from multiple
    normal distributions with parameters *mu* (mean) and *sigma* (standard deviation).

    The parameters of the distributions are provided as input arrays.
    Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
    be the shape specified as the parameter of the operator, and *m* be the dimension
    of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.

    For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
    will be an *m*-dimensional array that holds randomly drawn samples from the distribution
    which is parameterized by the input values at index *i*. If the shape parameter of the
    operator is not set, then one sample will be drawn per distribution and the output array
    has the same shape as the input arrays.

    Examples::

       mu = [ 0.0, 2.5 ]
       sigma = [ 1.0, 3.7 ]

       // Draw a single sample for each distribution
       sample_normal(mu, sigma) = [-0.56410581,  0.95934606]

       // Draw a vector containing two samples for each distribution
       sample_normal(mu, sigma, shape=(2)) = [[-0.56410581,  0.2928229 ],
                                              [ 0.95934606,  4.48287058]]


    Defined in src/operator/random/multisample_op.cc:L279

    Parameters
    ----------
    mu : Symbol
        Means of the distributions.
    shape : Shape(tuple), optional, default=[]
        Shape to be sampled from each random distribution.
    dtype : {'None', 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
    sigma : Symbol
        Standard deviations of the distributions.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _sample_poisson(lam=None, shape=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Concurrent sampling from multiple
    Poisson distributions with parameters lambda (rate).

    The parameters of the distributions are provided as an input array.
    Let *[s]* be the shape of the input array, *n* be the dimension of *[s]*, *[t]*
    be the shape specified as the parameter of the operator, and *m* be the dimension
    of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.

    For any valid *n*-dimensional index *i* with respect to the input array, *output[i]*
    will be an *m*-dimensional array that holds randomly drawn samples from the distribution
    which is parameterized by the input value at index *i*. If the shape parameter of the
    operator is not set, then one sample will be drawn per distribution and the output array
    has the same shape as the input array.

    Samples will always be returned as a floating point data type.

    Examples::

       lam = [ 1.0, 8.5 ]

       // Draw a single sample for each distribution
       sample_poisson(lam) = [  0.,  13.]

       // Draw a vector containing two samples for each distribution
       sample_poisson(lam, shape=(2)) = [[  0.,   4.],
                                         [ 13.,   8.]]


    Defined in src/operator/random/multisample_op.cc:L286

    Parameters
    ----------
    lam : Symbol
        Lambda (rate) parameters of the distributions.
    shape : Shape(tuple), optional, default=[]
        Shape to be sampled from each random distribution.
    dtype : {'None', 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _sample_uniform(low=None, high=None, shape=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Concurrent sampling from multiple
    uniform distributions on the intervals given by *[low,high)*.

    The parameters of the distributions are provided as input arrays.
    Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
    be the shape specified as the parameter of the operator, and *m* be the dimension
    of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.

    For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
    will be an *m*-dimensional array that holds randomly drawn samples from the distribution
    which is parameterized by the input values at index *i*. If the shape parameter of the
    operator is not set, then one sample will be drawn per distribution and the output array
    has the same shape as the input arrays.

    Examples::

       low = [ 0.0, 2.5 ]
       high = [ 1.0, 3.7 ]

       // Draw a single sample for each distribution
       sample_uniform(low, high) = [ 0.40451524,  3.18687344]

       // Draw a vector containing two samples for each distribution
       sample_uniform(low, high, shape=(2)) = [[ 0.40451524,  0.18017688],
                                               [ 3.18687344,  3.68352246]]


    Defined in src/operator/random/multisample_op.cc:L277

    Parameters
    ----------
    low : Symbol
        Lower bounds of the distributions.
    shape : Shape(tuple), optional, default=[]
        Shape to be sampled from each random distribution.
    dtype : {'None', 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).
    high : Symbol
        Upper bounds of the distributions.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _sample_unique_zipfian(range_max=_Null, shape=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Draw random samples from an an approximately log-uniform
    or Zipfian distribution without replacement.

    This operation takes a 2-D shape `(batch_size, num_sampled)`,
    and randomly generates *num_sampled* samples from the range of integers [0, range_max)
    for each instance in the batch.

    The elements in each instance are drawn without replacement from the base distribution.
    The base distribution for this operator is an approximately log-uniform or Zipfian distribution:

      P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)

    Additionaly, it also returns the number of trials used to obtain `num_sampled` samples for
    each instance in the batch.

    Example::

       samples, trials = _sample_unique_zipfian(750000, shape=(4, 8192))
       unique(samples[0]) = 8192
       unique(samples[3]) = 8192
       trials[0] = 16435



    Defined in src/operator/random/unique_sample_op.cc:L66

    Parameters
    ----------
    range_max : int, required
        The number of possible classes.
    shape : Shape(tuple), optional, default=[]
        2-D shape of the output, where shape[0] is the batch size, and shape[1] is the number of candidates to sample for each batch.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _scatter_elemwise_div(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Divides arguments element-wise.  If the left-hand-side input is 'row_sparse', then
    only the values which exist in the left-hand sparse array are computed.  The 'missing' values
    are ignored.

    The storage type of ``_scatter_elemwise_div`` output depends on storage types of inputs

    - _scatter_elemwise_div(row_sparse, row_sparse) = row_sparse
    - _scatter_elemwise_div(row_sparse, dense) = row_sparse
    - _scatter_elemwise_div(row_sparse, csr) = row_sparse
    - otherwise, ``_scatter_elemwise_div`` behaves exactly like elemwise_div and generates output
    with default storage



    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _scatter_minus_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Subtracts a scalar to a tensor element-wise.  If the left-hand-side input is
    'row_sparse' or 'csr', then only the values which exist in the left-hand sparse array are computed.
    The 'missing' values are ignored.

    The storage type of ``_scatter_minus_scalar`` output depends on storage types of inputs

    - _scatter_minus_scalar(row_sparse, scalar) = row_sparse
    - _scatter_minus_scalar(csr, scalar) = csr
    - otherwise, ``_scatter_minus_scalar`` behaves exactly like _minus_scalar and generates output
    with default storage



    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _scatter_plus_scalar(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Adds a scalar to a tensor element-wise.  If the left-hand-side input is
    'row_sparse' or 'csr', then only the values which exist in the left-hand sparse array are computed.
    The 'missing' values are ignored.

    The storage type of ``_scatter_plus_scalar`` output depends on storage types of inputs

    - _scatter_plus_scalar(row_sparse, scalar) = row_sparse
    - _scatter_plus_scalar(csr, scalar) = csr
    - otherwise, ``_scatter_plus_scalar`` behaves exactly like _plus_scalar and generates output
    with default storage



    Parameters
    ----------
    data : Symbol
        source input
    scalar : float
        scalar input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _scatter_set_nd(lhs=None, rhs=None, indices=None, shape=_Null, name=None, attr=None, out=None, **kwargs):
    r"""This operator has the same functionality as scatter_nd
    except that it does not reset the elements not indexed by the input
    index `NDArray` in the input data `NDArray`. output should be explicitly
    given and be the same as lhs.

    .. note:: This operator is for internal use only.

    Examples::

      data = [2, 3, 0]
      indices = [[1, 1, 0], [0, 1, 0]]
      out = [[1, 1], [1, 1]]
      _scatter_set_nd(lhs=out, rhs=data, indices=indices, out=out)
      out = [[0, 1], [2, 3]]



    Parameters
    ----------
    lhs : Symbol
        source input
    rhs : Symbol
        value to assign
    indices : Symbol
        indices
    shape : Shape(tuple), required
        Shape of output.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _set_value(src=_Null, name=None, attr=None, out=None, **kwargs):
    r"""

    Parameters
    ----------
    src : real_t
        Source input to the function.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _shuffle(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Randomly shuffle the elements.

    This shuffles the array along the first axis.
    The order of the elements in each subarray does not change.
    For example, if a 2D array is given, the order of the rows randomly changes,
    but the order of the elements in each row does not change.


    Parameters
    ----------
    data : Symbol
        Data to be shuffled.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _slice_assign(lhs=None, rhs=None, begin=_Null, end=_Null, step=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Assign the rhs to a cropped subset of lhs.

    Requirements
    ------------
    - output should be explicitly given and be the same as lhs.
    - lhs and rhs are of the same data type, and on the same device.


    From:src/operator/tensor/matrix_op.cc:440

    Parameters
    ----------
    lhs : Symbol
        Source input
    rhs : Symbol
        value to assign
    begin : Shape(tuple), required
        starting indices for the slice operation, supports negative indices.
    end : Shape(tuple), required
        ending indices for the slice operation, supports negative indices.
    step : Shape(tuple), optional, default=[]
        step for the slice operation, supports negative values.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _slice_assign_scalar(data=None, scalar=_Null, begin=_Null, end=_Null, step=_Null, name=None, attr=None, out=None, **kwargs):
    r"""(Assign the scalar to a cropped subset of the input.

    Requirements
    ------------
    - output should be explicitly given and be the same as input
    )

    From:src/operator/tensor/matrix_op.cc:465

    Parameters
    ----------
    data : Symbol
        Source input
    scalar : double, optional, default=0
        The scalar value for assignment.
    begin : Shape(tuple), required
        starting indices for the slice operation, supports negative indices.
    end : Shape(tuple), required
        ending indices for the slice operation, supports negative indices.
    step : Shape(tuple), optional, default=[]
        step for the slice operation, supports negative values.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _square_sum(data=None, axis=_Null, keepdims=_Null, exclude=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Computes the square sum of array elements over a given axis
    for row-sparse matrix. This is a temporary solution for fusing ops square and
    sum together for row-sparse matrix to save memory for storing gradients.
    It will become deprecated once the functionality of fusing operators is finished
    in the future.

    Example::

      dns = mx.nd.array([[0, 0], [1, 2], [0, 0], [3, 4], [0, 0]])
      rsp = dns.tostype('row_sparse')
      sum = mx.nd._internal._square_sum(rsp, axis=1)
      sum = [0, 5, 0, 25, 0]


    Defined in src/operator/tensor/square_sum.cc:L63

    Parameters
    ----------
    data : Symbol
        The input
    axis : Shape or None, optional, default=None
        The axis or axes along which to perform the reduction.

          The default, `axis=()`, will compute over all elements into a
          scalar array with shape `(1,)`.

          If `axis` is int, a reduction is performed on a particular axis.

          If `axis` is a tuple of ints, a reduction is performed on all the axes
          specified in the tuple.

          If `exclude` is true, reduction will be performed on the axes that are
          NOT in axis instead.

          Negative values means indexing from right to left.
    keepdims : boolean, optional, default=0
        If this is set to `True`, the reduced axes are left in the result as dimension with size one.
    exclude : boolean, optional, default=0
        Whether to perform reduction on axis that are NOT in axis instead.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _sub(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Subtracts arguments element-wise.

    The storage type of ``elemwise_sub`` output depends on storage types of inputs

       - elemwise_sub(row_sparse, row_sparse) = row_sparse
       - elemwise_sub(csr, csr) = csr
       - elemwise_sub(default, csr) = default
       - elemwise_sub(csr, default) = default
       - elemwise_sub(default, rsp) = default
       - elemwise_sub(rsp, default) = default
       - otherwise, ``elemwise_sub`` generates output with default storage



    Parameters
    ----------
    lhs : Symbol
        first input
    rhs : Symbol
        second input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _unravel_index(data=None, shape=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Converts an array of flat indices into a batch of index arrays. The operator follows numpy conventions so a single multi index is given by a column of the output matrix.

    Examples::

       A = [22,41,37]
       unravel(A, shape=(7,6)) = [[3,6,6],[4,5,1]]



    Defined in src/operator/tensor/ravel.cc:L65

    Parameters
    ----------
    data : Symbol
        Array of flat indices
    shape : Shape(tuple), optional, default=[]
        Shape of the array into which the multi-indices apply.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _while_loop(*data, **kwargs):
    r"""Run a while loop over with user-defined condition and computation

    From:src/operator/control_flow.cc:1317
    This function support variable length of positional input.

    Parameters
    ----------
    cond : Symbol
        Input graph for the loop condition.
    func : Symbol
        Input graph for the loop body.
    data : Symbol[]
        The input arrays that include data arrays and states.
    num_outputs : int, required
        The number of outputs of the subgraph.
    num_out_data : int, required
        The number of outputs from the function body.
    max_iterations : int, required
        Maximum number of iterations.
    cond_input_locs : tuple of <>, required
        The locations of cond's inputs in the given inputs.
    func_input_locs : tuple of <>, required
        The locations of func's inputs in the given inputs.
    func_var_locs : tuple of <>, required
        The locations of loop_vars among func's inputs.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _zeros(shape=_Null, ctx=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""fill target with zeros

    Parameters
    ----------
    shape : Shape(tuple), optional, default=[]
        The shape of the output
    ctx : string, optional, default=''
        Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.
    dtype : {'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'},optional, default='float32'
        Target data type.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def _zeros_without_dtype(shape=_Null, ctx=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""fill target with zeros without default dtype

    Parameters
    ----------
    shape : Shape(tuple), optional, default=[]
        The shape of the output
    ctx : string, optional, default=''
        Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.
    dtype : int, optional, default='-1'
        Target data type.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

__all__ = ['_CachedOp', '_CrossDeviceCopy', '_CustomFunction', '_Div', '_DivScalar', '_Equal', '_EqualScalar', '_Greater', '_GreaterEqualScalar', '_GreaterScalar', '_Greater_Equal', '_Hypot', '_HypotScalar', '_Lesser', '_LesserEqualScalar', '_LesserScalar', '_Lesser_Equal', '_LogicalAndScalar', '_LogicalOrScalar', '_LogicalXorScalar', '_Logical_And', '_Logical_Or', '_Logical_Xor', '_Maximum', '_MaximumScalar', '_Minimum', '_MinimumScalar', '_Minus', '_MinusScalar', '_Mod', '_ModScalar', '_Mul', '_MulScalar', '_NDArray', '_Native', '_NoGradient', '_NotEqualScalar', '_Not_Equal', '_Plus', '_PlusScalar', '_Power', '_PowerScalar', '_RDivScalar', '_RMinusScalar', '_RModScalar', '_RPowerScalar', '_add', '_arange', '_backward_Activation', '_backward_BatchNorm', '_backward_BatchNorm_v1', '_backward_BilinearSampler', '_backward_CachedOp', '_backward_Concat', '_backward_Convolution', '_backward_Convolution_v1', '_backward_Correlation', '_backward_Crop', '_backward_Custom', '_backward_CustomFunction', '_backward_Deconvolution', '_backward_Dropout', '_backward_Embedding', '_backward_FullyConnected', '_backward_GridGenerator', '_backward_IdentityAttachKLSparseReg', '_backward_InstanceNorm', '_backward_L2Normalization', '_backward_LRN', '_backward_LayerNorm', '_backward_LeakyReLU', '_backward_MakeLoss', '_backward_Pad', '_backward_Pooling', '_backward_Pooling_v1', '_backward_RNN', '_backward_ROIAlign', '_backward_ROIPooling', '_backward_SVMOutput', '_backward_SequenceLast', '_backward_SequenceMask', '_backward_SequenceReverse', '_backward_SliceChannel', '_backward_Softmax', '_backward_SoftmaxActivation', '_backward_SoftmaxOutput', '_backward_SparseEmbedding', '_backward_SpatialTransformer', '_backward_SwapAxis', '_backward_UpSampling', '_backward__CrossDeviceCopy', '_backward__NDArray', '_backward__Native', '_backward__contrib_DeformableConvolution', '_backward__contrib_DeformablePSROIPooling', '_backward__contrib_MultiBoxDetection', '_backward__contrib_MultiBoxPrior', '_backward__contrib_MultiBoxTarget', '_backward__contrib_MultiProposal', '_backward__contrib_PSROIPooling', '_backward__contrib_Proposal', '_backward__contrib_SyncBatchNorm', '_backward__contrib_count_sketch', '_backward__contrib_fft', '_backward__contrib_ifft', '_backward_abs', '_backward_add', '_backward_arccos', '_backward_arccosh', '_backward_arcsin', '_backward_arcsinh', '_backward_arctan', '_backward_arctanh', '_backward_batch_dot', '_backward_broadcast_add', '_backward_broadcast_div', '_backward_broadcast_hypot', '_backward_broadcast_maximum', '_backward_broadcast_minimum', '_backward_broadcast_mod', '_backward_broadcast_mul', '_backward_broadcast_power', '_backward_broadcast_sub', '_backward_cast', '_backward_cbrt', '_backward_clip', '_backward_cond', '_backward_contrib_AdaptiveAvgPooling2D', '_backward_contrib_BilinearResize2D', '_backward_contrib_bipartite_matching', '_backward_contrib_boolean_mask', '_backward_contrib_box_iou', '_backward_contrib_box_nms', '_backward_copy', '_backward_cos', '_backward_cosh', '_backward_ctc_loss', '_backward_degrees', '_backward_diag', '_backward_div', '_backward_div_scalar', '_backward_dot', '_backward_erf', '_backward_expm1', '_backward_foreach', '_backward_gamma', '_backward_gammaln', '_backward_gather_nd', '_backward_hard_sigmoid', '_backward_hypot', '_backward_hypot_scalar', '_backward_linalg_gelqf', '_backward_linalg_gemm', '_backward_linalg_gemm2', '_backward_linalg_potrf', '_backward_linalg_potri', '_backward_linalg_sumlogdiag', '_backward_linalg_syevd', '_backward_linalg_syrk', '_backward_linalg_trmm', '_backward_linalg_trsm', '_backward_linear_reg_out', '_backward_log', '_backward_log10', '_backward_log1p', '_backward_log2', '_backward_log_softmax', '_backward_logistic_reg_out', '_backward_mae_reg_out', '_backward_max', '_backward_maximum', '_backward_maximum_scalar', '_backward_mean', '_backward_min', '_backward_minimum', '_backward_minimum_scalar', '_backward_mod', '_backward_mod_scalar', '_backward_mul', '_backward_mul_scalar', '_backward_nanprod', '_backward_nansum', '_backward_norm', '_backward_pick', '_backward_power', '_backward_power_scalar', '_backward_prod', '_backward_radians', '_backward_rcbrt', '_backward_rdiv_scalar', '_backward_reciprocal', '_backward_relu', '_backward_repeat', '_backward_reverse', '_backward_rmod_scalar', '_backward_rpower_scalar', '_backward_rsqrt', '_backward_sample_multinomial', '_backward_sigmoid', '_backward_sign', '_backward_sin', '_backward_sinh', '_backward_slice', '_backward_slice_axis', '_backward_slice_like', '_backward_smooth_l1', '_backward_softmax', '_backward_softmax_cross_entropy', '_backward_softmin', '_backward_softsign', '_backward_sparse_retain', '_backward_sqrt', '_backward_square', '_backward_square_sum', '_backward_squeeze', '_backward_stack', '_backward_sub', '_backward_sum', '_backward_take', '_backward_tan', '_backward_tanh', '_backward_tile', '_backward_topk', '_backward_where', '_backward_while_loop', '_broadcast_backward', '_cond', '_copy', '_copyto', '_crop_assign', '_crop_assign_scalar', '_cvcopyMakeBorder', '_cvimdecode', '_cvimread', '_cvimresize', '_div', '_div_scalar', '_equal', '_equal_scalar', '_eye', '_foreach', '_full', '_grad_add', '_greater', '_greater_equal', '_greater_equal_scalar', '_greater_scalar', '_histogram', '_hypot', '_hypot_scalar', '_identity_with_attr_like_rhs', '_imdecode', '_lesser', '_lesser_equal', '_lesser_equal_scalar', '_lesser_scalar', '_logical_and', '_logical_and_scalar', '_logical_or', '_logical_or_scalar', '_logical_xor', '_logical_xor_scalar', '_maximum', '_maximum_scalar', '_minimum', '_minimum_scalar', '_minus', '_minus_scalar', '_mod', '_mod_scalar', '_mul', '_mul_scalar', '_not_equal', '_not_equal_scalar', '_onehot_encode', '_ones', '_plus', '_plus_scalar', '_power', '_power_scalar', '_ravel_multi_index', '_rdiv_scalar', '_rminus_scalar', '_rmod_scalar', '_rnn_param_concat', '_rpower_scalar', '_sample_exponential', '_sample_gamma', '_sample_generalized_negative_binomial', '_sample_multinomial', '_sample_negative_binomial', '_sample_normal', '_sample_poisson', '_sample_uniform', '_sample_unique_zipfian', '_scatter_elemwise_div', '_scatter_minus_scalar', '_scatter_plus_scalar', '_scatter_set_nd', '_set_value', '_shuffle', '_slice_assign', '_slice_assign_scalar', '_square_sum', '_sub', '_unravel_index', '_while_loop', '_zeros', '_zeros_without_dtype']