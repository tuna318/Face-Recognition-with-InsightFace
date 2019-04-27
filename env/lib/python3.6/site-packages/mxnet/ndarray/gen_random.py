# File content is auto-generated. Do not modify.
# pylint: skip-file
from ._internal import NDArrayBase
from ..base import _Null

def exponential(lam=_Null, shape=_Null, ctx=_Null, dtype=_Null, out=None, name=None, **kwargs):
    r"""Draw random samples from an exponential distribution.

    Samples are distributed according to an exponential distribution parametrized by *lambda* (rate).

    Example::

       exponential(lam=4, shape=(2,2)) = [[ 0.0097189 ,  0.08999364],
                                          [ 0.04146638,  0.31715935]]


    Defined in src/operator/random/sample_op.cc:L137

    Parameters
    ----------
    lam : float, optional, default=1
        Lambda parameter (rate) of the exponential distribution.
    shape : Shape(tuple), optional, default=[]
        Shape of the output.
    ctx : string, optional, default=''
        Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
    dtype : {'None', 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def exponential_like(data=None, lam=_Null, out=None, name=None, **kwargs):
    r"""Draw random samples from an exponential distribution according to the input array shape.

    Samples are distributed according to an exponential distribution parametrized by *lambda* (rate).

    Example::

       exponential(lam=4, data=ones(2,2)) = [[ 0.0097189 ,  0.08999364],
                                             [ 0.04146638,  0.31715935]]


    Defined in src/operator/random/sample_op.cc:L242

    Parameters
    ----------
    lam : float, optional, default=1
        Lambda parameter (rate) of the exponential distribution.
    data : NDArray
        The input

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def gamma(alpha=_Null, beta=_Null, shape=_Null, ctx=_Null, dtype=_Null, out=None, name=None, **kwargs):
    r"""Draw random samples from a gamma distribution.

    Samples are distributed according to a gamma distribution parametrized by *alpha* (shape) and *beta* (scale).

    Example::

       gamma(alpha=9, beta=0.5, shape=(2,2)) = [[ 7.10486984,  3.37695289],
                                                [ 3.91697288,  3.65933681]]


    Defined in src/operator/random/sample_op.cc:L125

    Parameters
    ----------
    alpha : float, optional, default=1
        Alpha parameter (shape) of the gamma distribution.
    beta : float, optional, default=1
        Beta parameter (scale) of the gamma distribution.
    shape : Shape(tuple), optional, default=[]
        Shape of the output.
    ctx : string, optional, default=''
        Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
    dtype : {'None', 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def gamma_like(data=None, alpha=_Null, beta=_Null, out=None, name=None, **kwargs):
    r"""Draw random samples from a gamma distribution according to the input array shape.

    Samples are distributed according to a gamma distribution parametrized by *alpha* (shape) and *beta* (scale).

    Example::

       gamma(alpha=9, beta=0.5, data=ones(2,2)) = [[ 7.10486984,  3.37695289],
                                                   [ 3.91697288,  3.65933681]]


    Defined in src/operator/random/sample_op.cc:L231

    Parameters
    ----------
    alpha : float, optional, default=1
        Alpha parameter (shape) of the gamma distribution.
    beta : float, optional, default=1
        Beta parameter (scale) of the gamma distribution.
    data : NDArray
        The input

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def generalized_negative_binomial(mu=_Null, alpha=_Null, shape=_Null, ctx=_Null, dtype=_Null, out=None, name=None, **kwargs):
    r"""Draw random samples from a generalized negative binomial distribution.

    Samples are distributed according to a generalized negative binomial distribution parametrized by
    *mu* (mean) and *alpha* (dispersion). *alpha* is defined as *1/k* where *k* is the failure limit of the
    number of unsuccessful experiments (generalized to real numbers).
    Samples will always be returned as a floating point data type.

    Example::

       generalized_negative_binomial(mu=2.0, alpha=0.3, shape=(2,2)) = [[ 2.,  1.],
                                                                        [ 6.,  4.]]


    Defined in src/operator/random/sample_op.cc:L179

    Parameters
    ----------
    mu : float, optional, default=1
        Mean of the negative binomial distribution.
    alpha : float, optional, default=1
        Alpha (dispersion) parameter of the negative binomial distribution.
    shape : Shape(tuple), optional, default=[]
        Shape of the output.
    ctx : string, optional, default=''
        Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
    dtype : {'None', 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def generalized_negative_binomial_like(data=None, mu=_Null, alpha=_Null, out=None, name=None, **kwargs):
    r"""Draw random samples from a generalized negative binomial distribution according to the
    input array shape.

    Samples are distributed according to a generalized negative binomial distribution parametrized by
    *mu* (mean) and *alpha* (dispersion). *alpha* is defined as *1/k* where *k* is the failure limit of the
    number of unsuccessful experiments (generalized to real numbers).
    Samples will always be returned as a floating point data type.

    Example::

       generalized_negative_binomial(mu=2.0, alpha=0.3, data=ones(2,2)) = [[ 2.,  1.],
                                                                           [ 6.,  4.]]


    Defined in src/operator/random/sample_op.cc:L283

    Parameters
    ----------
    mu : float, optional, default=1
        Mean of the negative binomial distribution.
    alpha : float, optional, default=1
        Alpha (dispersion) parameter of the negative binomial distribution.
    data : NDArray
        The input

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def negative_binomial(k=_Null, p=_Null, shape=_Null, ctx=_Null, dtype=_Null, out=None, name=None, **kwargs):
    r"""Draw random samples from a negative binomial distribution.

    Samples are distributed according to a negative binomial distribution parametrized by
    *k* (limit of unsuccessful experiments) and *p* (failure probability in each experiment).
    Samples will always be returned as a floating point data type.

    Example::

       negative_binomial(k=3, p=0.4, shape=(2,2)) = [[ 4.,  7.],
                                                     [ 2.,  5.]]


    Defined in src/operator/random/sample_op.cc:L164

    Parameters
    ----------
    k : int, optional, default='1'
        Limit of unsuccessful experiments.
    p : float, optional, default=1
        Failure probability in each experiment.
    shape : Shape(tuple), optional, default=[]
        Shape of the output.
    ctx : string, optional, default=''
        Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
    dtype : {'None', 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def negative_binomial_like(data=None, k=_Null, p=_Null, out=None, name=None, **kwargs):
    r"""Draw random samples from a negative binomial distribution according to the input array shape.

    Samples are distributed according to a negative binomial distribution parametrized by
    *k* (limit of unsuccessful experiments) and *p* (failure probability in each experiment).
    Samples will always be returned as a floating point data type.

    Example::

       negative_binomial(k=3, p=0.4, data=ones(2,2)) = [[ 4.,  7.],
                                                        [ 2.,  5.]]


    Defined in src/operator/random/sample_op.cc:L267

    Parameters
    ----------
    k : int, optional, default='1'
        Limit of unsuccessful experiments.
    p : float, optional, default=1
        Failure probability in each experiment.
    data : NDArray
        The input

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def normal(loc=_Null, scale=_Null, shape=_Null, ctx=_Null, dtype=_Null, out=None, name=None, **kwargs):
    r"""Draw random samples from a normal (Gaussian) distribution.

    .. note:: The existing alias ``normal`` is deprecated.

    Samples are distributed according to a normal distribution parametrized by *loc* (mean) and *scale*
    (standard deviation).

    Example::

       normal(loc=0, scale=1, shape=(2,2)) = [[ 1.89171135, -1.16881478],
                                              [-1.23474145,  1.55807114]]


    Defined in src/operator/random/sample_op.cc:L113

    Parameters
    ----------
    loc : float, optional, default=0
        Mean of the distribution.
    scale : float, optional, default=1
        Standard deviation of the distribution.
    shape : Shape(tuple), optional, default=[]
        Shape of the output.
    ctx : string, optional, default=''
        Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
    dtype : {'None', 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def normal_like(data=None, loc=_Null, scale=_Null, out=None, name=None, **kwargs):
    r"""Draw random samples from a normal (Gaussian) distribution according to the input array shape.

    Samples are distributed according to a normal distribution parametrized by *loc* (mean) and *scale*
    (standard deviation).

    Example::

       normal(loc=0, scale=1, data=ones(2,2)) = [[ 1.89171135, -1.16881478],
                                                 [-1.23474145,  1.55807114]]


    Defined in src/operator/random/sample_op.cc:L220

    Parameters
    ----------
    loc : float, optional, default=0
        Mean of the distribution.
    scale : float, optional, default=1
        Standard deviation of the distribution.
    data : NDArray
        The input

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def poisson(lam=_Null, shape=_Null, ctx=_Null, dtype=_Null, out=None, name=None, **kwargs):
    r"""Draw random samples from a Poisson distribution.

    Samples are distributed according to a Poisson distribution parametrized by *lambda* (rate).
    Samples will always be returned as a floating point data type.

    Example::

       poisson(lam=4, shape=(2,2)) = [[ 5.,  2.],
                                      [ 4.,  6.]]


    Defined in src/operator/random/sample_op.cc:L150

    Parameters
    ----------
    lam : float, optional, default=1
        Lambda parameter (rate) of the Poisson distribution.
    shape : Shape(tuple), optional, default=[]
        Shape of the output.
    ctx : string, optional, default=''
        Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
    dtype : {'None', 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def poisson_like(data=None, lam=_Null, out=None, name=None, **kwargs):
    r"""Draw random samples from a Poisson distribution according to the input array shape.

    Samples are distributed according to a Poisson distribution parametrized by *lambda* (rate).
    Samples will always be returned as a floating point data type.

    Example::

       poisson(lam=4, data=ones(2,2)) = [[ 5.,  2.],
                                         [ 4.,  6.]]


    Defined in src/operator/random/sample_op.cc:L254

    Parameters
    ----------
    lam : float, optional, default=1
        Lambda parameter (rate) of the Poisson distribution.
    data : NDArray
        The input

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def randint(low=_Null, high=_Null, shape=_Null, ctx=_Null, dtype=_Null, out=None, name=None, **kwargs):
    r"""Draw random samples from a discrete uniform distribution.

    Samples are uniformly distributed over the half-open interval *[low, high)*
    (includes *low*, but excludes *high*).

    Example::

       randint(low=0, high=5, shape=(2,2)) = [[ 0,  2],
                                              [ 3,  1]]



    Defined in src/operator/random/sample_op.cc:L193

    Parameters
    ----------
    low : , required
        Lower bound of the distribution.
    high : , required
        Upper bound of the distribution.
    shape : Shape(tuple), optional, default=[]
        Shape of the output.
    ctx : string, optional, default=''
        Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
    dtype : {'None', 'int32', 'int64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to int32 if not defined (dtype=None).

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def uniform(low=_Null, high=_Null, shape=_Null, ctx=_Null, dtype=_Null, out=None, name=None, **kwargs):
    r"""Draw random samples from a uniform distribution.

    .. note:: The existing alias ``uniform`` is deprecated.

    Samples are uniformly distributed over the half-open interval *[low, high)*
    (includes *low*, but excludes *high*).

    Example::

       uniform(low=0, high=1, shape=(2,2)) = [[ 0.60276335,  0.85794562],
                                              [ 0.54488319,  0.84725171]]



    Defined in src/operator/random/sample_op.cc:L96

    Parameters
    ----------
    low : float, optional, default=0
        Lower bound of the distribution.
    high : float, optional, default=1
        Upper bound of the distribution.
    shape : Shape(tuple), optional, default=[]
        Shape of the output.
    ctx : string, optional, default=''
        Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.
    dtype : {'None', 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to float32 if not defined (dtype=None).

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def uniform_like(data=None, low=_Null, high=_Null, out=None, name=None, **kwargs):
    r"""Draw random samples from a uniform distribution according to the input array shape.

    Samples are uniformly distributed over the half-open interval *[low, high)*
    (includes *low*, but excludes *high*).

    Example::

       uniform(low=0, high=1, data=ones(2,2)) = [[ 0.60276335,  0.85794562],
                                                 [ 0.54488319,  0.84725171]]



    Defined in src/operator/random/sample_op.cc:L208

    Parameters
    ----------
    low : float, optional, default=0
        Lower bound of the distribution.
    high : float, optional, default=1
        Upper bound of the distribution.
    data : NDArray
        The input

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

__all__ = ['exponential', 'exponential_like', 'gamma', 'gamma_like', 'generalized_negative_binomial', 'generalized_negative_binomial_like', 'negative_binomial', 'negative_binomial_like', 'normal', 'normal_like', 'poisson', 'poisson_like', 'randint', 'uniform', 'uniform_like']