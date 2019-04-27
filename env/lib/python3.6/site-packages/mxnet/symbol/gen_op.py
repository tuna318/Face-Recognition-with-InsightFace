# File content is auto-generated. Do not modify.
# pylint: skip-file
from ._internal import SymbolBase
from ..base import _Null

def Activation(data=None, act_type=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Applies an activation function element-wise to the input.

    The following activation functions are supported:

    - `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
    - `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`
    - `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`
    - `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`
    - `softsign`: :math:`y = \frac{x}{1 + abs(x)}`



    Defined in src/operator/nn/activation.cc:L168

    Parameters
    ----------
    data : Symbol
        The input array.
    act_type : {'relu', 'sigmoid', 'softrelu', 'softsign', 'tanh'}, required
        Activation function to be applied.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.

    Examples
    --------
    A one-hidden-layer MLP with ReLU activation:

    >>> data = Variable('data')
    >>> mlp = FullyConnected(data=data, num_hidden=128, name='proj')
    >>> mlp = Activation(data=mlp, act_type='relu', name='activation')
    >>> mlp = FullyConnected(data=mlp, num_hidden=10, name='mlp')
    >>> mlp
    <Symbol mlp>

    ReLU activation

    >>> test_suites = [
    ... ('relu', lambda x: np.maximum(x, 0)),
    ... ('sigmoid', lambda x: 1 / (1 + np.exp(-x))),
    ... ('tanh', lambda x: np.tanh(x)),
    ... ('softrelu', lambda x: np.log(1 + np.exp(x)))
    ... ]
    >>> x = test_utils.random_arrays((2, 3, 4))
    >>> for act_type, numpy_impl in test_suites:
    ... op = Activation(act_type=act_type, name='act')
    ... y = test_utils.simple_forward(op, act_data=x)
    ... y_np = numpy_impl(x)
    ... print('%s: %s' % (act_type, test_utils.almost_equal(y, y_np)))
    relu: True
    sigmoid: True
    tanh: True
    softrelu: True
    """
    return (0,)

def BatchNorm(data=None, gamma=None, beta=None, moving_mean=None, moving_var=None, eps=_Null, momentum=_Null, fix_gamma=_Null, use_global_stats=_Null, output_mean_var=_Null, axis=_Null, cudnn_off=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Batch normalization.

    Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
    well as offset ``beta``.

    Assume the input has more than one dimension and we normalize along axis 1.
    We first compute the mean and variance along this axis:

    .. math::

      data\_mean[i] = mean(data[:,i,:,...]) \\
      data\_var[i] = var(data[:,i,:,...])

    Then compute the normalized output, which has the same shape as input, as following:

    .. math::

      out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]

    Both *mean* and *var* returns a scalar by treating the input as a vector.

    Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
    have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
    the inverse of ``data_var``, which are needed for the backward pass. Note that gradient of these
    two outputs are blocked.

    Besides the inputs and the outputs, this operator accepts two auxiliary
    states, ``moving_mean`` and ``moving_var``, which are *k*-length
    vectors. They are global statistics for the whole dataset, which are updated
    by::

      moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
      moving_var = moving_var * momentum + data_var * (1 - momentum)

    If ``use_global_stats`` is set to be true, then ``moving_mean`` and
    ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
    the output. It is often used during inference.

    The parameter ``axis`` specifies which axis of the input shape denotes
    the 'channel' (separately normalized groups).  The default is 1.  Specifying -1 sets the channel
    axis to be the last item in the input shape.

    Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,
    then set ``gamma`` to 1 and its gradient to 0.

    .. Note::
      When ``fix_gamma`` is set to True, no sparse support is provided. If ``fix_gamma is`` set to False,
      the sparse tensors will fallback.



    Defined in src/operator/nn/batch_norm.cc:L574

    Parameters
    ----------
    data : Symbol
        Input data to batch normalization
    gamma : Symbol
        gamma array
    beta : Symbol
        beta array
    moving_mean : Symbol
        running mean of input
    moving_var : Symbol
        running variance of input
    eps : double, optional, default=0.001
        Epsilon to prevent div 0. Must be no less than CUDNN_BN_MIN_EPSILON defined in cudnn.h when using cudnn (usually 1e-5)
    momentum : float, optional, default=0.9
        Momentum for moving average
    fix_gamma : boolean, optional, default=1
        Fix gamma while training
    use_global_stats : boolean, optional, default=0
        Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator.
    output_mean_var : boolean, optional, default=0
        Output the mean and inverse std 
    axis : int, optional, default='1'
        Specify which shape axis the channel is specified
    cudnn_off : boolean, optional, default=0
        Do not select CUDNN operator, if available

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def BatchNorm_v1(data=None, gamma=None, beta=None, eps=_Null, momentum=_Null, fix_gamma=_Null, use_global_stats=_Null, output_mean_var=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Batch normalization.

    This operator is DEPRECATED. Perform BatchNorm on the input.

    Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
    well as offset ``beta``.

    Assume the input has more than one dimension and we normalize along axis 1.
    We first compute the mean and variance along this axis:

    .. math::

      data\_mean[i] = mean(data[:,i,:,...]) \\
      data\_var[i] = var(data[:,i,:,...])

    Then compute the normalized output, which has the same shape as input, as following:

    .. math::

      out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]

    Both *mean* and *var* returns a scalar by treating the input as a vector.

    Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
    have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
    ``data_var`` as well, which are needed for the backward pass.

    Besides the inputs and the outputs, this operator accepts two auxiliary
    states, ``moving_mean`` and ``moving_var``, which are *k*-length
    vectors. They are global statistics for the whole dataset, which are updated
    by::

      moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
      moving_var = moving_var * momentum + data_var * (1 - momentum)

    If ``use_global_stats`` is set to be true, then ``moving_mean`` and
    ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
    the output. It is often used during inference.

    Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,
    then set ``gamma`` to 1 and its gradient to 0.

    There's no sparse support for this operator, and it will exhibit problematic behavior if used with
    sparse tensors.



    Defined in src/operator/batch_norm_v1.cc:L95

    Parameters
    ----------
    data : Symbol
        Input data to batch normalization
    gamma : Symbol
        gamma array
    beta : Symbol
        beta array
    eps : float, optional, default=0.001
        Epsilon to prevent div 0
    momentum : float, optional, default=0.9
        Momentum for moving average
    fix_gamma : boolean, optional, default=1
        Fix gamma while training
    use_global_stats : boolean, optional, default=0
        Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator.
    output_mean_var : boolean, optional, default=0
        Output All,normal mean and var

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def BilinearSampler(data=None, grid=None, cudnn_off=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Applies bilinear sampling to input feature map.

    Bilinear Sampling is the key of  [NIPS2015] \"Spatial Transformer Networks\". The usage of the operator is very similar to remap function in OpenCV,
    except that the operator has the backward pass.

    Given :math:`data` and :math:`grid`, then the output is computed by

    .. math::
      x_{src} = grid[batch, 0, y_{dst}, x_{dst}] \\
      y_{src} = grid[batch, 1, y_{dst}, x_{dst}] \\
      output[batch, channel, y_{dst}, x_{dst}] = G(data[batch, channel, y_{src}, x_{src})

    :math:`x_{dst}`, :math:`y_{dst}` enumerate all spatial locations in :math:`output`, and :math:`G()` denotes the bilinear interpolation kernel.
    The out-boundary points will be padded with zeros.The shape of the output will be (data.shape[0], data.shape[1], grid.shape[2], grid.shape[3]).

    The operator assumes that :math:`data` has 'NCHW' layout and :math:`grid` has been normalized to [-1, 1].

    BilinearSampler often cooperates with GridGenerator which generates sampling grids for BilinearSampler.
    GridGenerator supports two kinds of transformation: ``affine`` and ``warp``.
    If users want to design a CustomOp to manipulate :math:`grid`, please firstly refer to the code of GridGenerator.

    Example 1::

      ## Zoom out data two times
      data = array([[[[1, 4, 3, 6],
                      [1, 8, 8, 9],
                      [0, 4, 1, 5],
                      [1, 0, 1, 3]]]])

      affine_matrix = array([[2, 0, 0],
                             [0, 2, 0]])

      affine_matrix = reshape(affine_matrix, shape=(1, 6))

      grid = GridGenerator(data=affine_matrix, transform_type='affine', target_shape=(4, 4))

      out = BilinearSampler(data, grid)

      out
      [[[[ 0,   0,     0,   0],
         [ 0,   3.5,   6.5, 0],
         [ 0,   1.25,  2.5, 0],
         [ 0,   0,     0,   0]]]


    Example 2::

      ## shift data horizontally by -1 pixel

      data = array([[[[1, 4, 3, 6],
                      [1, 8, 8, 9],
                      [0, 4, 1, 5],
                      [1, 0, 1, 3]]]])

      warp_maxtrix = array([[[[1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1]],
                             [[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]]]])

      grid = GridGenerator(data=warp_matrix, transform_type='warp')
      out = BilinearSampler(data, grid)

      out
      [[[[ 4,  3,  6,  0],
         [ 8,  8,  9,  0],
         [ 4,  1,  5,  0],
         [ 0,  1,  3,  0]]]


    Defined in src/operator/bilinear_sampler.cc:L256

    Parameters
    ----------
    data : Symbol
        Input data to the BilinearsamplerOp.
    grid : Symbol
        Input grid to the BilinearsamplerOp.grid has two channels: x_src, y_src
    cudnn_off : boolean or None, optional, default=None
        whether to turn cudnn off

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def BlockGrad(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Stops gradient computation.

    Stops the accumulated gradient of the inputs from flowing through this operator
    in the backward direction. In other words, this operator prevents the contribution
    of its inputs to be taken into account for computing gradients.

    Example::

      v1 = [1, 2]
      v2 = [0, 1]
      a = Variable('a')
      b = Variable('b')
      b_stop_grad = stop_gradient(3 * b)
      loss = MakeLoss(b_stop_grad + a)

      executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))
      executor.forward(is_train=True, a=v1, b=v2)
      executor.outputs
      [ 1.  5.]

      executor.backward()
      executor.grad_arrays
      [ 0.  0.]
      [ 1.  1.]



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L267

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

def CTCLoss(data=None, label=None, data_lengths=None, label_lengths=None, use_data_lengths=_Null, use_label_lengths=_Null, blank_label=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Connectionist Temporal Classification Loss.

    .. note:: The existing alias ``contrib_CTCLoss`` is deprecated.

    The shapes of the inputs and outputs:

    - **data**: `(sequence_length, batch_size, alphabet_size)`
    - **label**: `(batch_size, label_sequence_length)`
    - **out**: `(batch_size)`

    The `data` tensor consists of sequences of activation vectors (without applying softmax),
    with i-th channel in the last dimension corresponding to i-th label
    for i between 0 and alphabet_size-1 (i.e always 0-indexed).
    Alphabet size should include one additional value reserved for blank label.
    When `blank_label` is ``"first"``, the ``0``-th channel is be reserved for
    activation of blank label, or otherwise if it is "last", ``(alphabet_size-1)``-th channel should be
    reserved for blank label.

    ``label`` is an index matrix of integers. When `blank_label` is ``"first"``,
    the value 0 is then reserved for blank label, and should not be passed in this matrix. Otherwise,
    when `blank_label` is ``"last"``, the value `(alphabet_size-1)` is reserved for blank label.

    If a sequence of labels is shorter than *label_sequence_length*, use the special
    padding value at the end of the sequence to conform it to the correct
    length. The padding value is `0` when `blank_label` is ``"first"``, and `-1` otherwise.

    For example, suppose the vocabulary is `[a, b, c]`, and in one batch we have three sequences
    'ba', 'cbb', and 'abac'. When `blank_label` is ``"first"``, we can index the labels as
    `{'a': 1, 'b': 2, 'c': 3}`, and we reserve the 0-th channel for blank label in data tensor.
    The resulting `label` tensor should be padded to be::

      [[2, 1, 0, 0], [3, 2, 2, 0], [1, 2, 1, 3]]

    When `blank_label` is ``"last"``, we can index the labels as
    `{'a': 0, 'b': 1, 'c': 2}`, and we reserve the channel index 3 for blank label in data tensor.
    The resulting `label` tensor should be padded to be::

      [[1, 0, -1, -1], [2, 1, 1, -1], [0, 1, 0, 2]]

    ``out`` is a list of CTC loss values, one per example in the batch.

    See *Connectionist Temporal Classification: Labelling Unsegmented
    Sequence Data with Recurrent Neural Networks*, A. Graves *et al*. for more
    information on the definition and the algorithm.



    Defined in src/operator/nn/ctc_loss.cc:L100

    Parameters
    ----------
    data : Symbol
        Input ndarray
    label : Symbol
        Ground-truth labels for the loss.
    data_lengths : Symbol
        Lengths of data for each of the samples. Only required when use_data_lengths is true.
    label_lengths : Symbol
        Lengths of labels for each of the samples. Only required when use_label_lengths is true.
    use_data_lengths : boolean, optional, default=0
        Whether the data lenghts are decided by `data_lengths`. If false, the lengths are equal to the max sequence length.
    use_label_lengths : boolean, optional, default=0
        Whether the label lenghts are decided by `label_lengths`, or derived from `padding_mask`. If false, the lengths are derived from the first occurrence of the value of `padding_mask`. The value of `padding_mask` is ``0`` when first CTC label is reserved for blank, and ``-1`` when last label is reserved for blank. See `blank_label`.
    blank_label : {'first', 'last'},optional, default='first'
        Set the label that is reserved for blank label.If "first", 0-th label is reserved, and label values for tokens in the vocabulary are between ``1`` and ``alphabet_size-1``, and the padding mask is ``-1``. If "last", last label value ``alphabet_size-1`` is reserved for blank label instead, and label values for tokens in the vocabulary are between ``0`` and ``alphabet_size-2``, and the padding mask is ``0``.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def Cast(data=None, dtype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Casts all elements of the input to a new type.

    .. note:: ``Cast`` is deprecated. Use ``cast`` instead.

    Example::

       cast([0.9, 1.3], dtype='int32') = [0, 1]
       cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]
       cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L596

    Parameters
    ----------
    data : Symbol
        The input.
    dtype : {'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'}, required
        Output data type.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def Concat(*data, **kwargs):
    r"""Joins input arrays along a given axis.

    .. note:: `Concat` is deprecated. Use `concat` instead.

    The dimensions of the input arrays should be the same except the axis along
    which they will be concatenated.
    The dimension of the output array along the concatenated axis will be equal
    to the sum of the corresponding dimensions of the input arrays.

    The storage type of ``concat`` output depends on storage types of inputs

    - concat(csr, csr, ..., csr, dim=0) = csr
    - otherwise, ``concat`` generates output with default storage

    Example::

       x = [[1,1],[2,2]]
       y = [[3,3],[4,4],[5,5]]
       z = [[6,6], [7,7],[8,8]]

       concat(x,y,z,dim=0) = [[ 1.,  1.],
                              [ 2.,  2.],
                              [ 3.,  3.],
                              [ 4.,  4.],
                              [ 5.,  5.],
                              [ 6.,  6.],
                              [ 7.,  7.],
                              [ 8.,  8.]]

       Note that you cannot concat x,y,z along dimension 1 since dimension
       0 is not the same for all the input arrays.

       concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],
                             [ 4.,  4.,  7.,  7.],
                             [ 5.,  5.,  8.,  8.]]



    Defined in src/operator/nn/concat.cc:L368
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

    Examples
    --------
    Concat two (or more) inputs along a specific dimension:

    >>> a = Variable('a')
    >>> b = Variable('b')
    >>> c = Concat(a, b, dim=1, name='my-concat')
    >>> c
    <Symbol my-concat>
    >>> SymbolDoc.get_output_shape(c, a=(128, 10, 3, 3), b=(128, 15, 3, 3))
    {'my-concat_output': (128L, 25L, 3L, 3L)}

    Note the shape should be the same except on the dimension that is being
    concatenated.
    """
    return (0,)

def Convolution(data=None, weight=None, bias=None, kernel=_Null, stride=_Null, dilate=_Null, pad=_Null, num_filter=_Null, num_group=_Null, workspace=_Null, no_bias=_Null, cudnn_tune=_Null, cudnn_off=_Null, layout=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Compute *N*-D convolution on *(N+2)*-D input.

    In the 2-D convolution, given input data with shape *(batch_size,
    channel, height, width)*, the output is computed by

    .. math::

       out[n,i,:,:] = bias[i] + \sum_{j=0}^{channel} data[n,j,:,:] \star
       weight[i,j,:,:]

    where :math:`\star` is the 2-D cross-correlation operator.

    For general 2-D convolution, the shapes are

    - **data**: *(batch_size, channel, height, width)*
    - **weight**: *(num_filter, channel, kernel[0], kernel[1])*
    - **bias**: *(num_filter,)*
    - **out**: *(batch_size, num_filter, out_height, out_width)*.

    Define::

      f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1

    then we have::

      out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
      out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])

    If ``no_bias`` is set to be true, then the ``bias`` term is ignored.

    The default data ``layout`` is *NCHW*, namely *(batch_size, channel, height,
    width)*. We can choose other layouts such as *NWC*.

    If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
    evenly into *g* parts along the channel axis, and also evenly split ``weight``
    along the first dimension. Next compute the convolution on the *i*-th part of
    the data with the *i*-th weight part. The output is obtained by concatenating all
    the *g* results.

    1-D convolution does not have *height* dimension but only *width* in space.

    - **data**: *(batch_size, channel, width)*
    - **weight**: *(num_filter, channel, kernel[0])*
    - **bias**: *(num_filter,)*
    - **out**: *(batch_size, num_filter, out_width)*.

    3-D convolution adds an additional *depth* dimension besides *height* and
    *width*. The shapes are

    - **data**: *(batch_size, channel, depth, height, width)*
    - **weight**: *(num_filter, channel, kernel[0], kernel[1], kernel[2])*
    - **bias**: *(num_filter,)*
    - **out**: *(batch_size, num_filter, out_depth, out_height, out_width)*.

    Both ``weight`` and ``bias`` are learnable parameters.

    There are other options to tune the performance.

    - **cudnn_tune**: enable this option leads to higher startup time but may give
      faster speed. Options are

      - **off**: no tuning
      - **limited_workspace**:run test and pick the fastest algorithm that doesn't
        exceed workspace limit.
      - **fastest**: pick the fastest algorithm and ignore workspace limit.
      - **None** (default): the behavior is determined by environment variable
        ``MXNET_CUDNN_AUTOTUNE_DEFAULT``. 0 for off, 1 for limited workspace
        (default), 2 for fastest.

    - **workspace**: A large number leads to more (GPU) memory usage but may improve
      the performance.



    Defined in src/operator/nn/convolution.cc:L461

    Parameters
    ----------
    data : Symbol
        Input data to the ConvolutionOp.
    weight : Symbol
        Weight matrix.
    bias : Symbol
        Bias parameter.
    kernel : Shape(tuple), required
        Convolution kernel size: (w,), (h, w) or (d, h, w)
    stride : Shape(tuple), optional, default=[]
        Convolution stride: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.
    dilate : Shape(tuple), optional, default=[]
        Convolution dilate: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.
    pad : Shape(tuple), optional, default=[]
        Zero pad for convolution: (w,), (h, w) or (d, h, w). Defaults to no padding.
    num_filter : int (non-negative), required
        Convolution filter(channel) number
    num_group : int (non-negative), optional, default=1
        Number of group partitions.
    workspace : long (non-negative), optional, default=1024
        Maximum temporary workspace allowed (MB) in convolution.This parameter has two usages. When CUDNN is not used, it determines the effective batch size of the convolution kernel. When CUDNN is used, it controls the maximum temporary storage used for tuning the best CUDNN kernel when `limited_workspace` strategy is used.
    no_bias : boolean, optional, default=0
        Whether to disable bias parameter.
    cudnn_tune : {None, 'fastest', 'limited_workspace', 'off'},optional, default='None'
        Whether to pick convolution algo by running performance test.
    cudnn_off : boolean, optional, default=0
        Turn off cudnn for this layer.
    layout : {None, 'NCDHW', 'NCHW', 'NCW', 'NDHWC', 'NHWC'},optional, default='None'
        Set layout for input, output and weight. Empty for
        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.NHWC and NDHWC are only supported on GPU.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def Convolution_v1(data=None, weight=None, bias=None, kernel=_Null, stride=_Null, dilate=_Null, pad=_Null, num_filter=_Null, num_group=_Null, workspace=_Null, no_bias=_Null, cudnn_tune=_Null, cudnn_off=_Null, layout=_Null, name=None, attr=None, out=None, **kwargs):
    r"""This operator is DEPRECATED. Apply convolution to input then add a bias.

    Parameters
    ----------
    data : Symbol
        Input data to the ConvolutionV1Op.
    weight : Symbol
        Weight matrix.
    bias : Symbol
        Bias parameter.
    kernel : Shape(tuple), required
        convolution kernel size: (h, w) or (d, h, w)
    stride : Shape(tuple), optional, default=[]
        convolution stride: (h, w) or (d, h, w)
    dilate : Shape(tuple), optional, default=[]
        convolution dilate: (h, w) or (d, h, w)
    pad : Shape(tuple), optional, default=[]
        pad for convolution: (h, w) or (d, h, w)
    num_filter : int (non-negative), required
        convolution filter(channel) number
    num_group : int (non-negative), optional, default=1
        Number of group partitions. Equivalent to slicing input into num_group
        partitions, apply convolution on each, then concatenate the results
    workspace : long (non-negative), optional, default=1024
        Maximum temporary workspace allowed for convolution (MB).This parameter determines the effective batch size of the convolution kernel, which may be smaller than the given batch size. Also, the workspace will be automatically enlarged to make sure that we can run the kernel with batch_size=1
    no_bias : boolean, optional, default=0
        Whether to disable bias parameter.
    cudnn_tune : {None, 'fastest', 'limited_workspace', 'off'},optional, default='None'
        Whether to pick convolution algo by running performance test.
        Leads to higher startup time but may give faster speed. Options are:
        'off': no tuning
        'limited_workspace': run test and pick the fastest algorithm that doesn't exceed workspace limit.
        'fastest': pick the fastest algorithm and ignore workspace limit.
        If set to None (default), behavior is determined by environment
        variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,
        1 for limited workspace (default), 2 for fastest.
    cudnn_off : boolean, optional, default=0
        Turn off cudnn for this layer.
    layout : {None, 'NCDHW', 'NCHW', 'NDHWC', 'NHWC'},optional, default='None'
        Set layout for input, output and weight. Empty for
        default layout: NCHW for 2d and NCDHW for 3d.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def Correlation(data1=None, data2=None, kernel_size=_Null, max_displacement=_Null, stride1=_Null, stride2=_Null, pad_size=_Null, is_multiply=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Applies correlation to inputs.

    The correlation layer performs multiplicative patch comparisons between two feature maps.

    Given two multi-channel feature maps :math:`f_{1}, f_{2}`, with :math:`w`, :math:`h`, and :math:`c` being their width, height, and number of channels,
    the correlation layer lets the network compare each patch from :math:`f_{1}` with each patch from :math:`f_{2}`.

    For now we consider only a single comparison of two patches. The 'correlation' of two patches centered at :math:`x_{1}` in the first map and
    :math:`x_{2}` in the second map is then defined as:

    .. math::

       c(x_{1}, x_{2}) = \sum_{o \in [-k,k] \times [-k,k]} <f_{1}(x_{1} + o), f_{2}(x_{2} + o)>

    for a square patch of size :math:`K:=2k+1`.

    Note that the equation above is identical to one step of a convolution in neural networks, but instead of convolving data with a filter, it convolves data with other
    data. For this reason, it has no training weights.

    Computing :math:`c(x_{1}, x_{2})` involves :math:`c * K^{2}` multiplications. Comparing all patch combinations involves :math:`w^{2}*h^{2}` such computations.

    Given a maximum displacement :math:`d`, for each location :math:`x_{1}` it computes correlations :math:`c(x_{1}, x_{2})` only in a neighborhood of size :math:`D:=2d+1`,
    by limiting the range of :math:`x_{2}`. We use strides :math:`s_{1}, s_{2}`, to quantize :math:`x_{1}` globally and to quantize :math:`x_{2}` within the neighborhood
    centered around :math:`x_{1}`.

    The final output is defined by the following expression:

    .. math::
      out[n, q, i, j] = c(x_{i, j}, x_{q})

    where :math:`i` and :math:`j` enumerate spatial locations in :math:`f_{1}`, and :math:`q` denotes the :math:`q^{th}` neighborhood of :math:`x_{i,j}`.


    Defined in src/operator/correlation.cc:L198

    Parameters
    ----------
    data1 : Symbol
        Input data1 to the correlation.
    data2 : Symbol
        Input data2 to the correlation.
    kernel_size : int (non-negative), optional, default=1
        kernel size for Correlation must be an odd number
    max_displacement : int (non-negative), optional, default=1
        Max displacement of Correlation 
    stride1 : int (non-negative), optional, default=1
        stride1 quantize data1 globally
    stride2 : int (non-negative), optional, default=1
        stride2 quantize data2 within the neighborhood centered around data1
    pad_size : int (non-negative), optional, default=0
        pad for Correlation
    is_multiply : boolean, optional, default=1
        operation type is either multiplication or subduction

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def Crop(*data, **kwargs):
    r"""

    .. note:: `Crop` is deprecated. Use `slice` instead.

    Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w or
    with width and height of the second input symbol, i.e., with one input, we need h_w to
    specify the crop height and width, otherwise the second input symbol's size will be used


    Defined in src/operator/crop.cc:L50
    This function support variable length of positional input.

    Parameters
    ----------
    data : Symbol or Symbol[]
        Tensor or List of Tensors, the second input will be used as crop_like shape reference
    offset : Shape(tuple), optional, default=[0,0]
        crop offset coordinate: (y, x)
    h_w : Shape(tuple), optional, default=[0,0]
        crop height and width: (h, w)
    center_crop : boolean, optional, default=0
        If set to true, then it will use be the center_crop,or it will crop using the shape of crop_like

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def Custom(*data, **kwargs):
    r"""Apply a custom operator implemented in a frontend language (like Python).

    Custom operators should override required methods like `forward` and `backward`.
    The custom operator must be registered before it can be used.
    Please check the tutorial here: http://mxnet.io/faq/new_op.html.



    Defined in src/operator/custom/custom.cc:L547

    Parameters
    ----------
    data : Symbol[]
        Input data for the custom operator.
    op_type : string
        Name of the custom operator. This is the name that is passed to `mx.operator.register` to register the operator.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def Deconvolution(data=None, weight=None, bias=None, kernel=_Null, stride=_Null, dilate=_Null, pad=_Null, adj=_Null, target_shape=_Null, num_filter=_Null, num_group=_Null, workspace=_Null, no_bias=_Null, cudnn_tune=_Null, cudnn_off=_Null, layout=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Computes 1D or 2D transposed convolution (aka fractionally strided convolution) of the input tensor. This operation can be seen as the gradient of Convolution operation with respect to its input. Convolution usually reduces the size of the input. Transposed convolution works the other way, going from a smaller input to a larger output while preserving the connectivity pattern.

    Parameters
    ----------
    data : Symbol
        Input tensor to the deconvolution operation.
    weight : Symbol
        Weights representing the kernel.
    bias : Symbol
        Bias added to the result after the deconvolution operation.
    kernel : Shape(tuple), required
        Deconvolution kernel size: (w,), (h, w) or (d, h, w). This is same as the kernel size used for the corresponding convolution
    stride : Shape(tuple), optional, default=[]
        The stride used for the corresponding convolution: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.
    dilate : Shape(tuple), optional, default=[]
        Dilation factor for each dimension of the input: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.
    pad : Shape(tuple), optional, default=[]
        The amount of implicit zero padding added during convolution for each dimension of the input: (w,), (h, w) or (d, h, w). ``(kernel-1)/2`` is usually a good choice. If `target_shape` is set, `pad` will be ignored and a padding that will generate the target shape will be used. Defaults to no padding.
    adj : Shape(tuple), optional, default=[]
        Adjustment for output shape: (w,), (h, w) or (d, h, w). If `target_shape` is set, `adj` will be ignored and computed accordingly.
    target_shape : Shape(tuple), optional, default=[]
        Shape of the output tensor: (w,), (h, w) or (d, h, w).
    num_filter : int (non-negative), required
        Number of output filters.
    num_group : int (non-negative), optional, default=1
        Number of groups partition.
    workspace : long (non-negative), optional, default=512
        Maximum temporary workspace allowed (MB) in deconvolution.This parameter has two usages. When CUDNN is not used, it determines the effective batch size of the deconvolution kernel. When CUDNN is used, it controls the maximum temporary storage used for tuning the best CUDNN kernel when `limited_workspace` strategy is used.
    no_bias : boolean, optional, default=1
        Whether to disable bias parameter.
    cudnn_tune : {None, 'fastest', 'limited_workspace', 'off'},optional, default='None'
        Whether to pick convolution algorithm by running performance test.
    cudnn_off : boolean, optional, default=0
        Turn off cudnn for this layer.
    layout : {None, 'NCDHW', 'NCHW', 'NCW', 'NDHWC', 'NHWC'},optional, default='None'
        Set layout for input, output and weight. Empty for default layout, NCW for 1d, NCHW for 2d and NCDHW for 3d.NHWC and NDHWC are only supported on GPU.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def Dropout(data=None, p=_Null, mode=_Null, axes=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Applies dropout operation to input array.

    - During training, each element of the input is set to zero with probability p.
      The whole array is rescaled by :math:`1/(1-p)` to keep the expected
      sum of the input unchanged.

    - During testing, this operator does not change the input if mode is 'training'.
      If mode is 'always', the same computaion as during training will be applied.

    Example::

      random.seed(998)
      input_array = array([[3., 0.5,  -0.5,  2., 7.],
                          [2., -0.4,   7.,  3., 0.2]])
      a = symbol.Variable('a')
      dropout = symbol.Dropout(a, p = 0.2)
      executor = dropout.simple_bind(a = input_array.shape)

      ## If training
      executor.forward(is_train = True, a = input_array)
      executor.outputs
      [[ 3.75   0.625 -0.     2.5    8.75 ]
       [ 2.5   -0.5    8.75   3.75   0.   ]]

      ## If testing
      executor.forward(is_train = False, a = input_array)
      executor.outputs
      [[ 3.     0.5   -0.5    2.     7.   ]
       [ 2.    -0.4    7.     3.     0.2  ]]


    Defined in src/operator/nn/dropout.cc:L76

    Parameters
    ----------
    data : Symbol
        Input array to which dropout will be applied.
    p : float, optional, default=0.5
        Fraction of the input that gets dropped out during training time.
    mode : {'always', 'training'},optional, default='training'
        Whether to only turn on dropout during training or to also turn on for inference.
    axes : Shape(tuple), optional, default=[]
        Axes for variational dropout kernel.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.

    Examples
    --------
    Apply dropout to corrupt input as zero with probability 0.2:

    >>> data = Variable('data')
    >>> data_dp = Dropout(data=data, p=0.2)

    >>> shape = (100, 100)  # take larger shapes to be more statistical stable
    >>> x = np.ones(shape)
    >>> op = Dropout(p=0.5, name='dp')
    >>> # dropout is identity during testing
    >>> y = test_utils.simple_forward(op, dp_data=x, is_train=False)
    >>> test_utils.almost_equal(x, y)
    True
    >>> y = test_utils.simple_forward(op, dp_data=x, is_train=True)
    >>> # expectation is (approximately) unchanged
    >>> np.abs(x.mean() - y.mean()) < 0.1
    True
    >>> set(np.unique(y)) == set([0, 2])
    True
    """
    return (0,)

def ElementWiseSum(*args, **kwargs):
    r"""Adds all input arguments element-wise.

    .. math::
       add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n

    ``add_n`` is potentially more efficient than calling ``add`` by `n` times.

    The storage type of ``add_n`` output depends on storage types of inputs

    - add_n(row_sparse, row_sparse, ..) = row_sparse
    - add_n(default, csr, default) = default
    - add_n(any input combinations longer than 4 (>4) with at least one default type) = default
    - otherwise, ``add_n`` falls all inputs back to default storage and generates default storage



    Defined in src/operator/tensor/elemwise_sum.cc:L156
    This function support variable length of positional input.

    Parameters
    ----------
    args : Symbol[]
        Positional input arguments

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def Embedding(data=None, weight=None, input_dim=_Null, output_dim=_Null, dtype=_Null, sparse_grad=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Maps integer indices to vector representations (embeddings).

    This operator maps words to real-valued vectors in a high-dimensional space,
    called word embeddings. These embeddings can capture semantic and syntactic properties of the words.
    For example, it has been noted that in the learned embedding spaces, similar words tend
    to be close to each other and dissimilar words far apart.

    For an input array of shape (d1, ..., dK),
    the shape of an output array is (d1, ..., dK, output_dim).
    All the input values should be integers in the range [0, input_dim).

    If the input_dim is ip0 and output_dim is op0, then shape of the embedding weight matrix must be
    (ip0, op0).

    By default, if any index mentioned is too large, it is replaced by the index that addresses
    the last vector in an embedding matrix.

    Examples::

      input_dim = 4
      output_dim = 5

      // Each row in weight matrix y represents a word. So, y = (w0,w1,w2,w3)
      y = [[  0.,   1.,   2.,   3.,   4.],
           [  5.,   6.,   7.,   8.,   9.],
           [ 10.,  11.,  12.,  13.,  14.],
           [ 15.,  16.,  17.,  18.,  19.]]

      // Input array x represents n-grams(2-gram). So, x = [(w1,w3), (w0,w2)]
      x = [[ 1.,  3.],
           [ 0.,  2.]]

      // Mapped input x to its vector representation y.
      Embedding(x, y, 4, 5) = [[[  5.,   6.,   7.,   8.,   9.],
                                [ 15.,  16.,  17.,  18.,  19.]],

                               [[  0.,   1.,   2.,   3.,   4.],
                                [ 10.,  11.,  12.,  13.,  14.]]]


    The storage type of weight can be either row_sparse or default.

    .. Note::

        If "sparse_grad" is set to True, the storage type of gradient w.r.t weights will be
        "row_sparse". Only a subset of optimizers support sparse gradients, including SGD, AdaGrad
        and Adam. Note that by default lazy updates is turned on, which may perform differently
        from standard updates. For more details, please check the Optimization API at:
        https://mxnet.incubator.apache.org/api/python/optimization/optimization.html



    Defined in src/operator/tensor/indexing_op.cc:L519

    Parameters
    ----------
    data : Symbol
        The input array to the embedding operator.
    weight : Symbol
        The embedding weight matrix.
    input_dim : int, required
        Vocabulary size of the input indices.
    output_dim : int, required
        Dimension of the embedding vectors.
    dtype : {'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'},optional, default='float32'
        Data type of weight.
    sparse_grad : boolean, optional, default=0
        Compute row sparse gradient in the backward calculation. If set to True, the grad's storage type is row_sparse.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.

    Examples
    --------
    Assume we want to map the 26 English alphabet letters to 16-dimensional
    vectorial representations.

    >>> vocabulary_size = 26
    >>> embed_dim = 16
    >>> seq_len, batch_size = (10, 64)
    >>> input = Variable('letters')
    >>> op = Embedding(data=input, input_dim=vocabulary_size, output_dim=embed_dim,
    ...name='embed')
    >>> SymbolDoc.get_output_shape(op, letters=(seq_len, batch_size))
    {'embed_output': (10L, 64L, 16L)}

    >>> vocab_size, embed_dim = (26, 16)
    >>> batch_size = 12
    >>> word_vecs = test_utils.random_arrays((vocab_size, embed_dim))
    >>> op = Embedding(name='embed', input_dim=vocab_size, output_dim=embed_dim)
    >>> x = np.random.choice(vocab_size, batch_size)
    >>> y = test_utils.simple_forward(op, embed_data=x, embed_weight=word_vecs)
    >>> y_np = word_vecs[x]
    >>> test_utils.almost_equal(y, y_np)
    True
    """
    return (0,)

def Flatten(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Flattens the input array into a 2-D array by collapsing the higher dimensions.

    .. note:: `Flatten` is deprecated. Use `flatten` instead.

    For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation reshapes
    the input array into an output array of shape ``(d1, d2*...*dk)``.

    Note that the bahavior of this function is different from numpy.ndarray.flatten,
    which behaves similar to mxnet.ndarray.reshape((-1,)).

    Example::

        x = [[
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ],
        [    [1,2,3],
            [4,5,6],
            [7,8,9]
        ]],

        flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
           [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]



    Defined in src/operator/tensor/matrix_op.cc:L259

    Parameters
    ----------
    data : Symbol
        Input array.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.

    Examples
    --------
    Flatten is usually applied before `FullyConnected`, to reshape the 4D tensor
    produced by convolutional layers to 2D matrix:

    >>> data = Variable('data')  # say this is 4D from some conv/pool
    >>> flatten = Flatten(data=data, name='flat')  # now this is 2D
    >>> SymbolDoc.get_output_shape(flatten, data=(2, 3, 4, 5))
    {'flat_output': (2L, 60L)}

    >>> test_dims = [(2, 3, 4, 5), (2, 3), (2,)]
    >>> op = Flatten(name='flat')
    >>> for dims in test_dims:
    ... x = test_utils.random_arrays(dims)
    ... y = test_utils.simple_forward(op, flat_data=x)
    ... y_np = x.reshape((dims[0], np.prod(dims[1:]).astype('int32')))
    ... print('%s: %s' % (dims, test_utils.almost_equal(y, y_np)))
    (2, 3, 4, 5): True
    (2, 3): True
    (2,): True
    """
    return (0,)

def FullyConnected(data=None, weight=None, bias=None, num_hidden=_Null, no_bias=_Null, flatten=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Applies a linear transformation: :math:`Y = XW^T + b`.

    If ``flatten`` is set to be true, then the shapes are:

    - **data**: `(batch_size, x1, x2, ..., xn)`
    - **weight**: `(num_hidden, x1 * x2 * ... * xn)`
    - **bias**: `(num_hidden,)`
    - **out**: `(batch_size, num_hidden)`

    If ``flatten`` is set to be false, then the shapes are:

    - **data**: `(x1, x2, ..., xn, input_dim)`
    - **weight**: `(num_hidden, input_dim)`
    - **bias**: `(num_hidden,)`
    - **out**: `(x1, x2, ..., xn, num_hidden)`

    The learnable parameters include both ``weight`` and ``bias``.

    If ``no_bias`` is set to be true, then the ``bias`` term is ignored.

    .. Note::

        The sparse support for FullyConnected is limited to forward evaluation with `row_sparse`
        weight and bias, where the length of `weight.indices` and `bias.indices` must be equal
        to `num_hidden`. This could be useful for model inference with `row_sparse` weights
        trained with importance sampling or noise contrastive estimation.

        To compute linear transformation with 'csr' sparse data, sparse.dot is recommended instead
        of sparse.FullyConnected.



    Defined in src/operator/nn/fully_connected.cc:L271

    Parameters
    ----------
    data : Symbol
        Input data.
    weight : Symbol
        Weight matrix.
    bias : Symbol
        Bias parameter.
    num_hidden : int, required
        Number of hidden nodes of the output.
    no_bias : boolean, optional, default=0
        Whether to disable bias parameter.
    flatten : boolean, optional, default=1
        Whether to collapse all but the first axis of the input data tensor.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.

    Examples
    --------
    Construct a fully connected operator with target dimension 512.

    >>> data = Variable('data')  # or some constructed NN
    >>> op = FullyConnected(data=data,
    ... num_hidden=512,
    ... name='FC1')
    >>> op
    <Symbol FC1>
    >>> SymbolDoc.get_output_shape(op, data=(128, 100))
    {'FC1_output': (128L, 512L)}

    A simple 3-layer MLP with ReLU activation:

    >>> net = Variable('data')
    >>> for i, dim in enumerate([128, 64]):
    ... net = FullyConnected(data=net, num_hidden=dim, name='FC%d' % i)
    ... net = Activation(data=net, act_type='relu', name='ReLU%d' % i)
    >>> # 10-class predictor (e.g. MNIST)
    >>> net = FullyConnected(data=net, num_hidden=10, name='pred')
    >>> net
    <Symbol pred>

    >>> dim_in, dim_out = (3, 4)
    >>> x, w, b = test_utils.random_arrays((10, dim_in), (dim_out, dim_in), (dim_out,))
    >>> op = FullyConnected(num_hidden=dim_out, name='FC')
    >>> out = test_utils.simple_forward(op, FC_data=x, FC_weight=w, FC_bias=b)
    >>> # numpy implementation of FullyConnected
    >>> out_np = np.dot(x, w.T) + b
    >>> test_utils.almost_equal(out, out_np)
    True
    """
    return (0,)

def GridGenerator(data=None, transform_type=_Null, target_shape=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Generates 2D sampling grid for bilinear sampling.

    Parameters
    ----------
    data : Symbol
        Input data to the function.
    transform_type : {'affine', 'warp'}, required
        The type of transformation. For `affine`, input data should be an affine matrix of size (batch, 6). For `warp`, input data should be an optical flow of size (batch, 2, h, w).
    target_shape : Shape(tuple), optional, default=[0,0]
        Specifies the output shape (H, W). This is required if transformation type is `affine`. If transformation type is `warp`, this parameter is ignored.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def IdentityAttachKLSparseReg(data=None, sparseness_target=_Null, penalty=_Null, momentum=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Apply a sparse regularization to the output a sigmoid activation function.

    Parameters
    ----------
    data : Symbol
        Input data.
    sparseness_target : float, optional, default=0.1
        The sparseness target
    penalty : float, optional, default=0.001
        The tradeoff parameter for the sparseness penalty
    momentum : float, optional, default=0.9
        The momentum for running average

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def InstanceNorm(data=None, gamma=None, beta=None, eps=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Applies instance normalization to the n-dimensional input array.

    This operator takes an n-dimensional input array where (n>2) and normalizes
    the input using the following formula:

    .. math::

      out = \frac{x - mean[data]}{ \sqrt{Var[data]} + \epsilon} * gamma + beta

    This layer is similar to batch normalization layer (`BatchNorm`)
    with two differences: first, the normalization is
    carried out per example (instance), not over a batch. Second, the
    same normalization is applied both at test and train time. This
    operation is also known as `contrast normalization`.

    If the input data is of shape [batch, channel, spacial_dim1, spacial_dim2, ...],
    `gamma` and `beta` parameters must be vectors of shape [channel].

    This implementation is based on paper:

    .. [1] Instance Normalization: The Missing Ingredient for Fast Stylization,
       D. Ulyanov, A. Vedaldi, V. Lempitsky, 2016 (arXiv:1607.08022v2).

    Examples::

      // Input of shape (2,1,2)
      x = [[[ 1.1,  2.2]],
           [[ 3.3,  4.4]]]

      // gamma parameter of length 1
      gamma = [1.5]

      // beta parameter of length 1
      beta = [0.5]

      // Instance normalization is calculated with the above formula
      InstanceNorm(x,gamma,beta) = [[[-0.997527  ,  1.99752665]],
                                    [[-0.99752653,  1.99752724]]]



    Defined in src/operator/instance_norm.cc:L95

    Parameters
    ----------
    data : Symbol
        An n-dimensional input array (n > 2) of the form [batch, channel, spatial_dim1, spatial_dim2, ...].
    gamma : Symbol
        A vector of length 'channel', which multiplies the normalized input.
    beta : Symbol
        A vector of length 'channel', which is added to the product of the normalized input and the weight.
    eps : float, optional, default=0.001
        An `epsilon` parameter to prevent division by 0.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def L2Normalization(data=None, eps=_Null, mode=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Normalize the input array using the L2 norm.

    For 1-D NDArray, it computes::

      out = data / sqrt(sum(data ** 2) + eps)

    For N-D NDArray, if the input array has shape (N, N, ..., N),

    with ``mode`` = ``instance``, it normalizes each instance in the multidimensional
    array by its L2 norm.::

      for i in 0...N
        out[i,:,:,...,:] = data[i,:,:,...,:] / sqrt(sum(data[i,:,:,...,:] ** 2) + eps)

    with ``mode`` = ``channel``, it normalizes each channel in the array by its L2 norm.::

      for i in 0...N
        out[:,i,:,...,:] = data[:,i,:,...,:] / sqrt(sum(data[:,i,:,...,:] ** 2) + eps)

    with ``mode`` = ``spatial``, it normalizes the cross channel norm for each position
    in the array by its L2 norm.::

      for dim in 2...N
        for i in 0...N
          out[.....,i,...] = take(out, indices=i, axis=dim) / sqrt(sum(take(out, indices=i, axis=dim) ** 2) + eps)
              -dim-

    Example::

      x = [[[1,2],
            [3,4]],
           [[2,2],
            [5,6]]]

      L2Normalization(x, mode='instance')
      =[[[ 0.18257418  0.36514837]
         [ 0.54772252  0.73029673]]
        [[ 0.24077171  0.24077171]
         [ 0.60192931  0.72231513]]]

      L2Normalization(x, mode='channel')
      =[[[ 0.31622776  0.44721359]
         [ 0.94868326  0.89442718]]
        [[ 0.37139067  0.31622776]
         [ 0.92847669  0.94868326]]]

      L2Normalization(x, mode='spatial')
      =[[[ 0.44721359  0.89442718]
         [ 0.60000002  0.80000001]]
        [[ 0.70710677  0.70710677]
         [ 0.6401844   0.76822126]]]



    Defined in src/operator/l2_normalization.cc:L196

    Parameters
    ----------
    data : Symbol
        Input array to normalize.
    eps : float, optional, default=1e-10
        A small constant for numerical stability.
    mode : {'channel', 'instance', 'spatial'},optional, default='instance'
        Specify the dimension along which to compute L2 norm.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def LRN(data=None, alpha=_Null, beta=_Null, knorm=_Null, nsize=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Applies local response normalization to the input.

    The local response normalization layer performs "lateral inhibition" by normalizing
    over local input regions.

    If :math:`a_{x,y}^{i}` is the activity of a neuron computed by applying kernel :math:`i` at position
    :math:`(x, y)` and then applying the ReLU nonlinearity, the response-normalized
    activity :math:`b_{x,y}^{i}` is given by the expression:

    .. math::
       b_{x,y}^{i} = \frac{a_{x,y}^{i}}{\Bigg({k + \frac{\alpha}{n} \sum_{j=max(0, i-\frac{n}{2})}^{min(N-1, i+\frac{n}{2})} (a_{x,y}^{j})^{2}}\Bigg)^{\beta}}

    where the sum runs over :math:`n` "adjacent" kernel maps at the same spatial position, and :math:`N` is the total
    number of kernels in the layer.



    Defined in src/operator/nn/lrn.cc:L164

    Parameters
    ----------
    data : Symbol
        Input data to LRN
    alpha : float, optional, default=0.0001
        The variance scaling parameter :math:`lpha` in the LRN expression.
    beta : float, optional, default=0.75
        The power parameter :math:`eta` in the LRN expression.
    knorm : float, optional, default=2
        The parameter :math:`k` in the LRN expression.
    nsize : int (non-negative), required
        normalization window width in elements.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def LayerNorm(data=None, gamma=None, beta=None, axis=_Null, eps=_Null, output_mean_var=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Layer normalization.

    Normalizes the channels of the input tensor by mean and variance, and applies a scale ``gamma`` as
    well as offset ``beta``.

    Assume the input has more than one dimension and we normalize along axis 1.
    We first compute the mean and variance along this axis and then 
    compute the normalized output, which has the same shape as input, as following:

    .. math::

      out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis) + \epsilon}} * gamma + beta

    Both ``gamma`` and ``beta`` are learnable parameters.

    Unlike BatchNorm and InstanceNorm,  the *mean* and *var* are computed along the channel dimension.

    Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
    have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
    ``data_std``. Note that no gradient will be passed through these two outputs.

    The parameter ``axis`` specifies which axis of the input shape denotes
    the 'channel' (separately normalized groups).  The default is -1, which sets the channel
    axis to be the last item in the input shape.



    Defined in src/operator/nn/layer_norm.cc:L94

    Parameters
    ----------
    data : Symbol
        Input data to layer normalization
    gamma : Symbol
        gamma array
    beta : Symbol
        beta array
    axis : int, optional, default='-1'
        The axis to perform layer normalization. Usually, this should be be axis of the channel dimension. Negative values means indexing from right to left.
    eps : float, optional, default=1e-05
        An `epsilon` parameter to prevent division by 0.
    output_mean_var : boolean, optional, default=0
        Output the mean and std calculated along the given axis.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def LeakyReLU(data=None, gamma=None, act_type=_Null, slope=_Null, lower_bound=_Null, upper_bound=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Applies Leaky rectified linear unit activation element-wise to the input.

    Leaky ReLUs attempt to fix the "dying ReLU" problem by allowing a small `slope`
    when the input is negative and has a slope of one when input is positive.

    The following modified ReLU Activation functions are supported:

    - *elu*: Exponential Linear Unit. `y = x > 0 ? x : slope * (exp(x)-1)`
    - *selu*: Scaled Exponential Linear Unit. `y = lambda * (x > 0 ? x : alpha * (exp(x) - 1))` where
      *lambda = 1.0507009873554804934193349852946* and *alpha = 1.6732632423543772848170429916717*.
    - *leaky*: Leaky ReLU. `y = x > 0 ? x : slope * x`
    - *prelu*: Parametric ReLU. This is same as *leaky* except that `slope` is learnt during training.
    - *rrelu*: Randomized ReLU. same as *leaky* but the `slope` is uniformly and randomly chosen from
      *[lower_bound, upper_bound)* for training, while fixed to be
      *(lower_bound+upper_bound)/2* for inference.



    Defined in src/operator/leaky_relu.cc:L65

    Parameters
    ----------
    data : Symbol
        Input data to activation function.
    gamma : Symbol
        Slope parameter for PReLU. Only required when act_type is 'prelu'. It should be either a vector of size 1, or the same size as the second dimension of data.
    act_type : {'elu', 'leaky', 'prelu', 'rrelu', 'selu'},optional, default='leaky'
        Activation function to be applied.
    slope : float, optional, default=0.25
        Init slope for the activation. (For leaky and elu only)
    lower_bound : float, optional, default=0.125
        Lower bound of random slope. (For rrelu only)
    upper_bound : float, optional, default=0.334
        Upper bound of random slope. (For rrelu only)

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def LinearRegressionOutput(data=None, label=None, grad_scale=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Computes and optimizes for squared loss during backward propagation.
    Just outputs ``data`` during forward propagation.

    If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target value,
    then the squared loss estimated over :math:`n` samples is defined as

    :math:`\text{SquaredLoss}(\textbf{Y}, \hat{\textbf{Y}} ) = \frac{1}{n} \sum_{i=0}^{n-1} \lVert  \textbf{y}_i - \hat{\textbf{y}}_i  \rVert_2`

    .. note::
       Use the LinearRegressionOutput as the final output layer of a net.

    The storage type of ``label`` can be ``default`` or ``csr``

    - LinearRegressionOutput(default, default) = default
    - LinearRegressionOutput(default, csr) = default

    By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of regression outputs of a training example.
    The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.



    Defined in src/operator/regression_output.cc:L92

    Parameters
    ----------
    data : Symbol
        Input data to the function.
    label : Symbol
        Input label to the function.
    grad_scale : float, optional, default=1
        Scale the gradient by a float factor

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def LogisticRegressionOutput(data=None, label=None, grad_scale=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Applies a logistic function to the input.

    The logistic function, also known as the sigmoid function, is computed as
    :math:`\frac{1}{1+exp(-\textbf{x})}`.

    Commonly, the sigmoid is used to squash the real-valued output of a linear model
    :math:`wTx+b` into the [0,1] range so that it can be interpreted as a probability.
    It is suitable for binary classification or probability prediction tasks.

    .. note::
       Use the LogisticRegressionOutput as the final output layer of a net.

    The storage type of ``label`` can be ``default`` or ``csr``

    - LogisticRegressionOutput(default, default) = default
    - LogisticRegressionOutput(default, csr) = default

    The loss function used is the Binary Cross Entropy Loss:

    :math:`-{(y\log(p) + (1 - y)\log(1 - p))}`

    Where `y` is the ground truth probability of positive outcome for a given example, and `p` the probability predicted by the model. By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of regression outputs of a training example.
    The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.



    Defined in src/operator/regression_output.cc:L152

    Parameters
    ----------
    data : Symbol
        Input data to the function.
    label : Symbol
        Input label to the function.
    grad_scale : float, optional, default=1
        Scale the gradient by a float factor

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def MAERegressionOutput(data=None, label=None, grad_scale=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Computes mean absolute error of the input.

    MAE is a risk metric corresponding to the expected value of the absolute error.

    If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target value,
    then the mean absolute error (MAE) estimated over :math:`n` samples is defined as

    :math:`\text{MAE}(\textbf{Y}, \hat{\textbf{Y}} ) = \frac{1}{n} \sum_{i=0}^{n-1} \lVert \textbf{y}_i - \hat{\textbf{y}}_i \rVert_1`

    .. note::
       Use the MAERegressionOutput as the final output layer of a net.

    The storage type of ``label`` can be ``default`` or ``csr``

    - MAERegressionOutput(default, default) = default
    - MAERegressionOutput(default, csr) = default

    By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of regression outputs of a training example.
    The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.



    Defined in src/operator/regression_output.cc:L120

    Parameters
    ----------
    data : Symbol
        Input data to the function.
    label : Symbol
        Input label to the function.
    grad_scale : float, optional, default=1
        Scale the gradient by a float factor

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def MakeLoss(data=None, grad_scale=_Null, valid_thresh=_Null, normalization=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Make your own loss function in network construction.

    This operator accepts a customized loss function symbol as a terminal loss and
    the symbol should be an operator with no backward dependency.
    The output of this function is the gradient of loss with respect to the input data.

    For example, if you are a making a cross entropy loss function. Assume ``out`` is the
    predicted output and ``label`` is the true label, then the cross entropy can be defined as::

      cross_entropy = label * log(out) + (1 - label) * log(1 - out)
      loss = MakeLoss(cross_entropy)

    We will need to use ``MakeLoss`` when we are creating our own loss function or we want to
    combine multiple loss functions. Also we may want to stop some variables' gradients
    from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.

    In addition, we can give a scale to the loss by setting ``grad_scale``,
    so that the gradient of the loss will be rescaled in the backpropagation.

    .. note:: This operator should be used as a Symbol instead of NDArray.



    Defined in src/operator/make_loss.cc:L71

    Parameters
    ----------
    data : Symbol
        Input array.
    grad_scale : float, optional, default=1
        Gradient scale as a supplement to unary and binary operators
    valid_thresh : float, optional, default=0
        clip each element in the array to 0 when it is less than ``valid_thresh``. This is used when ``normalization`` is set to ``'valid'``.
    normalization : {'batch', 'null', 'valid'},optional, default='null'
        If this is set to null, the output gradient will not be normalized. If this is set to batch, the output gradient will be divided by the batch size. If this is set to valid, the output gradient will be divided by the number of valid input elements.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def Pad(data=None, mode=_Null, pad_width=_Null, constant_value=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Pads an input array with a constant or edge values of the array.

    .. note:: `Pad` is deprecated. Use `pad` instead.

    .. note:: Current implementation only supports 4D and 5D input arrays with padding applied
       only on axes 1, 2 and 3. Expects axes 4 and 5 in `pad_width` to be zero.

    This operation pads an input array with either a `constant_value` or edge values
    along each axis of the input array. The amount of padding is specified by `pad_width`.

    `pad_width` is a tuple of integer padding widths for each axis of the format
    ``(before_1, after_1, ... , before_N, after_N)``. The `pad_width` should be of length ``2*N``
    where ``N`` is the number of dimensions of the array.

    For dimension ``N`` of the input array, ``before_N`` and ``after_N`` indicates how many values
    to add before and after the elements of the array along dimension ``N``.
    The widths of the higher two dimensions ``before_1``, ``after_1``, ``before_2``,
    ``after_2`` must be 0.

    Example::

       x = [[[[  1.   2.   3.]
              [  4.   5.   6.]]

             [[  7.   8.   9.]
              [ 10.  11.  12.]]]


            [[[ 11.  12.  13.]
              [ 14.  15.  16.]]

             [[ 17.  18.  19.]
              [ 20.  21.  22.]]]]

       pad(x,mode="edge", pad_width=(0,0,0,0,1,1,1,1)) =

             [[[[  1.   1.   2.   3.   3.]
                [  1.   1.   2.   3.   3.]
                [  4.   4.   5.   6.   6.]
                [  4.   4.   5.   6.   6.]]

               [[  7.   7.   8.   9.   9.]
                [  7.   7.   8.   9.   9.]
                [ 10.  10.  11.  12.  12.]
                [ 10.  10.  11.  12.  12.]]]


              [[[ 11.  11.  12.  13.  13.]
                [ 11.  11.  12.  13.  13.]
                [ 14.  14.  15.  16.  16.]
                [ 14.  14.  15.  16.  16.]]

               [[ 17.  17.  18.  19.  19.]
                [ 17.  17.  18.  19.  19.]
                [ 20.  20.  21.  22.  22.]
                [ 20.  20.  21.  22.  22.]]]]

       pad(x, mode="constant", constant_value=0, pad_width=(0,0,0,0,1,1,1,1)) =

             [[[[  0.   0.   0.   0.   0.]
                [  0.   1.   2.   3.   0.]
                [  0.   4.   5.   6.   0.]
                [  0.   0.   0.   0.   0.]]

               [[  0.   0.   0.   0.   0.]
                [  0.   7.   8.   9.   0.]
                [  0.  10.  11.  12.   0.]
                [  0.   0.   0.   0.   0.]]]


              [[[  0.   0.   0.   0.   0.]
                [  0.  11.  12.  13.   0.]
                [  0.  14.  15.  16.   0.]
                [  0.   0.   0.   0.   0.]]

               [[  0.   0.   0.   0.   0.]
                [  0.  17.  18.  19.   0.]
                [  0.  20.  21.  22.   0.]
                [  0.   0.   0.   0.   0.]]]]




    Defined in src/operator/pad.cc:L766

    Parameters
    ----------
    data : Symbol
        An n-dimensional input array.
    mode : {'constant', 'edge', 'reflect'}, required
        Padding type to use. "constant" pads with `constant_value` "edge" pads using the edge values of the input array "reflect" pads by reflecting values with respect to the edges.
    pad_width : Shape(tuple), required
        Widths of the padding regions applied to the edges of each axis. It is a tuple of integer padding widths for each axis of the format ``(before_1, after_1, ... , before_N, after_N)``. It should be of length ``2*N`` where ``N`` is the number of dimensions of the array.This is equivalent to pad_width in numpy.pad, but flattened.
    constant_value : double, optional, default=0
        The value used for padding when `mode` is "constant".

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def Pooling(data=None, kernel=_Null, pool_type=_Null, global_pool=_Null, cudnn_off=_Null, pooling_convention=_Null, stride=_Null, pad=_Null, p_value=_Null, count_include_pad=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Performs pooling on the input.

    The shapes for 1-D pooling are

    - **data**: *(batch_size, channel, width)*,
    - **out**: *(batch_size, num_filter, out_width)*.

    The shapes for 2-D pooling are

    - **data**: *(batch_size, channel, height, width)*
    - **out**: *(batch_size, num_filter, out_height, out_width)*, with::

        out_height = f(height, kernel[0], pad[0], stride[0])
        out_width = f(width, kernel[1], pad[1], stride[1])

    The definition of *f* depends on ``pooling_convention``, which has two options:

    - **valid** (default)::

        f(x, k, p, s) = floor((x+2*p-k)/s)+1

    - **full**, which is compatible with Caffe::

        f(x, k, p, s) = ceil((x+2*p-k)/s)+1

    But ``global_pool`` is set to be true, then do a global pooling, namely reset
    ``kernel=(height, width)``.

    Three pooling options are supported by ``pool_type``:

    - **avg**: average pooling
    - **max**: max pooling
    - **sum**: sum pooling
    - **lp**: Lp pooling

    For 3-D pooling, an additional *depth* dimension is added before
    *height*. Namely the input data will have shape *(batch_size, channel, depth,
    height, width)*.

    Notes on Lp pooling:

    Lp pooling was first introduced by this paper: https://arxiv.org/pdf/1204.3968.pdf.
    L-1 pooling is simply sum pooling, while L-inf pooling is simply max pooling.
    We can see that Lp pooling stands between those two, in practice the most common value for p is 2.

    For each window ``X``, the mathematical expression for Lp pooling is:

    :math:`f(X) = \sqrt[p]{\sum_{x}^{X} x^p}`



    Defined in src/operator/nn/pooling.cc:L379

    Parameters
    ----------
    data : Symbol
        Input data to the pooling operator.
    kernel : Shape(tuple), optional, default=[]
        Pooling kernel size: (y, x) or (d, y, x)
    pool_type : {'avg', 'lp', 'max', 'sum'},optional, default='max'
        Pooling type to be applied.
    global_pool : boolean, optional, default=0
        Ignore kernel size, do global pooling based on current input feature map. 
    cudnn_off : boolean, optional, default=0
        Turn off cudnn pooling and use MXNet pooling operator. 
    pooling_convention : {'full', 'same', 'valid'},optional, default='valid'
        Pooling convention to be applied.
    stride : Shape(tuple), optional, default=[]
        Stride: for pooling (y, x) or (d, y, x). Defaults to 1 for each dimension.
    pad : Shape(tuple), optional, default=[]
        Pad for pooling: (y, x) or (d, y, x). Defaults to no padding.
    p_value : int or None, optional, default='None'
        Value of p for Lp pooling, can be 1 or 2, required for Lp Pooling.
    count_include_pad : boolean or None, optional, default=None
        Only used for AvgPool, specify whether to count padding elements for averagecalculation. For example, with a 5*5 kernel on a 3*3 corner of a image,the sum of the 9 valid elements will be divided by 25 if this is set to true,or it will be divided by 9 if this is set to false. Defaults to true.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def Pooling_v1(data=None, kernel=_Null, pool_type=_Null, global_pool=_Null, pooling_convention=_Null, stride=_Null, pad=_Null, name=None, attr=None, out=None, **kwargs):
    r"""This operator is DEPRECATED.
    Perform pooling on the input.

    The shapes for 2-D pooling is

    - **data**: *(batch_size, channel, height, width)*
    - **out**: *(batch_size, num_filter, out_height, out_width)*, with::

        out_height = f(height, kernel[0], pad[0], stride[0])
        out_width = f(width, kernel[1], pad[1], stride[1])

    The definition of *f* depends on ``pooling_convention``, which has two options:

    - **valid** (default)::

        f(x, k, p, s) = floor((x+2*p-k)/s)+1

    - **full**, which is compatible with Caffe::

        f(x, k, p, s) = ceil((x+2*p-k)/s)+1

    But ``global_pool`` is set to be true, then do a global pooling, namely reset
    ``kernel=(height, width)``.

    Three pooling options are supported by ``pool_type``:

    - **avg**: average pooling
    - **max**: max pooling
    - **sum**: sum pooling

    1-D pooling is special case of 2-D pooling with *weight=1* and
    *kernel[1]=1*.

    For 3-D pooling, an additional *depth* dimension is added before
    *height*. Namely the input data will have shape *(batch_size, channel, depth,
    height, width)*.



    Defined in src/operator/pooling_v1.cc:L104

    Parameters
    ----------
    data : Symbol
        Input data to the pooling operator.
    kernel : Shape(tuple), optional, default=[]
        pooling kernel size: (y, x) or (d, y, x)
    pool_type : {'avg', 'max', 'sum'},optional, default='max'
        Pooling type to be applied.
    global_pool : boolean, optional, default=0
        Ignore kernel size, do global pooling based on current input feature map. 
    pooling_convention : {'full', 'valid'},optional, default='valid'
        Pooling convention to be applied.
    stride : Shape(tuple), optional, default=[]
        stride: for pooling (y, x) or (d, y, x)
    pad : Shape(tuple), optional, default=[]
        pad for pooling: (y, x) or (d, y, x)

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def RNN(data=None, parameters=None, state=None, state_cell=None, state_size=_Null, num_layers=_Null, bidirectional=_Null, mode=_Null, p=_Null, state_outputs=_Null, projection_size=_Null, lstm_state_clip_min=_Null, lstm_state_clip_max=_Null, lstm_state_clip_nan=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Applies recurrent layers to input data. Currently, vanilla RNN, LSTM and GRU are
    implemented, with both multi-layer and bidirectional support.

    When the input data is of type float32 and the environment variables MXNET_CUDA_ALLOW_TENSOR_CORE
    and MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION are set to 1, this operator will try to use
    pseudo-float16 precision (float32 math with float16 I/O) precision in order to use
    Tensor Cores on suitable NVIDIA GPUs. This can sometimes give significant speedups.

    **Vanilla RNN**

    Applies a single-gate recurrent layer to input X. Two kinds of activation function are supported:
    ReLU and Tanh.

    With ReLU activation function:

    .. math::
        h_t = relu(W_{ih} * x_t + b_{ih}  +  W_{hh} * h_{(t-1)} + b_{hh})

    With Tanh activtion function:

    .. math::
        h_t = \tanh(W_{ih} * x_t + b_{ih}  +  W_{hh} * h_{(t-1)} + b_{hh})

    Reference paper: Finding structure in time - Elman, 1988.
    https://crl.ucsd.edu/~elman/Papers/fsit.pdf

    **LSTM**

    Long Short-Term Memory - Hochreiter, 1997. http://www.bioinf.jku.at/publications/older/2604.pdf

    .. math::
      \begin{array}{ll}
                i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
                f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
                g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\
                o_t = \mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
                c_t = f_t * c_{(t-1)} + i_t * g_t \\
                h_t = o_t * \tanh(c_t)
                \end{array}

    **GRU**

    Gated Recurrent Unit - Cho et al. 2014. http://arxiv.org/abs/1406.1078

    The definition of GRU here is slightly different from paper but compatible with CUDNN.

    .. math::
      \begin{array}{ll}
                r_t = \mathrm{sigmoid}(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
                z_t = \mathrm{sigmoid}(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
                n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
                h_t = (1 - z_t) * n_t + z_t * h_{(t-1)} \\
                \end{array}

    Parameters
    ----------
    data : Symbol
        Input data to RNN
    parameters : Symbol
        Vector of all RNN trainable parameters concatenated
    state : Symbol
        initial hidden state of the RNN
    state_cell : Symbol
        initial cell state for LSTM networks (only for LSTM)
    state_size : int (non-negative), required
        size of the state for each layer
    num_layers : int (non-negative), required
        number of stacked layers
    bidirectional : boolean, optional, default=0
        whether to use bidirectional recurrent layers
    mode : {'gru', 'lstm', 'rnn_relu', 'rnn_tanh'}, required
        the type of RNN to compute
    p : float, optional, default=0
        drop rate of the dropout on the outputs of each RNN layer, except the last layer.
    state_outputs : boolean, optional, default=0
        Whether to have the states as symbol outputs.
    projection_size : int or None, optional, default='None'
        size of project size
    lstm_state_clip_min : double or None, optional, default=None
        Minimum clip value of LSTM states. This option must be used together with lstm_state_clip_max.
    lstm_state_clip_max : double or None, optional, default=None
        Maximum clip value of LSTM states. This option must be used together with lstm_state_clip_min.
    lstm_state_clip_nan : boolean, optional, default=0
        Whether to stop NaN from propagating in state by clipping it to min/max. If clipping range is not specified, this option is ignored.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def ROIPooling(data=None, rois=None, pooled_size=_Null, spatial_scale=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Performs region of interest(ROI) pooling on the input array.

    ROI pooling is a variant of a max pooling layer, in which the output size is fixed and
    region of interest is a parameter. Its purpose is to perform max pooling on the inputs
    of non-uniform sizes to obtain fixed-size feature maps. ROI pooling is a neural-net
    layer mostly used in training a `Fast R-CNN` network for object detection.

    This operator takes a 4D feature map as an input array and region proposals as `rois`,
    then it pools over sub-regions of input and produces a fixed-sized output array
    regardless of the ROI size.

    To crop the feature map accordingly, you can resize the bounding box coordinates
    by changing the parameters `rois` and `spatial_scale`.

    The cropped feature maps are pooled by standard max pooling operation to a fixed size output
    indicated by a `pooled_size` parameter. batch_size will change to the number of region
    bounding boxes after `ROIPooling`.

    The size of each region of interest doesn't have to be perfectly divisible by
    the number of pooling sections(`pooled_size`).

    Example::

      x = [[[[  0.,   1.,   2.,   3.,   4.,   5.],
             [  6.,   7.,   8.,   9.,  10.,  11.],
             [ 12.,  13.,  14.,  15.,  16.,  17.],
             [ 18.,  19.,  20.,  21.,  22.,  23.],
             [ 24.,  25.,  26.,  27.,  28.,  29.],
             [ 30.,  31.,  32.,  33.,  34.,  35.],
             [ 36.,  37.,  38.,  39.,  40.,  41.],
             [ 42.,  43.,  44.,  45.,  46.,  47.]]]]

      // region of interest i.e. bounding box coordinates.
      y = [[0,0,0,4,4]]

      // returns array of shape (2,2) according to the given roi with max pooling.
      ROIPooling(x, y, (2,2), 1.0) = [[[[ 14.,  16.],
                                        [ 26.,  28.]]]]

      // region of interest is changed due to the change in `spacial_scale` parameter.
      ROIPooling(x, y, (2,2), 0.7) = [[[[  7.,   9.],
                                        [ 19.,  21.]]]]



    Defined in src/operator/roi_pooling.cc:L295

    Parameters
    ----------
    data : Symbol
        The input array to the pooling operator,  a 4D Feature maps 
    rois : Symbol
        Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]], where (x1, y1) and (x2, y2) are top left and bottom right corners of designated region of interest. `batch_index` indicates the index of corresponding image in the input array
    pooled_size : Shape(tuple), required
        ROI pooling output shape (h,w) 
    spatial_scale : float, required
        Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal of total stride in convolutional layers

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def Reshape(data=None, shape=_Null, reverse=_Null, target_shape=_Null, keep_highest=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Reshapes the input array.

    .. note:: ``Reshape`` is deprecated, use ``reshape``

    Given an array and a shape, this function returns a copy of the array in the new shape.
    The shape is a tuple of integers such as (2,3,4). The size of the new shape should be same as the size of the input array.

    Example::

      reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]

    Some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}. The significance of each is explained below:

    - ``0``  copy this dimension from the input to the output shape.

      Example::

      - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)
      - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)

    - ``-1`` infers the dimension of the output shape by using the remainder of the input dimensions
      keeping the size of the new array same as that of the input array.
      At most one dimension of shape can be -1.

      Example::

      - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)
      - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)
      - input shape = (2,3,4), shape=(-1,), output shape = (24,)

    - ``-2`` copy all/remainder of the input dimensions to the output shape.

      Example::

      - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)
      - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)
      - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)

    - ``-3`` use the product of two consecutive dimensions of the input shape as the output dimension.

      Example::

      - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)
      - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)
      - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)
      - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)

    - ``-4`` split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).

      Example::

      - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)
      - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)

    If the argument `reverse` is set to 1, then the special values are inferred from right to left.

      Example::

      - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape would be (40,5)
      - with reverse=1, output shape will be (50,4).



    Defined in src/operator/tensor/matrix_op.cc:L169

    Parameters
    ----------
    data : Symbol
        Input data to reshape.
    shape : Shape(tuple), optional, default=[]
        The target shape
    reverse : boolean, optional, default=0
        If true then the special values are inferred from right to left
    target_shape : Shape(tuple), optional, default=[]
        (Deprecated! Use ``shape`` instead.) Target new shape. One and only one dim can be 0, in which case it will be inferred from the rest of dims
    keep_highest : boolean, optional, default=0
        (Deprecated! Use ``shape`` instead.) Whether keep the highest dim unchanged.If set to true, then the first dim in target_shape is ignored,and always fixed as input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def SVMOutput(data=None, label=None, margin=_Null, regularization_coefficient=_Null, use_linear=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Computes support vector machine based transformation of the input.

    This tutorial demonstrates using SVM as output layer for classification instead of softmax:
    https://github.com/dmlc/mxnet/tree/master/example/svm_mnist.



    Parameters
    ----------
    data : Symbol
        Input data for SVM transformation.
    label : Symbol
        Class label for the input data.
    margin : float, optional, default=1
        The loss function penalizes outputs that lie outside this margin. Default margin is 1.
    regularization_coefficient : float, optional, default=1
        Regularization parameter for the SVM. This balances the tradeoff between coefficient size and error.
    use_linear : boolean, optional, default=0
        Whether to use L1-SVM objective. L2-SVM objective is used by default.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def SequenceLast(data=None, sequence_length=None, use_sequence_length=_Null, axis=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Takes the last element of a sequence.

    This function takes an n-dimensional input array of the form
    [max_sequence_length, batch_size, other_feature_dims] and returns a (n-1)-dimensional array
    of the form [batch_size, other_feature_dims].

    Parameter `sequence_length` is used to handle variable-length sequences. `sequence_length` should be
    an input array of positive ints of dimension [batch_size]. To use this parameter,
    set `use_sequence_length` to `True`, otherwise each example in the batch is assumed
    to have the max sequence length.

    .. note:: Alternatively, you can also use `take` operator.

    Example::

       x = [[[  1.,   2.,   3.],
             [  4.,   5.,   6.],
             [  7.,   8.,   9.]],

            [[ 10.,   11.,   12.],
             [ 13.,   14.,   15.],
             [ 16.,   17.,   18.]],

            [[  19.,   20.,   21.],
             [  22.,   23.,   24.],
             [  25.,   26.,   27.]]]

       // returns last sequence when sequence_length parameter is not used
       SequenceLast(x) = [[  19.,   20.,   21.],
                          [  22.,   23.,   24.],
                          [  25.,   26.,   27.]]

       // sequence_length is used
       SequenceLast(x, sequence_length=[1,1,1], use_sequence_length=True) =
                [[  1.,   2.,   3.],
                 [  4.,   5.,   6.],
                 [  7.,   8.,   9.]]

       // sequence_length is used
       SequenceLast(x, sequence_length=[1,2,3], use_sequence_length=True) =
                [[  1.,    2.,   3.],
                 [  13.,  14.,  15.],
                 [  25.,  26.,  27.]]



    Defined in src/operator/sequence_last.cc:L92

    Parameters
    ----------
    data : Symbol
        n-dimensional input array of the form [max_sequence_length, batch_size, other_feature_dims] where n>2
    sequence_length : Symbol
        vector of sequence lengths of the form [batch_size]
    use_sequence_length : boolean, optional, default=0
        If set to true, this layer takes in an extra input parameter `sequence_length` to specify variable length sequence
    axis : int, optional, default='0'
        The sequence axis. Only values of 0 and 1 are currently supported.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def SequenceMask(data=None, sequence_length=None, use_sequence_length=_Null, value=_Null, axis=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Sets all elements outside the sequence to a constant value.

    This function takes an n-dimensional input array of the form
    [max_sequence_length, batch_size, other_feature_dims] and returns an array of the same shape.

    Parameter `sequence_length` is used to handle variable-length sequences. `sequence_length`
    should be an input array of positive ints of dimension [batch_size].
    To use this parameter, set `use_sequence_length` to `True`,
    otherwise each example in the batch is assumed to have the max sequence length and
    this operator works as the `identity` operator.

    Example::

       x = [[[  1.,   2.,   3.],
             [  4.,   5.,   6.]],

            [[  7.,   8.,   9.],
             [ 10.,  11.,  12.]],

            [[ 13.,  14.,   15.],
             [ 16.,  17.,   18.]]]

       // Batch 1
       B1 = [[  1.,   2.,   3.],
             [  7.,   8.,   9.],
             [ 13.,  14.,  15.]]

       // Batch 2
       B2 = [[  4.,   5.,   6.],
             [ 10.,  11.,  12.],
             [ 16.,  17.,  18.]]

       // works as identity operator when sequence_length parameter is not used
       SequenceMask(x) = [[[  1.,   2.,   3.],
                           [  4.,   5.,   6.]],

                          [[  7.,   8.,   9.],
                           [ 10.,  11.,  12.]],

                          [[ 13.,  14.,   15.],
                           [ 16.,  17.,   18.]]]

       // sequence_length [1,1] means 1 of each batch will be kept
       // and other rows are masked with default mask value = 0
       SequenceMask(x, sequence_length=[1,1], use_sequence_length=True) =
                    [[[  1.,   2.,   3.],
                      [  4.,   5.,   6.]],

                     [[  0.,   0.,   0.],
                      [  0.,   0.,   0.]],

                     [[  0.,   0.,   0.],
                      [  0.,   0.,   0.]]]

       // sequence_length [2,3] means 2 of batch B1 and 3 of batch B2 will be kept
       // and other rows are masked with value = 1
       SequenceMask(x, sequence_length=[2,3], use_sequence_length=True, value=1) =
                    [[[  1.,   2.,   3.],
                      [  4.,   5.,   6.]],

                     [[  7.,   8.,   9.],
                      [  10.,  11.,  12.]],

                     [[   1.,   1.,   1.],
                      [  16.,  17.,  18.]]]



    Defined in src/operator/sequence_mask.cc:L114

    Parameters
    ----------
    data : Symbol
        n-dimensional input array of the form [max_sequence_length, batch_size, other_feature_dims] where n>2
    sequence_length : Symbol
        vector of sequence lengths of the form [batch_size]
    use_sequence_length : boolean, optional, default=0
        If set to true, this layer takes in an extra input parameter `sequence_length` to specify variable length sequence
    value : float, optional, default=0
        The value to be used as a mask.
    axis : int, optional, default='0'
        The sequence axis. Only values of 0 and 1 are currently supported.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def SequenceReverse(data=None, sequence_length=None, use_sequence_length=_Null, axis=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Reverses the elements of each sequence.

    This function takes an n-dimensional input array of the form [max_sequence_length, batch_size, other_feature_dims]
    and returns an array of the same shape.

    Parameter `sequence_length` is used to handle variable-length sequences.
    `sequence_length` should be an input array of positive ints of dimension [batch_size].
    To use this parameter, set `use_sequence_length` to `True`,
    otherwise each example in the batch is assumed to have the max sequence length.

    Example::

       x = [[[  1.,   2.,   3.],
             [  4.,   5.,   6.]],

            [[  7.,   8.,   9.],
             [ 10.,  11.,  12.]],

            [[ 13.,  14.,   15.],
             [ 16.,  17.,   18.]]]

       // Batch 1
       B1 = [[  1.,   2.,   3.],
             [  7.,   8.,   9.],
             [ 13.,  14.,  15.]]

       // Batch 2
       B2 = [[  4.,   5.,   6.],
             [ 10.,  11.,  12.],
             [ 16.,  17.,  18.]]

       // returns reverse sequence when sequence_length parameter is not used
       SequenceReverse(x) = [[[ 13.,  14.,   15.],
                              [ 16.,  17.,   18.]],

                             [[  7.,   8.,   9.],
                              [ 10.,  11.,  12.]],

                             [[  1.,   2.,   3.],
                              [  4.,   5.,   6.]]]

       // sequence_length [2,2] means 2 rows of
       // both batch B1 and B2 will be reversed.
       SequenceReverse(x, sequence_length=[2,2], use_sequence_length=True) =
                         [[[  7.,   8.,   9.],
                           [ 10.,  11.,  12.]],

                          [[  1.,   2.,   3.],
                           [  4.,   5.,   6.]],

                          [[ 13.,  14.,   15.],
                           [ 16.,  17.,   18.]]]

       // sequence_length [2,3] means 2 of batch B2 and 3 of batch B3
       // will be reversed.
       SequenceReverse(x, sequence_length=[2,3], use_sequence_length=True) =
                        [[[  7.,   8.,   9.],
                          [ 16.,  17.,  18.]],

                         [[  1.,   2.,   3.],
                          [ 10.,  11.,  12.]],

                         [[ 13.,  14,   15.],
                          [  4.,   5.,   6.]]]



    Defined in src/operator/sequence_reverse.cc:L113

    Parameters
    ----------
    data : Symbol
        n-dimensional input array of the form [max_sequence_length, batch_size, other dims] where n>2 
    sequence_length : Symbol
        vector of sequence lengths of the form [batch_size]
    use_sequence_length : boolean, optional, default=0
        If set to true, this layer takes in an extra input parameter `sequence_length` to specify variable length sequence
    axis : int, optional, default='0'
        The sequence axis. Only 0 is currently supported.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def SliceChannel(data=None, num_outputs=_Null, axis=_Null, squeeze_axis=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Splits an array along a particular axis into multiple sub-arrays.

    .. note:: ``SliceChannel`` is deprecated. Use ``split`` instead.

    **Note** that `num_outputs` should evenly divide the length of the axis
    along which to split the array.

    Example::

       x  = [[[ 1.]
              [ 2.]]
             [[ 3.]
              [ 4.]]
             [[ 5.]
              [ 6.]]]
       x.shape = (3, 2, 1)

       y = split(x, axis=1, num_outputs=2) // a list of 2 arrays with shape (3, 1, 1)
       y = [[[ 1.]]
            [[ 3.]]
            [[ 5.]]]

           [[[ 2.]]
            [[ 4.]]
            [[ 6.]]]

       y[0].shape = (3, 1, 1)

       z = split(x, axis=0, num_outputs=3) // a list of 3 arrays with shape (1, 2, 1)
       z = [[[ 1.]
             [ 2.]]]

           [[[ 3.]
             [ 4.]]]

           [[[ 5.]
             [ 6.]]]

       z[0].shape = (1, 2, 1)

    `squeeze_axis=1` removes the axis with length 1 from the shapes of the output arrays.
    **Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 only
    along the `axis` which it is split.
    Also `squeeze_axis` can be set to true only if ``input.shape[axis] == num_outputs``.

    Example::

       z = split(x, axis=0, num_outputs=3, squeeze_axis=1) // a list of 3 arrays with shape (2, 1)
       z = [[ 1.]
            [ 2.]]

           [[ 3.]
            [ 4.]]

           [[ 5.]
            [ 6.]]
       z[0].shape = (2 ,1 )



    Defined in src/operator/slice_channel.cc:L107

    Parameters
    ----------
    data : Symbol
        The input
    num_outputs : int, required
        Number of splits. Note that this should evenly divide the length of the `axis`.
    axis : int, optional, default='1'
        Axis along which to split.
    squeeze_axis : boolean, optional, default=0
        If true, Removes the axis with length 1 from the shapes of the output arrays. **Note** that setting `squeeze_axis` to ``true`` removes axis with length 1 only along the `axis` which it is split. Also `squeeze_axis` can be set to ``true`` only if ``input.shape[axis] == num_outputs``.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def Softmax(data=None, grad_scale=_Null, ignore_label=_Null, multi_output=_Null, use_ignore=_Null, preserve_shape=_Null, normalization=_Null, out_grad=_Null, smooth_alpha=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Please use `SoftmaxOutput`.

    .. note::

      This operator has been renamed to `SoftmaxOutput`, which
      computes the gradient of cross-entropy loss w.r.t softmax output.
      To just compute softmax output, use the `softmax` operator.



    Defined in src/operator/softmax_output.cc:L138

    Parameters
    ----------
    data : Symbol
        Input array.
    grad_scale : float, optional, default=1
        Scales the gradient by a float factor.
    ignore_label : float, optional, default=-1
        The instances whose `labels` == `ignore_label` will be ignored during backward, if `use_ignore` is set to ``true``).
    multi_output : boolean, optional, default=0
        If set to ``true``, the softmax function will be computed along axis ``1``. This is applied when the shape of input array differs from the shape of label array.
    use_ignore : boolean, optional, default=0
        If set to ``true``, the `ignore_label` value will not contribute to the backward gradient.
    preserve_shape : boolean, optional, default=0
        If set to ``true``, the softmax function will be computed along the last axis (``-1``).
    normalization : {'batch', 'null', 'valid'},optional, default='null'
        Normalizes the gradient.
    out_grad : boolean, optional, default=0
        Multiplies gradient with output gradient element-wise.
    smooth_alpha : float, optional, default=0
        Constant for computing a label smoothed version of cross-entropyfor the backwards pass.  This constant gets subtracted from theone-hot encoding of the gold label and distributed uniformly toall other labels.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def SoftmaxActivation(data=None, mode=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Applies softmax activation to input. This is intended for internal layers.

    .. note::

      This operator has been deprecated, please use `softmax`.

    If `mode` = ``instance``, this operator will compute a softmax for each instance in the batch.
    This is the default mode.

    If `mode` = ``channel``, this operator will compute a k-class softmax at each position
    of each instance, where `k` = ``num_channel``. This mode can only be used when the input array
    has at least 3 dimensions.
    This can be used for `fully convolutional network`, `image segmentation`, etc.

    Example::

      >>> input_array = mx.nd.array([[3., 0.5, -0.5, 2., 7.],
      >>>                            [2., -.4, 7.,   3., 0.2]])
      >>> softmax_act = mx.nd.SoftmaxActivation(input_array)
      >>> print softmax_act.asnumpy()
      [[  1.78322066e-02   1.46375655e-03   5.38485940e-04   6.56010211e-03   9.73605454e-01]
       [  6.56221947e-03   5.95310994e-04   9.73919690e-01   1.78379621e-02   1.08472735e-03]]



    Defined in src/operator/nn/softmax_activation.cc:L59

    Parameters
    ----------
    data : Symbol
        The input array.
    mode : {'channel', 'instance'},optional, default='instance'
        Specifies how to compute the softmax. If set to ``instance``, it computes softmax for each instance. If set to ``channel``, It computes cross channel softmax for each position of each instance.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def SoftmaxOutput(data=None, label=None, grad_scale=_Null, ignore_label=_Null, multi_output=_Null, use_ignore=_Null, preserve_shape=_Null, normalization=_Null, out_grad=_Null, smooth_alpha=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Computes the gradient of cross entropy loss with respect to softmax output.

    - This operator computes the gradient in two steps.
      The cross entropy loss does not actually need to be computed.

      - Applies softmax function on the input array.
      - Computes and returns the gradient of cross entropy loss w.r.t. the softmax output.

    - The softmax function, cross entropy loss and gradient is given by:

      - Softmax Function:

        .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}

      - Cross Entropy Function:

        .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i \log(\text{output}_i)

      - The gradient of cross entropy loss w.r.t softmax output:

        .. math:: \text{gradient} = \text{output} - \text{label}

    - During forward propagation, the softmax function is computed for each instance in the input array.

      For general *N*-D input arrays with shape :math:`(d_1, d_2, ..., d_n)`. The size is
      :math:`s=d_1 \cdot d_2 \cdot \cdot \cdot d_n`. We can use the parameters `preserve_shape`
      and `multi_output` to specify the way to compute softmax:

      - By default, `preserve_shape` is ``false``. This operator will reshape the input array
        into a 2-D array with shape :math:`(d_1, \frac{s}{d_1})` and then compute the softmax function for
        each row in the reshaped array, and afterwards reshape it back to the original shape
        :math:`(d_1, d_2, ..., d_n)`.
      - If `preserve_shape` is ``true``, the softmax function will be computed along
        the last axis (`axis` = ``-1``).
      - If `multi_output` is ``true``, the softmax function will be computed along
        the second axis (`axis` = ``1``).

    - During backward propagation, the gradient of cross-entropy loss w.r.t softmax output array is computed.
      The provided label can be a one-hot label array or a probability label array.

      - If the parameter `use_ignore` is ``true``, `ignore_label` can specify input instances
        with a particular label to be ignored during backward propagation. **This has no effect when
        softmax `output` has same shape as `label`**.

        Example::

          data = [[1,2,3,4],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
          label = [1,0,2,3]
          ignore_label = 1
          SoftmaxOutput(data=data, label = label,\
                        multi_output=true, use_ignore=true,\
                        ignore_label=ignore_label)
          ## forward softmax output
          [[ 0.0320586   0.08714432  0.23688284  0.64391428]
           [ 0.25        0.25        0.25        0.25      ]
           [ 0.25        0.25        0.25        0.25      ]
           [ 0.25        0.25        0.25        0.25      ]]
          ## backward gradient output
          [[ 0.    0.    0.    0.  ]
           [-0.75  0.25  0.25  0.25]
           [ 0.25  0.25 -0.75  0.25]
           [ 0.25  0.25  0.25 -0.75]]
          ## notice that the first row is all 0 because label[0] is 1, which is equal to ignore_label.

      - The parameter `grad_scale` can be used to rescale the gradient, which is often used to
        give each loss function different weights.

      - This operator also supports various ways to normalize the gradient by `normalization`,
        The `normalization` is applied if softmax output has different shape than the labels.
        The `normalization` mode can be set to the followings:

        - ``'null'``: do nothing.
        - ``'batch'``: divide the gradient by the batch size.
        - ``'valid'``: divide the gradient by the number of instances which are not ignored.



    Defined in src/operator/softmax_output.cc:L123

    Parameters
    ----------
    data : Symbol
        Input array.
    label : Symbol
        Ground truth label.
    grad_scale : float, optional, default=1
        Scales the gradient by a float factor.
    ignore_label : float, optional, default=-1
        The instances whose `labels` == `ignore_label` will be ignored during backward, if `use_ignore` is set to ``true``).
    multi_output : boolean, optional, default=0
        If set to ``true``, the softmax function will be computed along axis ``1``. This is applied when the shape of input array differs from the shape of label array.
    use_ignore : boolean, optional, default=0
        If set to ``true``, the `ignore_label` value will not contribute to the backward gradient.
    preserve_shape : boolean, optional, default=0
        If set to ``true``, the softmax function will be computed along the last axis (``-1``).
    normalization : {'batch', 'null', 'valid'},optional, default='null'
        Normalizes the gradient.
    out_grad : boolean, optional, default=0
        Multiplies gradient with output gradient element-wise.
    smooth_alpha : float, optional, default=0
        Constant for computing a label smoothed version of cross-entropyfor the backwards pass.  This constant gets subtracted from theone-hot encoding of the gold label and distributed uniformly toall other labels.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def SpatialTransformer(data=None, loc=None, target_shape=_Null, transform_type=_Null, sampler_type=_Null, cudnn_off=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Applies a spatial transformer to input feature map.

    Parameters
    ----------
    data : Symbol
        Input data to the SpatialTransformerOp.
    loc : Symbol
        localisation net, the output dim should be 6 when transform_type is affine. You shold initialize the weight and bias with identity tranform.
    target_shape : Shape(tuple), optional, default=[0,0]
        output shape(h, w) of spatial transformer: (y, x)
    transform_type : {'affine'}, required
        transformation type
    sampler_type : {'bilinear'}, required
        sampling type
    cudnn_off : boolean or None, optional, default=None
        whether to turn cudnn off

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def SwapAxis(data=None, dim1=_Null, dim2=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Interchanges two axes of an array.

    Examples::

      x = [[1, 2, 3]])
      swapaxes(x, 0, 1) = [[ 1],
                           [ 2],
                           [ 3]]

      x = [[[ 0, 1],
            [ 2, 3]],
           [[ 4, 5],
            [ 6, 7]]]  // (2,2,2) array

     swapaxes(x, 0, 2) = [[[ 0, 4],
                           [ 2, 6]],
                          [[ 1, 5],
                           [ 3, 7]]]


    Defined in src/operator/swapaxis.cc:L70

    Parameters
    ----------
    data : Symbol
        Input array.
    dim1 : int (non-negative), optional, default=0
        the first axis to be swapped.
    dim2 : int (non-negative), optional, default=0
        the second axis to be swapped.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def UpSampling(*data, **kwargs):
    r"""Performs nearest neighbor/bilinear up sampling to inputs.
    This function support variable length of positional input.

    Parameters
    ----------
    data : Symbol[]
        Array of tensors to upsample
    scale : int, required
        Up sampling scale
    num_filter : int, optional, default='0'
        Input filter. Only used by bilinear sample_type.
    sample_type : {'bilinear', 'nearest'}, required
        upsampling method
    multi_input_mode : {'concat', 'sum'},optional, default='concat'
        How to handle multiple input. concat means concatenate upsampled images along the channel dimension. sum means add all images together, only available for nearest neighbor upsampling.
    workspace : long (non-negative), optional, default=512
        Tmp workspace for deconvolution (MB)

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def abs(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise absolute value of the input.

    Example::

       abs([-2, 0, 3]) = [2, 0, 3]

    The storage type of ``abs`` output depends upon the input storage type:

       - abs(default) = default
       - abs(row_sparse) = row_sparse
       - abs(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L662

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

def adam_update(weight=None, grad=None, mean=None, var=None, lr=_Null, beta1=_Null, beta2=_Null, epsilon=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, lazy_update=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Update function for Adam optimizer. Adam is seen as a generalization
    of AdaGrad.

    Adam update consists of the following steps, where g represents gradient and m, v
    are 1st and 2nd order moment estimates (mean and variance).

    .. math::

     g_t = \nabla J(W_{t-1})\\
     m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
     v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
     W_t = W_{t-1} - \alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon }

    It updates the weights using::

     m = beta1*m + (1-beta1)*grad
     v = beta2*v + (1-beta2)*(grad**2)
     w += - learning_rate * m / (sqrt(v) + epsilon)

    However, if grad's storage type is ``row_sparse``, ``lazy_update`` is True and the storage
    type of weight is the same as those of m and v,
    only the row slices whose indices appear in grad.indices are updated (for w, m and v)::

     for row in grad.indices:
         m[row] = beta1*m[row] + (1-beta1)*grad[row]
         v[row] = beta2*v[row] + (1-beta2)*(grad[row]**2)
         w[row] += - learning_rate * m[row] / (sqrt(v[row]) + epsilon)



    Defined in src/operator/optimizer_op.cc:L495

    Parameters
    ----------
    weight : Symbol
        Weight
    grad : Symbol
        Gradient
    mean : Symbol
        Moving mean
    var : Symbol
        Moving variance
    lr : float, required
        Learning rate
    beta1 : float, optional, default=0.9
        The decay rate for the 1st moment estimates.
    beta2 : float, optional, default=0.999
        The decay rate for the 2nd moment estimates.
    epsilon : float, optional, default=1e-08
        A small constant for numerical stability.
    wd : float, optional, default=0
        Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
    rescale_grad : float, optional, default=1
        Rescale gradient to grad = rescale_grad*grad.
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
    lazy_update : boolean, optional, default=1
        If true, lazy updates are applied if gradient's stype is row_sparse and all of w, m and v have the same stype

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def add_n(*args, **kwargs):
    r"""Adds all input arguments element-wise.

    .. math::
       add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n

    ``add_n`` is potentially more efficient than calling ``add`` by `n` times.

    The storage type of ``add_n`` output depends on storage types of inputs

    - add_n(row_sparse, row_sparse, ..) = row_sparse
    - add_n(default, csr, default) = default
    - add_n(any input combinations longer than 4 (>4) with at least one default type) = default
    - otherwise, ``add_n`` falls all inputs back to default storage and generates default storage



    Defined in src/operator/tensor/elemwise_sum.cc:L156
    This function support variable length of positional input.

    Parameters
    ----------
    args : Symbol[]
        Positional input arguments

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def arccos(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise inverse cosine of the input array.

    The input should be in range `[-1, 1]`.
    The output is in the closed interval :math:`[0, \pi]`

    .. math::
       arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]

    The storage type of ``arccos`` output is always dense



    Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L123

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

def arccosh(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns the element-wise inverse hyperbolic cosine of the input array, \
    computed element-wise.

    The storage type of ``arccosh`` output is always dense



    Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L264

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

def arcsin(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise inverse sine of the input array.

    The input should be in the range `[-1, 1]`.
    The output is in the closed interval of [:math:`-\pi/2`, :math:`\pi/2`].

    .. math::
       arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]

    The storage type of ``arcsin`` output depends upon the input storage type:

       - arcsin(default) = default
       - arcsin(row_sparse) = row_sparse
       - arcsin(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L104

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

def arcsinh(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns the element-wise inverse hyperbolic sine of the input array, \
    computed element-wise.

    The storage type of ``arcsinh`` output depends upon the input storage type:

       - arcsinh(default) = default
       - arcsinh(row_sparse) = row_sparse
       - arcsinh(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L250

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

def arctan(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise inverse tangent of the input array.

    The output is in the closed interval :math:`[-\pi/2, \pi/2]`

    .. math::
       arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]

    The storage type of ``arctan`` output depends upon the input storage type:

       - arctan(default) = default
       - arctan(row_sparse) = row_sparse
       - arctan(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L144

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

def arctanh(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns the element-wise inverse hyperbolic tangent of the input array, \
    computed element-wise.

    The storage type of ``arctanh`` output depends upon the input storage type:

       - arctanh(default) = default
       - arctanh(row_sparse) = row_sparse
       - arctanh(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L281

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

def argmax(data=None, axis=_Null, keepdims=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Returns indices of the maximum values along an axis.

    In the case of multiple occurrences of maximum values, the indices corresponding to the first occurrence
    are returned.

    Examples::

      x = [[ 0.,  1.,  2.],
           [ 3.,  4.,  5.]]

      // argmax along axis 0
      argmax(x, axis=0) = [ 1.,  1.,  1.]

      // argmax along axis 1
      argmax(x, axis=1) = [ 2.,  2.]

      // argmax along axis 1 keeping same dims as an input array
      argmax(x, axis=1, keepdims=True) = [[ 2.],
                                          [ 2.]]



    Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L52

    Parameters
    ----------
    data : Symbol
        The input
    axis : int or None, optional, default='None'
        The axis along which to perform the reduction. Negative values means indexing from right to left. ``Requires axis to be set as int, because global reduction is not supported yet.``
    keepdims : boolean, optional, default=0
        If this is set to `True`, the reduced axis is left in the result as dimension with size one.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def argmax_channel(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns argmax indices of each channel from the input array.

    The result will be an NDArray of shape (num_channel,).

    In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence
    are returned.

    Examples::

      x = [[ 0.,  1.,  2.],
           [ 3.,  4.,  5.]]

      argmax_channel(x) = [ 2.,  2.]



    Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L97

    Parameters
    ----------
    data : Symbol
        The input array

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def argmin(data=None, axis=_Null, keepdims=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Returns indices of the minimum values along an axis.

    In the case of multiple occurrences of minimum values, the indices corresponding to the first occurrence
    are returned.

    Examples::

      x = [[ 0.,  1.,  2.],
           [ 3.,  4.,  5.]]

      // argmin along axis 0
      argmin(x, axis=0) = [ 0.,  0.,  0.]

      // argmin along axis 1
      argmin(x, axis=1) = [ 0.,  0.]

      // argmin along axis 1 keeping same dims as an input array
      argmin(x, axis=1, keepdims=True) = [[ 0.],
                                          [ 0.]]



    Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L77

    Parameters
    ----------
    data : Symbol
        The input
    axis : int or None, optional, default='None'
        The axis along which to perform the reduction. Negative values means indexing from right to left. ``Requires axis to be set as int, because global reduction is not supported yet.``
    keepdims : boolean, optional, default=0
        If this is set to `True`, the reduced axis is left in the result as dimension with size one.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def argsort(data=None, axis=_Null, is_ascend=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Returns the indices that would sort an input array along the given axis.

    This function performs sorting along the given axis and returns an array of indices having same shape
    as an input array that index data in sorted order.

    Examples::

      x = [[ 0.3,  0.2,  0.4],
           [ 0.1,  0.3,  0.2]]

      // sort along axis -1
      argsort(x) = [[ 1.,  0.,  2.],
                    [ 0.,  2.,  1.]]

      // sort along axis 0
      argsort(x, axis=0) = [[ 1.,  0.,  1.]
                            [ 0.,  1.,  0.]]

      // flatten and then sort
      argsort(x) = [ 3.,  1.,  5.,  0.,  4.,  2.]


    Defined in src/operator/tensor/ordering_op.cc:L177

    Parameters
    ----------
    data : Symbol
        The input array
    axis : int or None, optional, default='-1'
        Axis along which to sort the input tensor. If not given, the flattened array is used. Default is -1.
    is_ascend : boolean, optional, default=1
        Whether to sort in ascending or descending order.
    dtype : {'float16', 'float32', 'float64', 'int32', 'uint8'},optional, default='float32'
        DType of the output indices. It is only valid when ret_typ is "indices" or "both". An error will be raised if the selected data type cannot precisely represent the indices.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def batch_dot(lhs=None, rhs=None, transpose_a=_Null, transpose_b=_Null, forward_stype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Batchwise dot product.

    ``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and
    ``y`` are data in batch, namely 3D arrays in shape of `(batch_size, :, :)`.

    For example, given ``x`` with shape `(batch_size, n, m)` and ``y`` with shape
    `(batch_size, m, k)`, the result array will have shape `(batch_size, n, k)`,
    which is computed by::

       batch_dot(x,y)[i,:,:] = dot(x[i,:,:], y[i,:,:])



    Defined in src/operator/tensor/dot.cc:L125

    Parameters
    ----------
    lhs : Symbol
        The first input
    rhs : Symbol
        The second input
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

def batch_take(a=None, indices=None, name=None, attr=None, out=None, **kwargs):
    r"""Takes elements from a data batch.

    .. note::
      `batch_take` is deprecated. Use `pick` instead.

    Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the result will be
    an output array of shape ``(i0,)`` with::

      output[i] = input[i, indices[i]]

    Examples::

      x = [[ 1.,  2.],
           [ 3.,  4.],
           [ 5.,  6.]]

      // takes elements with specified indices
      batch_take(x, [0,1,0]) = [ 1.  4.  5.]



    Defined in src/operator/tensor/indexing_op.cc:L750

    Parameters
    ----------
    a : Symbol
        The input array
    indices : Symbol
        The index array

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_add(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise sum of the input arrays with broadcasting.

    `broadcast_plus` is an alias to the function `broadcast_add`.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_add(x, y) = [[ 1.,  1.,  1.],
                              [ 2.,  2.,  2.]]

       broadcast_plus(x, y) = [[ 1.,  1.,  1.],
                               [ 2.,  2.,  2.]]

    Supported sparse operations:

       broadcast_add(csr, dense(1D)) = dense
       broadcast_add(dense(1D), csr) = dense



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L58

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_axes(data=None, axis=_Null, size=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Broadcasts the input array over particular axes.

    Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
    `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.

    Example::

       // given x of shape (1,2,1)
       x = [[[ 1.],
             [ 2.]]]

       // broadcast x on on axis 2
       broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],
                                             [ 2.,  2.,  2.]]]
       // broadcast x on on axes 0 and 2
       broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],
                                                     [ 2.,  2.,  2.]],
                                                    [[ 1.,  1.,  1.],
                                                     [ 2.,  2.,  2.]]]


    Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L238

    Parameters
    ----------
    data : Symbol
        The input
    axis : Shape(tuple), optional, default=[]
        The axes to perform the broadcasting.
    size : Shape(tuple), optional, default=[]
        Target sizes of the broadcasting axes.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_axis(data=None, axis=_Null, size=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Broadcasts the input array over particular axes.

    Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
    `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.

    Example::

       // given x of shape (1,2,1)
       x = [[[ 1.],
             [ 2.]]]

       // broadcast x on on axis 2
       broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],
                                             [ 2.,  2.,  2.]]]
       // broadcast x on on axes 0 and 2
       broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],
                                                     [ 2.,  2.,  2.]],
                                                    [[ 1.,  1.,  1.],
                                                     [ 2.,  2.,  2.]]]


    Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L238

    Parameters
    ----------
    data : Symbol
        The input
    axis : Shape(tuple), optional, default=[]
        The axes to perform the broadcasting.
    size : Shape(tuple), optional, default=[]
        Target sizes of the broadcasting axes.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_div(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise division of the input arrays with broadcasting.

    Example::

       x = [[ 6.,  6.,  6.],
            [ 6.,  6.,  6.]]

       y = [[ 2.],
            [ 3.]]

       broadcast_div(x, y) = [[ 3.,  3.,  3.],
                              [ 2.,  2.,  2.]]

    Supported sparse operations:

       broadcast_div(csr, dense(1D)) = csr



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L187

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_equal(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns the result of element-wise **equal to** (==) comparison operation with broadcasting.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_equal(x, y) = [[ 0.,  0.,  0.],
                                [ 1.,  1.,  1.]]



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L46

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_greater(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns the result of element-wise **greater than** (>) comparison operation with broadcasting.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_greater(x, y) = [[ 1.,  1.,  1.],
                                  [ 0.,  0.,  0.]]



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L82

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_greater_equal(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns the result of element-wise **greater than or equal to** (>=) comparison operation with broadcasting.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_greater_equal(x, y) = [[ 1.,  1.,  1.],
                                        [ 1.,  1.,  1.]]



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L100

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_hypot(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r""" Returns the hypotenuse of a right angled triangle, given its "legs"
    with broadcasting.

    It is equivalent to doing :math:`sqrt(x_1^2 + x_2^2)`.

    Example::

       x = [[ 3.,  3.,  3.]]

       y = [[ 4.],
            [ 4.]]

       broadcast_hypot(x, y) = [[ 5.,  5.,  5.],
                                [ 5.,  5.,  5.]]

       z = [[ 0.],
            [ 4.]]

       broadcast_hypot(x, z) = [[ 3.,  3.,  3.],
                                [ 5.,  5.,  5.]]



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L156

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_lesser(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns the result of element-wise **lesser than** (<) comparison operation with broadcasting.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_lesser(x, y) = [[ 0.,  0.,  0.],
                                 [ 0.,  0.,  0.]]



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L118

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_lesser_equal(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns the result of element-wise **lesser than or equal to** (<=) comparison operation with broadcasting.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_lesser_equal(x, y) = [[ 0.,  0.,  0.],
                                       [ 1.,  1.,  1.]]



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L136

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_like(lhs=None, rhs=None, lhs_axes=_Null, rhs_axes=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Broadcasts lhs to have the same shape as rhs.

    Broadcasting is a mechanism that allows NDArrays to perform arithmetic operations
    with arrays of different shapes efficiently without creating multiple copies of arrays.
    Also see, `Broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more explanation.

    Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
    `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.

    For example::

       broadcast_like([[1,2,3]], [[5,6,7],[7,8,9]]) = [[ 1.,  2.,  3.],
                                                       [ 1.,  2.,  3.]])

       broadcast_like([9], [1,2,3,4,5], lhs_axes=(0,), rhs_axes=(-1,)) = [9,9,9,9,9]



    Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L315

    Parameters
    ----------
    lhs : Symbol
        First input.
    rhs : Symbol
        Second input.
    lhs_axes : Shape or None, optional, default=None
        Axes to perform broadcast on in the first input array
    rhs_axes : Shape or None, optional, default=None
        Axes to copy from the second input array

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_logical_and(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns the result of element-wise **logical and** with broadcasting.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_logical_and(x, y) = [[ 0.,  0.,  0.],
                                      [ 1.,  1.,  1.]]



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L154

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_logical_or(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns the result of element-wise **logical or** with broadcasting.

    Example::

       x = [[ 1.,  1.,  0.],
            [ 1.,  1.,  0.]]

       y = [[ 1.],
            [ 0.]]

       broadcast_logical_or(x, y) = [[ 1.,  1.,  1.],
                                     [ 1.,  1.,  0.]]



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L172

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_logical_xor(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns the result of element-wise **logical xor** with broadcasting.

    Example::

       x = [[ 1.,  1.,  0.],
            [ 1.,  1.,  0.]]

       y = [[ 1.],
            [ 0.]]

       broadcast_logical_xor(x, y) = [[ 0.,  0.,  1.],
                                      [ 1.,  1.,  0.]]



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L190

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_maximum(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise maximum of the input arrays with broadcasting.

    This function compares two input arrays and returns a new array having the element-wise maxima.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_maximum(x, y) = [[ 1.,  1.,  1.],
                                  [ 1.,  1.,  1.]]



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L80

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_minimum(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise minimum of the input arrays with broadcasting.

    This function compares two input arrays and returns a new array having the element-wise minima.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_maximum(x, y) = [[ 0.,  0.,  0.],
                                  [ 1.,  1.,  1.]]



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L115

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_minus(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise difference of the input arrays with broadcasting.

    `broadcast_minus` is an alias to the function `broadcast_sub`.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_sub(x, y) = [[ 1.,  1.,  1.],
                              [ 0.,  0.,  0.]]

       broadcast_minus(x, y) = [[ 1.,  1.,  1.],
                                [ 0.,  0.,  0.]]

    Supported sparse operations:

       broadcast_sub/minus(csr, dense(1D)) = dense
       broadcast_sub/minus(dense(1D), csr) = dense



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L106

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_mod(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise modulo of the input arrays with broadcasting.

    Example::

       x = [[ 8.,  8.,  8.],
            [ 8.,  8.,  8.]]

       y = [[ 2.],
            [ 3.]]

       broadcast_mod(x, y) = [[ 0.,  0.,  0.],
                              [ 2.,  2.,  2.]]



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L222

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_mul(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise product of the input arrays with broadcasting.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_mul(x, y) = [[ 0.,  0.,  0.],
                              [ 1.,  1.,  1.]]

    Supported sparse operations:

       broadcast_mul(csr, dense(1D)) = csr



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L146

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_not_equal(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns the result of element-wise **not equal to** (!=) comparison operation with broadcasting.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_not_equal(x, y) = [[ 1.,  1.,  1.],
                                    [ 0.,  0.,  0.]]



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L64

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_plus(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise sum of the input arrays with broadcasting.

    `broadcast_plus` is an alias to the function `broadcast_add`.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_add(x, y) = [[ 1.,  1.,  1.],
                              [ 2.,  2.,  2.]]

       broadcast_plus(x, y) = [[ 1.,  1.,  1.],
                               [ 2.,  2.,  2.]]

    Supported sparse operations:

       broadcast_add(csr, dense(1D)) = dense
       broadcast_add(dense(1D), csr) = dense



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L58

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_power(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns result of first array elements raised to powers from second array, element-wise with broadcasting.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_power(x, y) = [[ 2.,  2.,  2.],
                                [ 4.,  4.,  4.]]



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L45

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_sub(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise difference of the input arrays with broadcasting.

    `broadcast_minus` is an alias to the function `broadcast_sub`.

    Example::

       x = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]

       y = [[ 0.],
            [ 1.]]

       broadcast_sub(x, y) = [[ 1.,  1.,  1.],
                              [ 0.,  0.,  0.]]

       broadcast_minus(x, y) = [[ 1.,  1.,  1.],
                                [ 0.,  0.,  0.]]

    Supported sparse operations:

       broadcast_sub/minus(csr, dense(1D)) = dense
       broadcast_sub/minus(dense(1D), csr) = dense



    Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L106

    Parameters
    ----------
    lhs : Symbol
        First input to the function
    rhs : Symbol
        Second input to the function

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def broadcast_to(data=None, shape=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Broadcasts the input array to a new shape.

    Broadcasting is a mechanism that allows NDArrays to perform arithmetic operations
    with arrays of different shapes efficiently without creating multiple copies of arrays.
    Also see, `Broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more explanation.

    Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
    `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.

    For example::

       broadcast_to([[1,2,3]], shape=(2,3)) = [[ 1.,  2.,  3.],
                                               [ 1.,  2.,  3.]])

    The dimension which you do not want to change can also be kept as `0` which means copy the original value.
    So with `shape=(2,0)`, we will obtain the same result as in the above example.



    Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L262

    Parameters
    ----------
    data : Symbol
        The input
    shape : Shape(tuple), optional, default=[]
        The shape of the desired array. We can set the dim to zero if it's same as the original. E.g `A = broadcast_to(B, shape=(10, 0, 0))` has the same meaning as `A = broadcast_axis(B, axis=0, size=10)`.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def cast(data=None, dtype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Casts all elements of the input to a new type.

    .. note:: ``Cast`` is deprecated. Use ``cast`` instead.

    Example::

       cast([0.9, 1.3], dtype='int32') = [0, 1]
       cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]
       cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L596

    Parameters
    ----------
    data : Symbol
        The input.
    dtype : {'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'}, required
        Output data type.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def cast_storage(data=None, stype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Casts tensor storage type to the new type.

    When an NDArray with default storage type is cast to csr or row_sparse storage,
    the result is compact, which means:

    - for csr, zero values will not be retained
    - for row_sparse, row slices of all zeros will not be retained

    The storage type of ``cast_storage`` output depends on stype parameter:

    - cast_storage(csr, 'default') = default
    - cast_storage(row_sparse, 'default') = default
    - cast_storage(default, 'csr') = csr
    - cast_storage(default, 'row_sparse') = row_sparse
    - cast_storage(csr, 'csr') = csr
    - cast_storage(row_sparse, 'row_sparse') = row_sparse

    Example::

        dense = [[ 0.,  1.,  0.],
                 [ 2.,  0.,  3.],
                 [ 0.,  0.,  0.],
                 [ 0.,  0.,  0.]]

        # cast to row_sparse storage type
        rsp = cast_storage(dense, 'row_sparse')
        rsp.indices = [0, 1]
        rsp.values = [[ 0.,  1.,  0.],
                      [ 2.,  0.,  3.]]

        # cast to csr storage type
        csr = cast_storage(dense, 'csr')
        csr.indices = [1, 0, 2]
        csr.values = [ 1.,  2.,  3.]
        csr.indptr = [0, 1, 3, 3, 3]



    Defined in src/operator/tensor/cast_storage.cc:L71

    Parameters
    ----------
    data : Symbol
        The input.
    stype : {'csr', 'default', 'row_sparse'}, required
        Output storage type.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def cbrt(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise cube-root value of the input.

    .. math::
       cbrt(x) = \sqrt[3]{x}

    Example::

       cbrt([1, 8, -125]) = [1, 2, -5]

    The storage type of ``cbrt`` output depends upon the input storage type:

       - cbrt(default) = default
       - cbrt(row_sparse) = row_sparse
       - cbrt(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L883

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

def ceil(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise ceiling of the input.

    The ceil of the scalar x is the smallest integer i, such that i >= x.

    Example::

       ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]

    The storage type of ``ceil`` output depends upon the input storage type:

       - ceil(default) = default
       - ceil(row_sparse) = row_sparse
       - ceil(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L740

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

def choose_element_0index(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Choose one element from each line(row for python, column for R/Julia) in lhs according to index indicated by rhs. This function assume rhs uses 0-based index.

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

def clip(data=None, a_min=_Null, a_max=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Clips (limits) the values in an array.

    Given an interval, values outside the interval are clipped to the interval edges.
    Clipping ``x`` between `a_min` and `a_x` would be::

       clip(x, a_min, a_max) = max(min(x, a_max), a_min))

    Example::

        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        clip(x,1,8) = [ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.]

    The storage type of ``clip`` output depends on storage types of inputs and the a_min, a_max \
    parameter values:

       - clip(default) = default
       - clip(row_sparse, a_min <= 0, a_max >= 0) = row_sparse
       - clip(csr, a_min <= 0, a_max >= 0) = csr
       - clip(row_sparse, a_min < 0, a_max < 0) = default
       - clip(row_sparse, a_min > 0, a_max > 0) = default
       - clip(csr, a_min < 0, a_max < 0) = csr
       - clip(csr, a_min > 0, a_max > 0) = csr



    Defined in src/operator/tensor/matrix_op.cc:L619

    Parameters
    ----------
    data : Symbol
        Input array.
    a_min : float, required
        Minimum value
    a_max : float, required
        Maximum value

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def concat(*data, **kwargs):
    r"""Joins input arrays along a given axis.

    .. note:: `Concat` is deprecated. Use `concat` instead.

    The dimensions of the input arrays should be the same except the axis along
    which they will be concatenated.
    The dimension of the output array along the concatenated axis will be equal
    to the sum of the corresponding dimensions of the input arrays.

    The storage type of ``concat`` output depends on storage types of inputs

    - concat(csr, csr, ..., csr, dim=0) = csr
    - otherwise, ``concat`` generates output with default storage

    Example::

       x = [[1,1],[2,2]]
       y = [[3,3],[4,4],[5,5]]
       z = [[6,6], [7,7],[8,8]]

       concat(x,y,z,dim=0) = [[ 1.,  1.],
                              [ 2.,  2.],
                              [ 3.,  3.],
                              [ 4.,  4.],
                              [ 5.,  5.],
                              [ 6.,  6.],
                              [ 7.,  7.],
                              [ 8.,  8.]]

       Note that you cannot concat x,y,z along dimension 1 since dimension
       0 is not the same for all the input arrays.

       concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],
                             [ 4.,  4.,  7.,  7.],
                             [ 5.,  5.,  8.,  8.]]



    Defined in src/operator/nn/concat.cc:L368
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

def cos(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Computes the element-wise cosine of the input array.

    The input should be in radians (:math:`2\pi` rad equals 360 degrees).

    .. math::
       cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]

    The storage type of ``cos`` output is always dense



    Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L63

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

def cosh(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns the hyperbolic cosine  of the input array, computed element-wise.

    .. math::
       cosh(x) = 0.5\times(exp(x) + exp(-x))

    The storage type of ``cosh`` output is always dense



    Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L216

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

def crop(data=None, begin=_Null, end=_Null, step=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Slices a region of the array.

    .. note:: ``crop`` is deprecated. Use ``slice`` instead.

    This function returns a sliced array between the indices given
    by `begin` and `end` with the corresponding `step`.

    For an input array of ``shape=(d_0, d_1, ..., d_n-1)``,
    slice operation with ``begin=(b_0, b_1...b_m-1)``,
    ``end=(e_0, e_1, ..., e_m-1)``, and ``step=(s_0, s_1, ..., s_m-1)``,
    where m <= n, results in an array with the shape
    ``(|e_0-b_0|/|s_0|, ..., |e_m-1-b_m-1|/|s_m-1|, d_m, ..., d_n-1)``.

    The resulting array's *k*-th dimension contains elements
    from the *k*-th dimension of the input array starting
    from index ``b_k`` (inclusive) with step ``s_k``
    until reaching ``e_k`` (exclusive).

    If the *k*-th elements are `None` in the sequence of `begin`, `end`,
    and `step`, the following rule will be used to set default values.
    If `s_k` is `None`, set `s_k=1`. If `s_k > 0`, set `b_k=0`, `e_k=d_k`;
    else, set `b_k=d_k-1`, `e_k=-1`.

    The storage type of ``slice`` output depends on storage types of inputs

    - slice(csr) = csr
    - otherwise, ``slice`` generates output with default storage

    .. note:: When input data storage type is csr, it only supports
       step=(), or step=(None,), or step=(1,) to generate a csr output.
       For other step parameter values, it falls back to slicing
       a dense tensor.

    Example::

      x = [[  1.,   2.,   3.,   4.],
           [  5.,   6.,   7.,   8.],
           [  9.,  10.,  11.,  12.]]

      slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],
                                         [ 6.,  7.,  8.]]
      slice(x, begin=(None, 0), end=(None, 3), step=(-1, 2)) = [[9., 11.],
                                                                [5.,  7.],
                                                                [1.,  3.]]


    Defined in src/operator/tensor/matrix_op.cc:L414

    Parameters
    ----------
    data : Symbol
        Source input
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

def ctc_loss(data=None, label=None, data_lengths=None, label_lengths=None, use_data_lengths=_Null, use_label_lengths=_Null, blank_label=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Connectionist Temporal Classification Loss.

    .. note:: The existing alias ``contrib_CTCLoss`` is deprecated.

    The shapes of the inputs and outputs:

    - **data**: `(sequence_length, batch_size, alphabet_size)`
    - **label**: `(batch_size, label_sequence_length)`
    - **out**: `(batch_size)`

    The `data` tensor consists of sequences of activation vectors (without applying softmax),
    with i-th channel in the last dimension corresponding to i-th label
    for i between 0 and alphabet_size-1 (i.e always 0-indexed).
    Alphabet size should include one additional value reserved for blank label.
    When `blank_label` is ``"first"``, the ``0``-th channel is be reserved for
    activation of blank label, or otherwise if it is "last", ``(alphabet_size-1)``-th channel should be
    reserved for blank label.

    ``label`` is an index matrix of integers. When `blank_label` is ``"first"``,
    the value 0 is then reserved for blank label, and should not be passed in this matrix. Otherwise,
    when `blank_label` is ``"last"``, the value `(alphabet_size-1)` is reserved for blank label.

    If a sequence of labels is shorter than *label_sequence_length*, use the special
    padding value at the end of the sequence to conform it to the correct
    length. The padding value is `0` when `blank_label` is ``"first"``, and `-1` otherwise.

    For example, suppose the vocabulary is `[a, b, c]`, and in one batch we have three sequences
    'ba', 'cbb', and 'abac'. When `blank_label` is ``"first"``, we can index the labels as
    `{'a': 1, 'b': 2, 'c': 3}`, and we reserve the 0-th channel for blank label in data tensor.
    The resulting `label` tensor should be padded to be::

      [[2, 1, 0, 0], [3, 2, 2, 0], [1, 2, 1, 3]]

    When `blank_label` is ``"last"``, we can index the labels as
    `{'a': 0, 'b': 1, 'c': 2}`, and we reserve the channel index 3 for blank label in data tensor.
    The resulting `label` tensor should be padded to be::

      [[1, 0, -1, -1], [2, 1, 1, -1], [0, 1, 0, 2]]

    ``out`` is a list of CTC loss values, one per example in the batch.

    See *Connectionist Temporal Classification: Labelling Unsegmented
    Sequence Data with Recurrent Neural Networks*, A. Graves *et al*. for more
    information on the definition and the algorithm.



    Defined in src/operator/nn/ctc_loss.cc:L100

    Parameters
    ----------
    data : Symbol
        Input ndarray
    label : Symbol
        Ground-truth labels for the loss.
    data_lengths : Symbol
        Lengths of data for each of the samples. Only required when use_data_lengths is true.
    label_lengths : Symbol
        Lengths of labels for each of the samples. Only required when use_label_lengths is true.
    use_data_lengths : boolean, optional, default=0
        Whether the data lenghts are decided by `data_lengths`. If false, the lengths are equal to the max sequence length.
    use_label_lengths : boolean, optional, default=0
        Whether the label lenghts are decided by `label_lengths`, or derived from `padding_mask`. If false, the lengths are derived from the first occurrence of the value of `padding_mask`. The value of `padding_mask` is ``0`` when first CTC label is reserved for blank, and ``-1`` when last label is reserved for blank. See `blank_label`.
    blank_label : {'first', 'last'},optional, default='first'
        Set the label that is reserved for blank label.If "first", 0-th label is reserved, and label values for tokens in the vocabulary are between ``1`` and ``alphabet_size-1``, and the padding mask is ``-1``. If "last", last label value ``alphabet_size-1`` is reserved for blank label instead, and label values for tokens in the vocabulary are between ``0`` and ``alphabet_size-2``, and the padding mask is ``0``.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def degrees(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Converts each element of the input array from radians to degrees.

    .. math::
       degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]

    The storage type of ``degrees`` output depends upon the input storage type:

       - degrees(default) = default
       - degrees(row_sparse) = row_sparse
       - degrees(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L163

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

def depth_to_space(data=None, block_size=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Rearranges(permutes) data from depth into blocks of spatial data.
    Similar to ONNX DepthToSpace operator:
    https://github.com/onnx/onnx/blob/master/docs/Operators.md#DepthToSpace.
    The output is a new tensor where the values from depth dimension are moved in spatial blocks 
    to height and width dimension. The reverse of this operation is ``space_to_depth``.

    .. math::

        \begin{gather*}
        x \prime = reshape(x, [N, block\_size, block\_size, C / (block\_size ^ 2), H * block\_size, W * block\_size]) \\
        x \prime \prime = transpose(x \prime, [0, 3, 4, 1, 5, 2]) \\
        y = reshape(x \prime \prime, [N, C / (block\_size ^ 2), H * block\_size, W * block\_size])
        \end{gather*}

    where :math:`x` is an input tensor with default layout as :math:`[N, C, H, W]`: [batch, channels, height, width] 
    and :math:`y` is the output tensor of layout :math:`[N, C / (block\_size ^ 2), H * block\_size, W * block\_size]`

    Example::

      x = [[[[0, 1, 2],
             [3, 4, 5]],
            [[6, 7, 8],
             [9, 10, 11]],
            [[12, 13, 14],
             [15, 16, 17]],
            [[18, 19, 20],
             [21, 22, 23]]]]

      depth_to_space(x, 2) = [[[[0, 6, 1, 7, 2, 8],
                                [12, 18, 13, 19, 14, 20],
                                [3, 9, 4, 10, 5, 11],
                                [15, 21, 16, 22, 17, 23]]]]


    Defined in src/operator/tensor/matrix_op.cc:L946

    Parameters
    ----------
    data : Symbol
        Input ndarray
    block_size : int, required
        Blocks of [block_size. block_size] are moved

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def diag(data=None, k=_Null, axis1=_Null, axis2=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Extracts a diagonal or constructs a diagonal array.

    ``diag``'s behavior depends on the input array dimensions:

    - 1-D arrays: constructs a 2-D array with the input as its diagonal, all other elements are zero.
    - N-D arrays: extracts the diagonals of the sub-arrays with axes specified by ``axis1`` and ``axis2``.
      The output shape would be decided by removing the axes numbered ``axis1`` and ``axis2`` from the
      input shape and appending to the result a new axis with the size of the diagonals in question.

      For example, when the input shape is `(2, 3, 4, 5)`, ``axis1`` and ``axis2`` are 0 and 2
      respectively and ``k`` is 0, the resulting shape would be `(3, 5, 2)`.

    Examples::

      x = [[1, 2, 3],
           [4, 5, 6]]

      diag(x) = [1, 5]

      diag(x, k=1) = [2, 6]

      diag(x, k=-1) = [4]

      x = [1, 2, 3]

      diag(x) = [[1, 0, 0],
                 [0, 2, 0],
                 [0, 0, 3]]

      diag(x, k=1) = [[0, 1, 0],
                      [0, 0, 2],
                      [0, 0, 0]]

      diag(x, k=-1) = [[0, 0, 0],
                       [1, 0, 0],
                       [0, 2, 0]]

      x = [[[1, 2],
            [3, 4]],

           [[5, 6],
            [7, 8]]]

      diag(x) = [[1, 7],
                 [2, 8]]

      diag(x, k=1) = [[3],
                      [4]]

      diag(x, axis1=-2, axis2=-1) = [[1, 4],
                                     [5, 8]]



    Defined in src/operator/tensor/diag_op.cc:L87

    Parameters
    ----------
    data : Symbol
        Input ndarray
    k : int, optional, default='0'
        Diagonal in question. The default is 0. Use k>0 for diagonals above the main diagonal, and k<0 for diagonals below the main diagonal. If input has shape (S0 S1) k must be between -S0 and S1
    axis1 : int, optional, default='0'
        The first axis of the sub-arrays of interest. Ignored when the input is a 1-D array.
    axis2 : int, optional, default='1'
        The second axis of the sub-arrays of interest. Ignored when the input is a 1-D array.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def dot(lhs=None, rhs=None, transpose_a=_Null, transpose_b=_Null, forward_stype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Dot product of two arrays.

    ``dot``'s behavior depends on the input array dimensions:

    - 1-D arrays: inner product of vectors
    - 2-D arrays: matrix multiplication
    - N-D arrays: a sum product over the last axis of the first input and the first
      axis of the second input

      For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape `(k,r,s)`, the
      result array will have shape `(n,m,r,s)`. It is computed by::

        dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])

      Example::

        x = reshape([0,1,2,3,4,5,6,7], shape=(2,2,2))
        y = reshape([7,6,5,4,3,2,1,0], shape=(2,2,2))
        dot(x,y)[0,0,1,1] = 0
        sum(x[0,0,:]*y[:,1,1]) = 0

    The storage type of ``dot`` output depends on storage types of inputs, transpose option and
    forward_stype option for output storage type. Implemented sparse operations include:

    - dot(default, default, transpose_a=True/False, transpose_b=True/False) = default
    - dot(csr, default, transpose_a=True) = default
    - dot(csr, default, transpose_a=True) = row_sparse
    - dot(csr, default) = default
    - dot(csr, row_sparse) = default
    - dot(default, csr) = csr (CPU only)
    - dot(default, csr, forward_stype='default') = default
    - dot(default, csr, transpose_b=True, forward_stype='default') = default

    If the combination of input storage types and forward_stype does not match any of the
    above patterns, ``dot`` will fallback and generate output with default storage.

    .. Note::

        If the storage type of the lhs is "csr", the storage type of gradient w.r.t rhs will be
        "row_sparse". Only a subset of optimizers support sparse gradients, including SGD, AdaGrad
        and Adam. Note that by default lazy updates is turned on, which may perform differently
        from standard updates. For more details, please check the Optimization API at:
        https://mxnet.incubator.apache.org/api/python/optimization/optimization.html



    Defined in src/operator/tensor/dot.cc:L77

    Parameters
    ----------
    lhs : Symbol
        The first input
    rhs : Symbol
        The second input
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

def elemwise_add(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
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

def elemwise_div(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
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

def elemwise_mul(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
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

def elemwise_sub(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
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

def erf(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise gauss error function of the input.

    Example::

       erf([0, -1., 10.]) = [0., -0.8427, 1.]



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L897

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

def exp(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise exponential value of the input.

    .. math::
       exp(x) = e^x \approx 2.718^x

    Example::

       exp([0, 1, 2]) = [1., 2.71828175, 7.38905621]

    The storage type of ``exp`` output is always dense



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L939

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

def expand_dims(data=None, axis=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Inserts a new axis of size 1 into the array shape

    For example, given ``x`` with shape ``(2,3,4)``, then ``expand_dims(x, axis=1)``
    will return a new array with shape ``(2,1,3,4)``.



    Defined in src/operator/tensor/matrix_op.cc:L348

    Parameters
    ----------
    data : Symbol
        Source input
    axis : int, required
        Position where new axis is to be inserted. Suppose that the input `NDArray`'s dimension is `ndim`, the range of the inserted axis is `[-ndim, ndim]`

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def expm1(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns ``exp(x) - 1`` computed element-wise on the input.

    This function provides greater precision than ``exp(x) - 1`` for small values of ``x``.

    The storage type of ``expm1`` output depends upon the input storage type:

       - expm1(default) = default
       - expm1(row_sparse) = row_sparse
       - expm1(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L1018

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

def fill_element_0index(lhs=None, mhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Fill one element of each line(row for python, column for R/Julia) in lhs according to index indicated by rhs and values indicated by mhs. This function assume rhs uses 0-based index.

    Parameters
    ----------
    lhs : NDArray
        Left operand to the function.
    mhs : NDArray
        Middle operand to the function.
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

def fix(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise rounded value to the nearest \
    integer towards zero of the input.

    Example::

       fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]

    The storage type of ``fix`` output depends upon the input storage type:

       - fix(default) = default
       - fix(row_sparse) = row_sparse
       - fix(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L797

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

def flatten(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Flattens the input array into a 2-D array by collapsing the higher dimensions.

    .. note:: `Flatten` is deprecated. Use `flatten` instead.

    For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation reshapes
    the input array into an output array of shape ``(d1, d2*...*dk)``.

    Note that the bahavior of this function is different from numpy.ndarray.flatten,
    which behaves similar to mxnet.ndarray.reshape((-1,)).

    Example::

        x = [[
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ],
        [    [1,2,3],
            [4,5,6],
            [7,8,9]
        ]],

        flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
           [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]



    Defined in src/operator/tensor/matrix_op.cc:L259

    Parameters
    ----------
    data : Symbol
        Input array.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def flip(data=None, axis=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Reverses the order of elements along given axis while preserving array shape.

    Note: reverse and flip are equivalent. We use reverse in the following examples.

    Examples::

      x = [[ 0.,  1.,  2.,  3.,  4.],
           [ 5.,  6.,  7.,  8.,  9.]]

      reverse(x, axis=0) = [[ 5.,  6.,  7.,  8.,  9.],
                            [ 0.,  1.,  2.,  3.,  4.]]

      reverse(x, axis=1) = [[ 4.,  3.,  2.,  1.,  0.],
                            [ 9.,  8.,  7.,  6.,  5.]]


    Defined in src/operator/tensor/matrix_op.cc:L794

    Parameters
    ----------
    data : Symbol
        Input data array
    axis : Shape(tuple), required
        The axis which to reverse elements.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def floor(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise floor of the input.

    The floor of the scalar x is the largest integer i, such that i <= x.

    Example::

       floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]

    The storage type of ``floor`` output depends upon the input storage type:

       - floor(default) = default
       - floor(row_sparse) = row_sparse
       - floor(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L759

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

def ftml_update(weight=None, grad=None, d=None, v=None, z=None, lr=_Null, beta1=_Null, beta2=_Null, epsilon=_Null, t=_Null, wd=_Null, rescale_grad=_Null, clip_grad=_Null, name=None, attr=None, out=None, **kwargs):
    r"""The FTML optimizer described in
    *FTML - Follow the Moving Leader in Deep Learning*,
    available at http://proceedings.mlr.press/v70/zheng17a/zheng17a.pdf.

    .. math::

     g_t = \nabla J(W_{t-1})\\
     v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
     d_t = \frac{ 1 - \beta_1^t }{ \eta_t } (\sqrt{ \frac{ v_t }{ 1 - \beta_2^t } } + \epsilon)
     \sigma_t = d_t - \beta_1 d_{t-1}
     z_t = \beta_1 z_{ t-1 } + (1 - \beta_1^t) g_t - \sigma_t W_{t-1}
     W_t = - \frac{ z_t }{ d_t }



    Defined in src/operator/optimizer_op.cc:L447

    Parameters
    ----------
    weight : Symbol
        Weight
    grad : Symbol
        Gradient
    d : Symbol
        Internal state ``d_t``
    v : Symbol
        Internal state ``v_t``
    z : Symbol
        Internal state ``z_t``
    lr : float, required
        Learning rate.
    beta1 : float, optional, default=0.6
        Generally close to 0.5.
    beta2 : float, optional, default=0.999
        Generally close to 1.
    epsilon : double, optional, default=1e-08
        Epsilon to prevent div 0.
    t : int, required
        Number of update.
    wd : float, optional, default=0
        Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
    rescale_grad : float, optional, default=1
        Rescale gradient to grad = rescale_grad*grad.
    clip_grad : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def ftrl_update(weight=None, grad=None, z=None, n=None, lr=_Null, lamda1=_Null, beta=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Update function for Ftrl optimizer.
    Referenced from *Ad Click Prediction: a View from the Trenches*, available at
    http://dl.acm.org/citation.cfm?id=2488200.

    It updates the weights using::

     rescaled_grad = clip(grad * rescale_grad, clip_gradient)
     z += rescaled_grad - (sqrt(n + rescaled_grad**2) - sqrt(n)) * weight / learning_rate
     n += rescaled_grad**2
     w = (sign(z) * lamda1 - z) / ((beta + sqrt(n)) / learning_rate + wd) * (abs(z) > lamda1)

    If w, z and n are all of ``row_sparse`` storage type,
    only the row slices whose indices appear in grad.indices are updated (for w, z and n)::

     for row in grad.indices:
         rescaled_grad[row] = clip(grad[row] * rescale_grad, clip_gradient)
         z[row] += rescaled_grad[row] - (sqrt(n[row] + rescaled_grad[row]**2) - sqrt(n[row])) * weight[row] / learning_rate
         n[row] += rescaled_grad[row]**2
         w[row] = (sign(z[row]) * lamda1 - z[row]) / ((beta + sqrt(n[row])) / learning_rate + wd) * (abs(z[row]) > lamda1)



    Defined in src/operator/optimizer_op.cc:L632

    Parameters
    ----------
    weight : Symbol
        Weight
    grad : Symbol
        Gradient
    z : Symbol
        z
    n : Symbol
        Square of grad
    lr : float, required
        Learning rate
    lamda1 : float, optional, default=0.01
        The L1 regularization coefficient.
    beta : float, optional, default=1
        Per-Coordinate Learning Rate beta.
    wd : float, optional, default=0
        Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
    rescale_grad : float, optional, default=1
        Rescale gradient to grad = rescale_grad*grad.
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def gamma(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns the gamma function (extension of the factorial function \
    to the reals), computed element-wise on the input array.

    The storage type of ``gamma`` output is always dense



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

def gammaln(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise log of the absolute value of the gamma function \
    of the input.

    The storage type of ``gammaln`` output is always dense



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

def gather_nd(data=None, indices=None, name=None, attr=None, out=None, **kwargs):
    r"""Gather elements or slices from `data` and store to a tensor whose
    shape is defined by `indices`.

    Given `data` with shape `(X_0, X_1, ..., X_{N-1})` and indices with shape
    `(M, Y_0, ..., Y_{K-1})`, the output will have shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})`,
    where `M <= N`. If `M == N`, output shape will simply be `(Y_0, ..., Y_{K-1})`.

    The elements in output is defined as follows::

      output[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}] = data[indices[0, y_0, ..., y_{K-1}],
                                                          ...,
                                                          indices[M-1, y_0, ..., y_{K-1}],
                                                          x_M, ..., x_{N-1}]

    Examples::

      data = [[0, 1], [2, 3]]
      indices = [[1, 1, 0], [0, 1, 0]]
      gather_nd(data, indices) = [2, 3, 0]

      data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
      indices = [[0, 1], [1, 0]]
      gather_nd(data, indices) = [[3, 4], [5, 6]]



    Parameters
    ----------
    data : Symbol
        data
    indices : Symbol
        indices

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def hard_sigmoid(data=None, alpha=_Null, beta=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Computes hard sigmoid of x element-wise.

    .. math::
       y = max(0, min(1, alpha * x + beta))



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L115

    Parameters
    ----------
    data : Symbol
        The input array.
    alpha : float, optional, default=0.2
        Slope of hard sigmoid
    beta : float, optional, default=0.5
        Bias of hard sigmoid.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def identity(data=None, name=None, attr=None, out=None, **kwargs):
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

def khatri_rao(*args, **kwargs):
    r"""Computes the Khatri-Rao product of the input matrices.

    Given a collection of :math:`n` input matrices,

    .. math::
       A_1 \in \mathbb{R}^{M_1 \times M}, \ldots, A_n \in \mathbb{R}^{M_n \times N},

    the (column-wise) Khatri-Rao product is defined as the matrix,

    .. math::
       X = A_1 \otimes \cdots \otimes A_n \in \mathbb{R}^{(M_1 \cdots M_n) \times N},

    where the :math:`k` th column is equal to the column-wise outer product
    :math:`{A_1}_k \otimes \cdots \otimes {A_n}_k` where :math:`{A_i}_k` is the kth
    column of the ith matrix.

    Example::

      >>> A = mx.nd.array([[1, -1],
      >>>                  [2, -3]])
      >>> B = mx.nd.array([[1, 4],
      >>>                  [2, 5],
      >>>                  [3, 6]])
      >>> C = mx.nd.khatri_rao(A, B)
      >>> print(C.asnumpy())
      [[  1.  -4.]
       [  2.  -5.]
       [  3.  -6.]
       [  2. -12.]
       [  4. -15.]
       [  6. -18.]]



    Defined in src/operator/contrib/krprod.cc:L108
    This function support variable length of positional input.

    Parameters
    ----------
    args : Symbol[]
        Positional input matrices

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def linalg_gelqf(A=None, name=None, attr=None, out=None, **kwargs):
    r"""LQ factorization for general matrix.
    Input is a tensor *A* of dimension *n >= 2*.

    If *n=2*, we compute the LQ factorization (LAPACK *gelqf*, followed by *orglq*). *A*
    must have shape *(x, y)* with *x <= y*, and must have full rank *=x*. The LQ
    factorization consists of *L* with shape *(x, x)* and *Q* with shape *(x, y)*, so
    that:

       *A* = *L* \* *Q*

    Here, *L* is lower triangular (upper triangle equal to zero) with nonzero diagonal,
    and *Q* is row-orthonormal, meaning that

       *Q* \* *Q*\ :sup:`T`

    is equal to the identity matrix of shape *(x, x)*.

    If *n>2*, *gelqf* is performed separately on the trailing two dimensions for all
    inputs (batch mode).

    .. note:: The operator supports float32 and float64 data types only.

    Examples::

       // Single LQ factorization
       A = [[1., 2., 3.], [4., 5., 6.]]
       Q, L = gelqf(A)
       Q = [[-0.26726124, -0.53452248, -0.80178373],
            [0.87287156, 0.21821789, -0.43643578]]
       L = [[-3.74165739, 0.],
            [-8.55235974, 1.96396101]]

       // Batch LQ factorization
       A = [[[1., 2., 3.], [4., 5., 6.]],
            [[7., 8., 9.], [10., 11., 12.]]]
       Q, L = gelqf(A)
       Q = [[[-0.26726124, -0.53452248, -0.80178373],
             [0.87287156, 0.21821789, -0.43643578]],
            [[-0.50257071, -0.57436653, -0.64616234],
             [0.7620735, 0.05862104, -0.64483142]]]
       L = [[[-3.74165739, 0.],
             [-8.55235974, 1.96396101]],
            [[-13.92838828, 0.],
             [-19.09768702, 0.52758934]]]


    Defined in src/operator/tensor/la_op.cc:L569

    Parameters
    ----------
    A : Symbol
        Tensor of input matrices to be factorized

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def linalg_gemm(A=None, B=None, C=None, transpose_a=_Null, transpose_b=_Null, alpha=_Null, beta=_Null, axis=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Performs general matrix multiplication and accumulation.
    Input are tensors *A*, *B*, *C*, each of dimension *n >= 2* and having the same shape
    on the leading *n-2* dimensions.

    If *n=2*, the BLAS3 function *gemm* is performed:

       *out* = *alpha* \* *op*\ (*A*) \* *op*\ (*B*) + *beta* \* *C*

    Here, *alpha* and *beta* are scalar parameters, and *op()* is either the identity or
    matrix transposition (depending on *transpose_a*, *transpose_b*).

    If *n>2*, *gemm* is performed separately for a batch of matrices. The column indices of the matrices
    are given by the last dimensions of the tensors, the row indices by the axis specified with the *axis* 
    parameter. By default, the trailing two dimensions will be used for matrix encoding.

    For a non-default axis parameter, the operation performed is equivalent to a series of swapaxes/gemm/swapaxes
    calls. For example let *A*, *B*, *C* be 5 dimensional tensors. Then gemm(*A*, *B*, *C*, axis=1) is equivalent to

        A1 = swapaxes(A, dim1=1, dim2=3)
        B1 = swapaxes(B, dim1=1, dim2=3)
        C = swapaxes(C, dim1=1, dim2=3)
        C = gemm(A1, B1, C)
        C = swapaxis(C, dim1=1, dim2=3)

    without the overhead of the additional swapaxis operations.

    When the input data is of type float32 and the environment variables MXNET_CUDA_ALLOW_TENSOR_CORE
    and MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION are set to 1, this operator will try to use
    pseudo-float16 precision (float32 math with float16 I/O) precision in order to use
    Tensor Cores on suitable NVIDIA GPUs. This can sometimes give significant speedups.

    .. note:: The operator supports float32 and float64 data types only.

    Examples::

       // Single matrix multiply-add
       A = [[1.0, 1.0], [1.0, 1.0]]
       B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
       C = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
       gemm(A, B, C, transpose_b=True, alpha=2.0, beta=10.0)
               = [[14.0, 14.0, 14.0], [14.0, 14.0, 14.0]]

       // Batch matrix multiply-add
       A = [[[1.0, 1.0]], [[0.1, 0.1]]]
       B = [[[1.0, 1.0]], [[0.1, 0.1]]]
       C = [[[10.0]], [[0.01]]]
       gemm(A, B, C, transpose_b=True, alpha=2.0 , beta=10.0)
               = [[[104.0]], [[0.14]]]


    Defined in src/operator/tensor/la_op.cc:L87

    Parameters
    ----------
    A : Symbol
        Tensor of input matrices
    B : Symbol
        Tensor of input matrices
    C : Symbol
        Tensor of input matrices
    transpose_a : boolean, optional, default=0
        Multiply with transposed of first input (A).
    transpose_b : boolean, optional, default=0
        Multiply with transposed of second input (B).
    alpha : double, optional, default=1
        Scalar factor multiplied with A*B.
    beta : double, optional, default=1
        Scalar factor multiplied with C.
    axis : int, optional, default='-2'
        Axis corresponding to the matrix rows.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def linalg_gemm2(A=None, B=None, transpose_a=_Null, transpose_b=_Null, alpha=_Null, axis=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Performs general matrix multiplication.
    Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape
    on the leading *n-2* dimensions.

    If *n=2*, the BLAS3 function *gemm* is performed:

       *out* = *alpha* \* *op*\ (*A*) \* *op*\ (*B*)

    Here *alpha* is a scalar parameter and *op()* is either the identity or the matrix
    transposition (depending on *transpose_a*, *transpose_b*).

    If *n>2*, *gemm* is performed separately for a batch of matrices. The column indices of the matrices
    are given by the last dimensions of the tensors, the row indices by the axis specified with the *axis* 
    parameter. By default, the trailing two dimensions will be used for matrix encoding.

    For a non-default axis parameter, the operation performed is equivalent to a series of swapaxes/gemm/swapaxes
    calls. For example let *A*, *B* be 5 dimensional tensors. Then gemm(*A*, *B*, axis=1) is equivalent to

        A1 = swapaxes(A, dim1=1, dim2=3)
        B1 = swapaxes(B, dim1=1, dim2=3)
        C = gemm2(A1, B1)
        C = swapaxis(C, dim1=1, dim2=3)

    without the overhead of the additional swapaxis operations.

    When the input data is of type float32 and the environment variables MXNET_CUDA_ALLOW_TENSOR_CORE
    and MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION are set to 1, this operator will try to use
    pseudo-float16 precision (float32 math with float16 I/O) precision in order to use
    Tensor Cores on suitable NVIDIA GPUs. This can sometimes give significant speedups.

    .. note:: The operator supports float32 and float64 data types only.

    Examples::

       // Single matrix multiply
       A = [[1.0, 1.0], [1.0, 1.0]]
       B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
       gemm2(A, B, transpose_b=True, alpha=2.0)
                = [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]

       // Batch matrix multiply
       A = [[[1.0, 1.0]], [[0.1, 0.1]]]
       B = [[[1.0, 1.0]], [[0.1, 0.1]]]
       gemm2(A, B, transpose_b=True, alpha=2.0)
               = [[[4.0]], [[0.04 ]]]


    Defined in src/operator/tensor/la_op.cc:L162

    Parameters
    ----------
    A : Symbol
        Tensor of input matrices
    B : Symbol
        Tensor of input matrices
    transpose_a : boolean, optional, default=0
        Multiply with transposed of first input (A).
    transpose_b : boolean, optional, default=0
        Multiply with transposed of second input (B).
    alpha : double, optional, default=1
        Scalar factor multiplied with A*B.
    axis : int, optional, default='-2'
        Axis corresponding to the matrix row indices.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def linalg_potrf(A=None, name=None, attr=None, out=None, **kwargs):
    r"""Performs Cholesky factorization of a symmetric positive-definite matrix.
    Input is a tensor *A* of dimension *n >= 2*.

    If *n=2*, the Cholesky factor *B* of the symmetric, positive definite matrix *A* is
    computed. *B* is triangular (entries of upper or lower triangle are all zero), has
    positive diagonal entries, and:

      *A* = *B* \* *B*\ :sup:`T`  if *lower* = *true*
      *A* = *B*\ :sup:`T` \* *B*  if *lower* = *false*

    If *n>2*, *potrf* is performed separately on the trailing two dimensions for all inputs
    (batch mode).

    .. note:: The operator supports float32 and float64 data types only.

    Examples::

       // Single matrix factorization
       A = [[4.0, 1.0], [1.0, 4.25]]
       potrf(A) = [[2.0, 0], [0.5, 2.0]]

       // Batch matrix factorization
       A = [[[4.0, 1.0], [1.0, 4.25]], [[16.0, 4.0], [4.0, 17.0]]]
       potrf(A) = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]


    Defined in src/operator/tensor/la_op.cc:L213

    Parameters
    ----------
    A : Symbol
        Tensor of input matrices to be decomposed

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def linalg_potri(A=None, name=None, attr=None, out=None, **kwargs):
    r"""Performs matrix inversion from a Cholesky factorization.
    Input is a tensor *A* of dimension *n >= 2*.

    If *n=2*, *A* is a triangular matrix (entries of upper or lower triangle are all zero)
    with positive diagonal. We compute:

      *out* = *A*\ :sup:`-T` \* *A*\ :sup:`-1` if *lower* = *true*
      *out* = *A*\ :sup:`-1` \* *A*\ :sup:`-T` if *lower* = *false*

    In other words, if *A* is the Cholesky factor of a symmetric positive definite matrix
    *B* (obtained by *potrf*), then

      *out* = *B*\ :sup:`-1`

    If *n>2*, *potri* is performed separately on the trailing two dimensions for all inputs
    (batch mode).

    .. note:: The operator supports float32 and float64 data types only.

    .. note:: Use this operator only if you are certain you need the inverse of *B*, and
              cannot use the Cholesky factor *A* (*potrf*), together with backsubstitution
              (*trsm*). The latter is numerically much safer, and also cheaper.

    Examples::

       // Single matrix inverse
       A = [[2.0, 0], [0.5, 2.0]]
       potri(A) = [[0.26563, -0.0625], [-0.0625, 0.25]]

       // Batch matrix inverse
       A = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]
       potri(A) = [[[0.26563, -0.0625], [-0.0625, 0.25]],
                   [[0.06641, -0.01562], [-0.01562, 0,0625]]]


    Defined in src/operator/tensor/la_op.cc:L274

    Parameters
    ----------
    A : Symbol
        Tensor of lower triangular matrices

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def linalg_sumlogdiag(A=None, name=None, attr=None, out=None, **kwargs):
    r"""Computes the sum of the logarithms of the diagonal elements of a square matrix.
    Input is a tensor *A* of dimension *n >= 2*.

    If *n=2*, *A* must be square with positive diagonal entries. We sum the natural
    logarithms of the diagonal elements, the result has shape (1,).

    If *n>2*, *sumlogdiag* is performed separately on the trailing two dimensions for all
    inputs (batch mode).

    .. note:: The operator supports float32 and float64 data types only.

    Examples::

       // Single matrix reduction
       A = [[1.0, 1.0], [1.0, 7.0]]
       sumlogdiag(A) = [1.9459]

       // Batch matrix reduction
       A = [[[1.0, 1.0], [1.0, 7.0]], [[3.0, 0], [0, 17.0]]]
       sumlogdiag(A) = [1.9459, 3.9318]


    Defined in src/operator/tensor/la_op.cc:L445

    Parameters
    ----------
    A : Symbol
        Tensor of square matrices

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def linalg_syrk(A=None, transpose=_Null, alpha=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Multiplication of matrix with its transpose.
    Input is a tensor *A* of dimension *n >= 2*.

    If *n=2*, the operator performs the BLAS3 function *syrk*:

      *out* = *alpha* \* *A* \* *A*\ :sup:`T`

    if *transpose=False*, or

      *out* = *alpha* \* *A*\ :sup:`T` \ \* *A*

    if *transpose=True*.

    If *n>2*, *syrk* is performed separately on the trailing two dimensions for all
    inputs (batch mode).

    .. note:: The operator supports float32 and float64 data types only.

    Examples::

       // Single matrix multiply
       A = [[1., 2., 3.], [4., 5., 6.]]
       syrk(A, alpha=1., transpose=False)
                = [[14., 32.],
                   [32., 77.]]
       syrk(A, alpha=1., transpose=True)
                = [[17., 22., 27.],
                   [22., 29., 36.],
                   [27., 36., 45.]]

       // Batch matrix multiply
       A = [[[1., 1.]], [[0.1, 0.1]]]
       syrk(A, alpha=2., transpose=False) = [[[4.]], [[0.04]]]


    Defined in src/operator/tensor/la_op.cc:L501

    Parameters
    ----------
    A : Symbol
        Tensor of input matrices
    transpose : boolean, optional, default=0
        Use transpose of input matrix.
    alpha : double, optional, default=1
        Scalar factor to be applied to the result.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def linalg_trmm(A=None, B=None, transpose=_Null, rightside=_Null, lower=_Null, alpha=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Performs multiplication with a lower triangular matrix.
    Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape
    on the leading *n-2* dimensions.

    If *n=2*, *A* must be triangular. The operator performs the BLAS3 function
    *trmm*:

       *out* = *alpha* \* *op*\ (*A*) \* *B*

    if *rightside=False*, or

       *out* = *alpha* \* *B* \* *op*\ (*A*)

    if *rightside=True*. Here, *alpha* is a scalar parameter, and *op()* is either the
    identity or the matrix transposition (depending on *transpose*).

    If *n>2*, *trmm* is performed separately on the trailing two dimensions for all inputs
    (batch mode).

    .. note:: The operator supports float32 and float64 data types only.


    Examples::

       // Single triangular matrix multiply
       A = [[1.0, 0], [1.0, 1.0]]
       B = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
       trmm(A, B, alpha=2.0) = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]

       // Batch triangular matrix multiply
       A = [[[1.0, 0], [1.0, 1.0]], [[1.0, 0], [1.0, 1.0]]]
       B = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]
       trmm(A, B, alpha=2.0) = [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
                                [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]


    Defined in src/operator/tensor/la_op.cc:L333

    Parameters
    ----------
    A : Symbol
        Tensor of lower triangular matrices
    B : Symbol
        Tensor of matrices
    transpose : boolean, optional, default=0
        Use transposed of the triangular matrix
    rightside : boolean, optional, default=0
        Multiply triangular matrix from the right to non-triangular one.
    lower : boolean, optional, default=1
        True if the triangular matrix is lower triangular, false if it is upper triangular.
    alpha : double, optional, default=1
        Scalar factor to be applied to the result.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def linalg_trsm(A=None, B=None, transpose=_Null, rightside=_Null, lower=_Null, alpha=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Solves matrix equation involving a lower triangular matrix.
    Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape
    on the leading *n-2* dimensions.

    If *n=2*, *A* must be triangular. The operator performs the BLAS3 function
    *trsm*, solving for *out* in:

       *op*\ (*A*) \* *out* = *alpha* \* *B*

    if *rightside=False*, or

       *out* \* *op*\ (*A*) = *alpha* \* *B*

    if *rightside=True*. Here, *alpha* is a scalar parameter, and *op()* is either the
    identity or the matrix transposition (depending on *transpose*).

    If *n>2*, *trsm* is performed separately on the trailing two dimensions for all inputs
    (batch mode).

    .. note:: The operator supports float32 and float64 data types only.

    Examples::

       // Single matrix solve
       A = [[1.0, 0], [1.0, 1.0]]
       B = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
       trsm(A, B, alpha=0.5) = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

       // Batch matrix solve
       A = [[[1.0, 0], [1.0, 1.0]], [[1.0, 0], [1.0, 1.0]]]
       B = [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
            [[4.0, 4.0, 4.0], [8.0, 8.0, 8.0]]]
       trsm(A, B, alpha=0.5) = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                                [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]


    Defined in src/operator/tensor/la_op.cc:L396

    Parameters
    ----------
    A : Symbol
        Tensor of lower triangular matrices
    B : Symbol
        Tensor of matrices
    transpose : boolean, optional, default=0
        Use transposed of the triangular matrix
    rightside : boolean, optional, default=0
        Multiply triangular matrix from the right to non-triangular one.
    lower : boolean, optional, default=1
        True if the triangular matrix is lower triangular, false if it is upper triangular.
    alpha : double, optional, default=1
        Scalar factor to be applied to the result.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def log(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise Natural logarithmic value of the input.

    The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``

    The storage type of ``log`` output is always dense



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L951

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

def log10(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise Base-10 logarithmic value of the input.

    ``10**log10(x) = x``

    The storage type of ``log10`` output is always dense



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L963

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

def log1p(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise ``log(1 + x)`` value of the input.

    This function is more accurate than ``log(1 + x)``  for small ``x`` so that
    :math:`1+x\approx 1`

    The storage type of ``log1p`` output depends upon the input storage type:

       - log1p(default) = default
       - log1p(row_sparse) = row_sparse
       - log1p(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L1000

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

def log2(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise Base-2 logarithmic value of the input.

    ``2**log2(x) = x``

    The storage type of ``log2`` output is always dense



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L975

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

def log_softmax(data=None, axis=_Null, temperature=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Computes the log softmax of the input.
    This is equivalent to computing softmax followed by log.

    Examples::

      >>> x = mx.nd.array([1, 2, .1])
      >>> mx.nd.log_softmax(x).asnumpy()
      array([-1.41702998, -0.41702995, -2.31702995], dtype=float32)

      >>> x = mx.nd.array( [[1, 2, .1],[.1, 2, 1]] )
      >>> mx.nd.log_softmax(x, axis=0).asnumpy()
      array([[-0.34115392, -0.69314718, -1.24115396],
             [-1.24115396, -0.69314718, -0.34115392]], dtype=float32)




    Parameters
    ----------
    data : Symbol
        The input array.
    axis : int, optional, default='-1'
        The axis along which to compute softmax.
    temperature : double or None, optional, default=None
        Temperature parameter in softmax

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def logical_not(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns the result of logical NOT (!) function

    Example:
      logical_not([-2., 0., 1.]) = [0., 1., 0.]



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

def make_loss(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Make your own loss function in network construction.

    This operator accepts a customized loss function symbol as a terminal loss and
    the symbol should be an operator with no backward dependency.
    The output of this function is the gradient of loss with respect to the input data.

    For example, if you are a making a cross entropy loss function. Assume ``out`` is the
    predicted output and ``label`` is the true label, then the cross entropy can be defined as::

      cross_entropy = label * log(out) + (1 - label) * log(1 - out)
      loss = make_loss(cross_entropy)

    We will need to use ``make_loss`` when we are creating our own loss function or we want to
    combine multiple loss functions. Also we may want to stop some variables' gradients
    from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.

    The storage type of ``make_loss`` output depends upon the input storage type:

       - make_loss(default) = default
       - make_loss(row_sparse) = row_sparse



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L300

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

def max(data=None, axis=_Null, keepdims=_Null, exclude=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Computes the max of array elements over given axes.

    Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L191

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

def max_axis(data=None, axis=_Null, keepdims=_Null, exclude=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Computes the max of array elements over given axes.

    Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L191

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

def mean(data=None, axis=_Null, keepdims=_Null, exclude=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Computes the mean of array elements over given axes.

    Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L132

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

def min(data=None, axis=_Null, keepdims=_Null, exclude=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Computes the min of array elements over given axes.

    Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L205

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

def min_axis(data=None, axis=_Null, keepdims=_Null, exclude=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Computes the min of array elements over given axes.

    Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L205

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

def mp_sgd_mom_update(weight=None, grad=None, mom=None, weight32=None, lr=_Null, momentum=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, lazy_update=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Updater function for multi-precision sgd optimizer

    Parameters
    ----------
    weight : Symbol
        Weight
    grad : Symbol
        Gradient
    mom : Symbol
        Momentum
    weight32 : Symbol
        Weight32
    lr : float, required
        Learning rate
    momentum : float, optional, default=0
        The decay rate of momentum estimates at each epoch.
    wd : float, optional, default=0
        Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
    rescale_grad : float, optional, default=1
        Rescale gradient to grad = rescale_grad*grad.
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
    lazy_update : boolean, optional, default=1
        If true, lazy updates are applied if gradient's stype is row_sparse and both weight and momentum have the same stype

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def mp_sgd_update(weight=None, grad=None, weight32=None, lr=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, lazy_update=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Updater function for multi-precision sgd optimizer

    Parameters
    ----------
    weight : Symbol
        Weight
    grad : Symbol
        gradient
    weight32 : Symbol
        Weight32
    lr : float, required
        Learning rate
    wd : float, optional, default=0
        Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
    rescale_grad : float, optional, default=1
        Rescale gradient to grad = rescale_grad*grad.
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
    lazy_update : boolean, optional, default=1
        If true, lazy updates are applied if gradient's stype is row_sparse.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def nanprod(data=None, axis=_Null, keepdims=_Null, exclude=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Computes the product of array elements over given axes treating Not a Numbers (``NaN``) as one.



    Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L177

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

def nansum(data=None, axis=_Null, keepdims=_Null, exclude=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Computes the sum of array elements over given axes treating Not a Numbers (``NaN``) as zero.



    Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L162

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

def negative(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Numerical negative of the argument, element-wise.

    The storage type of ``negative`` output depends upon the input storage type:

       - negative(default) = default
       - negative(row_sparse) = row_sparse
       - negative(csr) = csr



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

def norm(data=None, ord=_Null, axis=_Null, keepdims=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Computes the norm on an NDArray.

    This operator computes the norm on an NDArray with the specified axis, depending
    on the value of the ord parameter. By default, it computes the L2 norm on the entire
    array. Currently only ord=2 supports sparse ndarrays.

    Examples::

      x = [[[1, 2],
            [3, 4]],
           [[2, 2],
            [5, 6]]]

      norm(x, ord=2, axis=1) = [[3.1622777 4.472136 ]
                                [5.3851647 6.3245554]]

      norm(x, ord=1, axis=1) = [[4., 6.],
                                [7., 8.]]

      rsp = x.cast_storage('row_sparse')

      norm(rsp) = [5.47722578]

      csr = x.cast_storage('csr')

      norm(csr) = [5.47722578]



    Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L350

    Parameters
    ----------
    data : Symbol
        The input
    ord : int, optional, default='2'
        Order of the norm. Currently ord=1 and ord=2 is supported.
    axis : Shape or None, optional, default=None
        The axis or axes along which to perform the reduction.
          The default, `axis=()`, will compute over all elements into a
          scalar array with shape `(1,)`.
          If `axis` is int, a reduction is performed on a particular axis.
          If `axis` is a 2-tuple, it specifies the axes that hold 2-D matrices,
          and the matrix norms of these matrices are computed.
    keepdims : boolean, optional, default=0
        If this is set to `True`, the reduced axis is left in the result as dimension with size one.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def normal(loc=_Null, scale=_Null, shape=_Null, ctx=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
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

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def one_hot(indices=None, depth=_Null, on_value=_Null, off_value=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Returns a one-hot array.

    The locations represented by `indices` take value `on_value`, while all
    other locations take value `off_value`.

    `one_hot` operation with `indices` of shape ``(i0, i1)`` and `depth`  of ``d`` would result
    in an output array of shape ``(i0, i1, d)`` with::

      output[i,j,:] = off_value
      output[i,j,indices[i,j]] = on_value

    Examples::

      one_hot([1,0,2,0], 3) = [[ 0.  1.  0.]
                               [ 1.  0.  0.]
                               [ 0.  0.  1.]
                               [ 1.  0.  0.]]

      one_hot([1,0,2,0], 3, on_value=8, off_value=1,
              dtype='int32') = [[1 8 1]
                                [8 1 1]
                                [1 1 8]
                                [8 1 1]]

      one_hot([[1,0],[1,0],[2,0]], 3) = [[[ 0.  1.  0.]
                                          [ 1.  0.  0.]]

                                         [[ 0.  1.  0.]
                                          [ 1.  0.  0.]]

                                         [[ 0.  0.  1.]
                                          [ 1.  0.  0.]]]


    Defined in src/operator/tensor/indexing_op.cc:L796

    Parameters
    ----------
    indices : Symbol
        array of locations where to set on_value
    depth : int, required
        Depth of the one hot dimension.
    on_value : double, optional, default=1
        The value assigned to the locations represented by indices.
    off_value : double, optional, default=0
        The value assigned to the locations not represented by indices.
    dtype : {'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'},optional, default='float32'
        DType of the output

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def ones_like(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Return an array of ones with the same shape and type
    as the input array.

    Examples::

      x = [[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]]

      ones_like(x) = [[ 1.,  1.,  1.],
                      [ 1.,  1.,  1.]]



    Parameters
    ----------
    data : Symbol
        The input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def pad(data=None, mode=_Null, pad_width=_Null, constant_value=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Pads an input array with a constant or edge values of the array.

    .. note:: `Pad` is deprecated. Use `pad` instead.

    .. note:: Current implementation only supports 4D and 5D input arrays with padding applied
       only on axes 1, 2 and 3. Expects axes 4 and 5 in `pad_width` to be zero.

    This operation pads an input array with either a `constant_value` or edge values
    along each axis of the input array. The amount of padding is specified by `pad_width`.

    `pad_width` is a tuple of integer padding widths for each axis of the format
    ``(before_1, after_1, ... , before_N, after_N)``. The `pad_width` should be of length ``2*N``
    where ``N`` is the number of dimensions of the array.

    For dimension ``N`` of the input array, ``before_N`` and ``after_N`` indicates how many values
    to add before and after the elements of the array along dimension ``N``.
    The widths of the higher two dimensions ``before_1``, ``after_1``, ``before_2``,
    ``after_2`` must be 0.

    Example::

       x = [[[[  1.   2.   3.]
              [  4.   5.   6.]]

             [[  7.   8.   9.]
              [ 10.  11.  12.]]]


            [[[ 11.  12.  13.]
              [ 14.  15.  16.]]

             [[ 17.  18.  19.]
              [ 20.  21.  22.]]]]

       pad(x,mode="edge", pad_width=(0,0,0,0,1,1,1,1)) =

             [[[[  1.   1.   2.   3.   3.]
                [  1.   1.   2.   3.   3.]
                [  4.   4.   5.   6.   6.]
                [  4.   4.   5.   6.   6.]]

               [[  7.   7.   8.   9.   9.]
                [  7.   7.   8.   9.   9.]
                [ 10.  10.  11.  12.  12.]
                [ 10.  10.  11.  12.  12.]]]


              [[[ 11.  11.  12.  13.  13.]
                [ 11.  11.  12.  13.  13.]
                [ 14.  14.  15.  16.  16.]
                [ 14.  14.  15.  16.  16.]]

               [[ 17.  17.  18.  19.  19.]
                [ 17.  17.  18.  19.  19.]
                [ 20.  20.  21.  22.  22.]
                [ 20.  20.  21.  22.  22.]]]]

       pad(x, mode="constant", constant_value=0, pad_width=(0,0,0,0,1,1,1,1)) =

             [[[[  0.   0.   0.   0.   0.]
                [  0.   1.   2.   3.   0.]
                [  0.   4.   5.   6.   0.]
                [  0.   0.   0.   0.   0.]]

               [[  0.   0.   0.   0.   0.]
                [  0.   7.   8.   9.   0.]
                [  0.  10.  11.  12.   0.]
                [  0.   0.   0.   0.   0.]]]


              [[[  0.   0.   0.   0.   0.]
                [  0.  11.  12.  13.   0.]
                [  0.  14.  15.  16.   0.]
                [  0.   0.   0.   0.   0.]]

               [[  0.   0.   0.   0.   0.]
                [  0.  17.  18.  19.   0.]
                [  0.  20.  21.  22.   0.]
                [  0.   0.   0.   0.   0.]]]]




    Defined in src/operator/pad.cc:L766

    Parameters
    ----------
    data : Symbol
        An n-dimensional input array.
    mode : {'constant', 'edge', 'reflect'}, required
        Padding type to use. "constant" pads with `constant_value` "edge" pads using the edge values of the input array "reflect" pads by reflecting values with respect to the edges.
    pad_width : Shape(tuple), required
        Widths of the padding regions applied to the edges of each axis. It is a tuple of integer padding widths for each axis of the format ``(before_1, after_1, ... , before_N, after_N)``. It should be of length ``2*N`` where ``N`` is the number of dimensions of the array.This is equivalent to pad_width in numpy.pad, but flattened.
    constant_value : double, optional, default=0
        The value used for padding when `mode` is "constant".

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def pick(data=None, index=None, axis=_Null, keepdims=_Null, mode=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Picks elements from an input array according to the input indices along the given axis.

    Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the result will be
    an output array of shape ``(i0,)`` with::

      output[i] = input[i, indices[i]]

    By default, if any index mentioned is too large, it is replaced by the index that addresses
    the last element along an axis (the `clip` mode).

    This function supports n-dimensional input and (n-1)-dimensional indices arrays.

    Examples::

      x = [[ 1.,  2.],
           [ 3.,  4.],
           [ 5.,  6.]]

      // picks elements with specified indices along axis 0
      pick(x, y=[0,1], 0) = [ 1.,  4.]

      // picks elements with specified indices along axis 1
      pick(x, y=[0,1,0], 1) = [ 1.,  4.,  5.]

      y = [[ 1.],
           [ 0.],
           [ 2.]]

      // picks elements with specified indices along axis 1 using 'wrap' mode
      // to place indicies that would normally be out of bounds
      pick(x, y=[2,-1,-2], 1, mode='wrap') = [ 1.,  4.,  5.]

      y = [[ 1.],
           [ 0.],
           [ 2.]]

      // picks elements with specified indices along axis 1 and dims are maintained
      pick(x,y, 1, keepdims=True) = [[ 2.],
                                     [ 3.],
                                     [ 6.]]



    Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L153

    Parameters
    ----------
    data : Symbol
        The input array
    index : Symbol
        The index array
    axis : int or None, optional, default='-1'
        int or None. The axis to picking the elements. Negative values means indexing from right to left. If is `None`, the elements in the index w.r.t the flattened input will be picked.
    keepdims : boolean, optional, default=0
        If true, the axis where we pick the elements is left in the result as dimension with size one.
    mode : {'clip', 'wrap'},optional, default='clip'
        Specify how out-of-bound indices behave. Default is "clip". "clip" means clip to the range. So, if all indices mentioned are too large, they are replaced by the index that addresses the last element along an axis.  "wrap" means to wrap around.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def prod(data=None, axis=_Null, keepdims=_Null, exclude=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Computes the product of array elements over given axes.

    Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L147

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

def radians(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Converts each element of the input array from degrees to radians.

    .. math::
       radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]

    The storage type of ``radians`` output depends upon the input storage type:

       - radians(default) = default
       - radians(row_sparse) = row_sparse
       - radians(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L182

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

def random_exponential(lam=_Null, shape=_Null, ctx=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
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

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def random_gamma(alpha=_Null, beta=_Null, shape=_Null, ctx=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
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

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def random_generalized_negative_binomial(mu=_Null, alpha=_Null, shape=_Null, ctx=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
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

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def random_negative_binomial(k=_Null, p=_Null, shape=_Null, ctx=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
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

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def random_normal(loc=_Null, scale=_Null, shape=_Null, ctx=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
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

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def random_poisson(lam=_Null, shape=_Null, ctx=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
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

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def random_randint(low=_Null, high=_Null, shape=_Null, ctx=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
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

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def random_uniform(low=_Null, high=_Null, shape=_Null, ctx=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
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

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def ravel_multi_index(data=None, shape=_Null, name=None, attr=None, out=None, **kwargs):
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

def rcbrt(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise inverse cube-root value of the input.

    .. math::
       rcbrt(x) = 1/\sqrt[3]{x}

    Example::

       rcbrt([1,8,-125]) = [1.0, 0.5, -0.2]



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L916

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

def reciprocal(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns the reciprocal of the argument, element-wise.

    Calculates 1/x.

    Example::

        reciprocal([-2, 1, 3, 1.6, 0.2]) = [-0.5, 1.0, 0.33333334, 0.625, 5.0]



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L640

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

def relu(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Computes rectified linear.

    .. math::
       max(features, 0)

    The storage type of ``relu`` output depends upon the input storage type:

       - relu(default) = default
       - relu(row_sparse) = row_sparse
       - relu(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L85

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

def repeat(data=None, repeats=_Null, axis=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Repeats elements of an array.

    By default, ``repeat`` flattens the input array into 1-D and then repeats the
    elements::

      x = [[ 1, 2],
           [ 3, 4]]

      repeat(x, repeats=2) = [ 1.,  1.,  2.,  2.,  3.,  3.,  4.,  4.]

    The parameter ``axis`` specifies the axis along which to perform repeat::

      repeat(x, repeats=2, axis=1) = [[ 1.,  1.,  2.,  2.],
                                      [ 3.,  3.,  4.,  4.]]

      repeat(x, repeats=2, axis=0) = [[ 1.,  2.],
                                      [ 1.,  2.],
                                      [ 3.,  4.],
                                      [ 3.,  4.]]

      repeat(x, repeats=2, axis=-1) = [[ 1.,  1.,  2.,  2.],
                                       [ 3.,  3.,  4.,  4.]]



    Defined in src/operator/tensor/matrix_op.cc:L692

    Parameters
    ----------
    data : Symbol
        Input data array
    repeats : int, required
        The number of repetitions for each element.
    axis : int or None, optional, default='None'
        The axis along which to repeat values. The negative numbers are interpreted counting from the backward. By default, use the flattened input array, and return a flat output array.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def reshape(data=None, shape=_Null, reverse=_Null, target_shape=_Null, keep_highest=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Reshapes the input array.

    .. note:: ``Reshape`` is deprecated, use ``reshape``

    Given an array and a shape, this function returns a copy of the array in the new shape.
    The shape is a tuple of integers such as (2,3,4). The size of the new shape should be same as the size of the input array.

    Example::

      reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]

    Some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}. The significance of each is explained below:

    - ``0``  copy this dimension from the input to the output shape.

      Example::

      - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)
      - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)

    - ``-1`` infers the dimension of the output shape by using the remainder of the input dimensions
      keeping the size of the new array same as that of the input array.
      At most one dimension of shape can be -1.

      Example::

      - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)
      - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)
      - input shape = (2,3,4), shape=(-1,), output shape = (24,)

    - ``-2`` copy all/remainder of the input dimensions to the output shape.

      Example::

      - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)
      - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)
      - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)

    - ``-3`` use the product of two consecutive dimensions of the input shape as the output dimension.

      Example::

      - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)
      - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)
      - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)
      - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)

    - ``-4`` split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).

      Example::

      - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)
      - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)

    If the argument `reverse` is set to 1, then the special values are inferred from right to left.

      Example::

      - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape would be (40,5)
      - with reverse=1, output shape will be (50,4).



    Defined in src/operator/tensor/matrix_op.cc:L169

    Parameters
    ----------
    data : Symbol
        Input data to reshape.
    shape : Shape(tuple), optional, default=[]
        The target shape
    reverse : boolean, optional, default=0
        If true then the special values are inferred from right to left
    target_shape : Shape(tuple), optional, default=[]
        (Deprecated! Use ``shape`` instead.) Target new shape. One and only one dim can be 0, in which case it will be inferred from the rest of dims
    keep_highest : boolean, optional, default=0
        (Deprecated! Use ``shape`` instead.) Whether keep the highest dim unchanged.If set to true, then the first dim in target_shape is ignored,and always fixed as input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def reshape_like(lhs=None, rhs=None, name=None, attr=None, out=None, **kwargs):
    r"""Reshape some or all dimensions of `lhs` to have the same shape as some or all dimensions of `rhs`.

    Returns a **view** of the `lhs` array with a new shape without altering any data.

    Example::

      x = [1, 2, 3, 4, 5, 6]
      y = [[0, -4], [3, 2], [2, 2]]
      reshape_like(x, y) = [[1, 2], [3, 4], [5, 6]]

    More precise control over how dimensions are inherited is achieved by specifying \
    slices over the `lhs` and `rhs` array dimensions. Only the sliced `lhs` dimensions \
    are reshaped to the `rhs` sliced dimensions, with the non-sliced `lhs` dimensions staying the same.

      Examples::

      - lhs shape = (30,7), rhs shape = (15,2,4), lhs_begin=0, lhs_end=1, rhs_begin=0, rhs_end=2, output shape = (15,2,7)
      - lhs shape = (3, 5), rhs shape = (1,15,4), lhs_begin=0, lhs_end=2, rhs_begin=1, rhs_end=2, output shape = (15)

    Negative indices are supported, and `None` can be used for either `lhs_end` or `rhs_end` to indicate the end of the range.

      Example::

      - lhs shape = (30, 12), rhs shape = (4, 2, 2, 3), lhs_begin=-1, lhs_end=None, rhs_begin=1, rhs_end=None, output shape = (30, 2, 2, 3)



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L455

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

def reverse(data=None, axis=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Reverses the order of elements along given axis while preserving array shape.

    Note: reverse and flip are equivalent. We use reverse in the following examples.

    Examples::

      x = [[ 0.,  1.,  2.,  3.,  4.],
           [ 5.,  6.,  7.,  8.,  9.]]

      reverse(x, axis=0) = [[ 5.,  6.,  7.,  8.,  9.],
                            [ 0.,  1.,  2.,  3.,  4.]]

      reverse(x, axis=1) = [[ 4.,  3.,  2.,  1.,  0.],
                            [ 9.,  8.,  7.,  6.,  5.]]


    Defined in src/operator/tensor/matrix_op.cc:L794

    Parameters
    ----------
    data : Symbol
        Input data array
    axis : Shape(tuple), required
        The axis which to reverse elements.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def rint(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise rounded value to the nearest integer of the input.

    .. note::
       - For input ``n.5`` ``rint`` returns ``n`` while ``round`` returns ``n+1``.
       - For input ``-n.5`` both ``rint`` and ``round`` returns ``-n-1``.

    Example::

       rint([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  1., -2.,  2.,  2.]

    The storage type of ``rint`` output depends upon the input storage type:

       - rint(default) = default
       - rint(row_sparse) = row_sparse
       - rint(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L721

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

def rmsprop_update(weight=None, grad=None, n=None, lr=_Null, gamma1=_Null, epsilon=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, clip_weights=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Update function for `RMSProp` optimizer.

    `RMSprop` is a variant of stochastic gradient descent where the gradients are
    divided by a cache which grows with the sum of squares of recent gradients?

    `RMSProp` is similar to `AdaGrad`, a popular variant of `SGD` which adaptively
    tunes the learning rate of each parameter. `AdaGrad` lowers the learning rate for
    each parameter monotonically over the course of training.
    While this is analytically motivated for convex optimizations, it may not be ideal
    for non-convex problems. `RMSProp` deals with this heuristically by allowing the
    learning rates to rebound as the denominator decays over time.

    Define the Root Mean Square (RMS) error criterion of the gradient as
    :math:`RMS[g]_t = \sqrt{E[g^2]_t + \epsilon}`, where :math:`g` represents
    gradient and :math:`E[g^2]_t` is the decaying average over past squared gradient.

    The :math:`E[g^2]_t` is given by:

    .. math::
      E[g^2]_t = \gamma * E[g^2]_{t-1} + (1-\gamma) * g_t^2

    The update step is

    .. math::
      \theta_{t+1} = \theta_t - \frac{\eta}{RMS[g]_t} g_t

    The RMSProp code follows the version in
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    Tieleman & Hinton, 2012.

    Hinton suggests the momentum term :math:`\gamma` to be 0.9 and the learning rate
    :math:`\eta` to be 0.001.



    Defined in src/operator/optimizer_op.cc:L553

    Parameters
    ----------
    weight : Symbol
        Weight
    grad : Symbol
        Gradient
    n : Symbol
        n
    lr : float, required
        Learning rate
    gamma1 : float, optional, default=0.95
        The decay rate of momentum estimates.
    epsilon : float, optional, default=1e-08
        A small constant for numerical stability.
    wd : float, optional, default=0
        Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
    rescale_grad : float, optional, default=1
        Rescale gradient to grad = rescale_grad*grad.
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
    clip_weights : float, optional, default=-1
        Clip weights to the range of [-clip_weights, clip_weights] If clip_weights <= 0, weight clipping is turned off. weights = max(min(weights, clip_weights), -clip_weights).

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def rmspropalex_update(weight=None, grad=None, n=None, g=None, delta=None, lr=_Null, gamma1=_Null, gamma2=_Null, epsilon=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, clip_weights=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Update function for RMSPropAlex optimizer.

    `RMSPropAlex` is non-centered version of `RMSProp`.

    Define :math:`E[g^2]_t` is the decaying average over past squared gradient and
    :math:`E[g]_t` is the decaying average over past gradient.

    .. math::
      E[g^2]_t = \gamma_1 * E[g^2]_{t-1} + (1 - \gamma_1) * g_t^2\\
      E[g]_t = \gamma_1 * E[g]_{t-1} + (1 - \gamma_1) * g_t\\
      \Delta_t = \gamma_2 * \Delta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t - E[g]_t^2 + \epsilon}} g_t\\

    The update step is

    .. math::
      \theta_{t+1} = \theta_t + \Delta_t

    The RMSPropAlex code follows the version in
    http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.

    Graves suggests the momentum term :math:`\gamma_1` to be 0.95, :math:`\gamma_2`
    to be 0.9 and the learning rate :math:`\eta` to be 0.0001.


    Defined in src/operator/optimizer_op.cc:L592

    Parameters
    ----------
    weight : Symbol
        Weight
    grad : Symbol
        Gradient
    n : Symbol
        n
    g : Symbol
        g
    delta : Symbol
        delta
    lr : float, required
        Learning rate
    gamma1 : float, optional, default=0.95
        Decay rate.
    gamma2 : float, optional, default=0.9
        Decay rate.
    epsilon : float, optional, default=1e-08
        A small constant for numerical stability.
    wd : float, optional, default=0
        Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
    rescale_grad : float, optional, default=1
        Rescale gradient to grad = rescale_grad*grad.
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
    clip_weights : float, optional, default=-1
        Clip weights to the range of [-clip_weights, clip_weights] If clip_weights <= 0, weight clipping is turned off. weights = max(min(weights, clip_weights), -clip_weights).

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def round(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise rounded value to the nearest integer of the input.

    Example::

       round([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  2., -2.,  2.,  2.]

    The storage type of ``round`` output depends upon the input storage type:

      - round(default) = default
      - round(row_sparse) = row_sparse
      - round(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L700

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

def rsqrt(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise inverse square-root value of the input.

    .. math::
       rsqrt(x) = 1/\sqrt{x}

    Example::

       rsqrt([4,9,16]) = [0.5, 0.33333334, 0.25]

    The storage type of ``rsqrt`` output is always dense



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L860

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

def sample_exponential(lam=None, shape=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
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

def sample_gamma(alpha=None, beta=None, shape=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
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

def sample_generalized_negative_binomial(mu=None, alpha=None, shape=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
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

def sample_multinomial(data=None, shape=_Null, get_prob=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
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

def sample_negative_binomial(k=None, p=None, shape=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
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

def sample_normal(mu=None, sigma=None, shape=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
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

def sample_poisson(lam=None, shape=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
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

def sample_uniform(low=None, high=None, shape=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
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

def scatter_nd(data=None, indices=None, shape=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Scatters data into a new tensor according to indices.

    Given `data` with shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})` and indices with shape
    `(M, Y_0, ..., Y_{K-1})`, the output will have shape `(X_0, X_1, ..., X_{N-1})`,
    where `M <= N`. If `M == N`, data shape should simply be `(Y_0, ..., Y_{K-1})`.

    The elements in output is defined as follows::

      output[indices[0, y_0, ..., y_{K-1}],
             ...,
             indices[M-1, y_0, ..., y_{K-1}],
             x_M, ..., x_{N-1}] = data[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}]

    all other entries in output are 0.

    .. warning::

        If the indices have duplicates, the result will be non-deterministic and
        the gradient of `scatter_nd` will not be correct!!


    Examples::

      data = [2, 3, 0]
      indices = [[1, 1, 0], [0, 1, 0]]
      shape = (2, 2)
      scatter_nd(data, indices, shape) = [[0, 0], [2, 3]]

      data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
      indices = [[0, 1], [1, 1]]
      shape = (2, 2, 2, 2)
      scatter_nd(data, indices, shape) = [[[[0, 0],
                                            [0, 0]],

                                           [[1, 2],
                                            [3, 4]]],

                                          [[[0, 0],
                                            [0, 0]],

                                           [[5, 6],
                                            [7, 8]]]]



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

def sgd_mom_update(weight=None, grad=None, mom=None, lr=_Null, momentum=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, lazy_update=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Momentum update function for Stochastic Gradient Descent (SGD) optimizer.

    Momentum update has better convergence rates on neural networks. Mathematically it looks
    like below:

    .. math::

      v_1 = \alpha * \nabla J(W_0)\\
      v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
      W_t = W_{t-1} + v_t

    It updates the weights using::

      v = momentum * v - learning_rate * gradient
      weight += v

    Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.

    However, if grad's storage type is ``row_sparse``, ``lazy_update`` is True and weight's storage
    type is the same as momentum's storage type,
    only the row slices whose indices appear in grad.indices are updated (for both weight and momentum)::

      for row in gradient.indices:
          v[row] = momentum[row] * v[row] - learning_rate * gradient[row]
          weight[row] += v[row]



    Defined in src/operator/optimizer_op.cc:L372

    Parameters
    ----------
    weight : Symbol
        Weight
    grad : Symbol
        Gradient
    mom : Symbol
        Momentum
    lr : float, required
        Learning rate
    momentum : float, optional, default=0
        The decay rate of momentum estimates at each epoch.
    wd : float, optional, default=0
        Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
    rescale_grad : float, optional, default=1
        Rescale gradient to grad = rescale_grad*grad.
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
    lazy_update : boolean, optional, default=1
        If true, lazy updates are applied if gradient's stype is row_sparse and both weight and momentum have the same stype

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def sgd_update(weight=None, grad=None, lr=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, lazy_update=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Update function for Stochastic Gradient Descent (SDG) optimizer.

    It updates the weights using::

     weight = weight - learning_rate * (gradient + wd * weight)

    However, if gradient is of ``row_sparse`` storage type and ``lazy_update`` is True,
    only the row slices whose indices appear in grad.indices are updated::

     for row in gradient.indices:
         weight[row] = weight[row] - learning_rate * (gradient[row] + wd * weight[row])



    Defined in src/operator/optimizer_op.cc:L331

    Parameters
    ----------
    weight : Symbol
        Weight
    grad : Symbol
        Gradient
    lr : float, required
        Learning rate
    wd : float, optional, default=0
        Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
    rescale_grad : float, optional, default=1
        Rescale gradient to grad = rescale_grad*grad.
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
    lazy_update : boolean, optional, default=1
        If true, lazy updates are applied if gradient's stype is row_sparse.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def shape_array(data=None, lhs_begin=_Null, lhs_end=_Null, rhs_begin=_Null, rhs_end=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Returns a 1D int64 array containing the shape of data.

    Example::

      shape_array([[1,2,3,4], [5,6,7,8]]) = [2,4]



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L506

    Parameters
    ----------
    data : Symbol
        Input Array.
    lhs_begin : int or None, optional, default='None'
        Defaults to 0. The beginning index along which the lhs dimensions are to be reshaped. Supports negative indices.
    lhs_end : int or None, optional, default='None'
        Defaults to None. The ending index along which the lhs dimensions are to be used for reshaping. Supports negative indices.
    rhs_begin : int or None, optional, default='None'
        Defaults to 0. The beginning index along which the rhs dimensions are to be used for reshaping. Supports negative indices.
    rhs_end : int or None, optional, default='None'
        Defaults to None. The ending index along which the rhs dimensions are to be used for reshaping. Supports negative indices.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def shuffle(data=None, name=None, attr=None, out=None, **kwargs):
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

def sigmoid(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Computes sigmoid of x element-wise.

    .. math::
       y = 1 / (1 + exp(-x))

    The storage type of ``sigmoid`` output is always dense



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L101

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

def sign(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise sign of the input.

    Example::

       sign([-2, 0, 3]) = [-1, 0, 1]

    The storage type of ``sign`` output depends upon the input storage type:

       - sign(default) = default
       - sign(row_sparse) = row_sparse
       - sign(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L681

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

def signsgd_update(weight=None, grad=None, lr=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Update function for SignSGD optimizer.

    .. math::

     g_t = \nabla J(W_{t-1})\\
     W_t = W_{t-1} - \eta_t \text{sign}(g_t)

    It updates the weights using::

     weight = weight - learning_rate * sign(gradient)

    .. note:: 
       - sparse ndarray not supported for this optimizer yet.


    Defined in src/operator/optimizer_op.cc:L57

    Parameters
    ----------
    weight : Symbol
        Weight
    grad : Symbol
        Gradient
    lr : float, required
        Learning rate
    wd : float, optional, default=0
        Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
    rescale_grad : float, optional, default=1
        Rescale gradient to grad = rescale_grad*grad.
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def signum_update(weight=None, grad=None, mom=None, lr=_Null, momentum=_Null, wd=_Null, rescale_grad=_Null, clip_gradient=_Null, wd_lh=_Null, name=None, attr=None, out=None, **kwargs):
    r"""SIGN momentUM (Signum) optimizer.

    .. math::

     g_t = \nabla J(W_{t-1})\\
     m_t = \beta m_{t-1} + (1 - \beta) g_t\\
     W_t = W_{t-1} - \eta_t \text{sign}(m_t)

    It updates the weights using::
     state = momentum * state + (1-momentum) * gradient
     weight = weight - learning_rate * sign(state)

    Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.

    .. note:: 
       - sparse ndarray not supported for this optimizer yet.


    Defined in src/operator/optimizer_op.cc:L86

    Parameters
    ----------
    weight : Symbol
        Weight
    grad : Symbol
        Gradient
    mom : Symbol
        Momentum
    lr : float, required
        Learning rate
    momentum : float, optional, default=0
        The decay rate of momentum estimates at each epoch.
    wd : float, optional, default=0
        Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
    rescale_grad : float, optional, default=1
        Rescale gradient to grad = rescale_grad*grad.
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
    wd_lh : float, optional, default=0
        The amount of weight decay that does not go into gradient/momentum calculationsotherwise do weight decay algorithmically only.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def sin(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Computes the element-wise sine of the input array.

    The input should be in radians (:math:`2\pi` rad equals 360 degrees).

    .. math::
       sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]

    The storage type of ``sin`` output depends upon the input storage type:

       - sin(default) = default
       - sin(row_sparse) = row_sparse
       - sin(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L46

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

def sinh(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns the hyperbolic sine of the input array, computed element-wise.

    .. math::
       sinh(x) = 0.5\times(exp(x) - exp(-x))

    The storage type of ``sinh`` output depends upon the input storage type:

       - sinh(default) = default
       - sinh(row_sparse) = row_sparse
       - sinh(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L201

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

def size_array(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns a 1D int64 array containing the size of data.

    Example::

      size_array([[1,2,3,4], [5,6,7,8]]) = [8]



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L558

    Parameters
    ----------
    data : Symbol
        Input Array.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def slice(data=None, begin=_Null, end=_Null, step=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Slices a region of the array.

    .. note:: ``crop`` is deprecated. Use ``slice`` instead.

    This function returns a sliced array between the indices given
    by `begin` and `end` with the corresponding `step`.

    For an input array of ``shape=(d_0, d_1, ..., d_n-1)``,
    slice operation with ``begin=(b_0, b_1...b_m-1)``,
    ``end=(e_0, e_1, ..., e_m-1)``, and ``step=(s_0, s_1, ..., s_m-1)``,
    where m <= n, results in an array with the shape
    ``(|e_0-b_0|/|s_0|, ..., |e_m-1-b_m-1|/|s_m-1|, d_m, ..., d_n-1)``.

    The resulting array's *k*-th dimension contains elements
    from the *k*-th dimension of the input array starting
    from index ``b_k`` (inclusive) with step ``s_k``
    until reaching ``e_k`` (exclusive).

    If the *k*-th elements are `None` in the sequence of `begin`, `end`,
    and `step`, the following rule will be used to set default values.
    If `s_k` is `None`, set `s_k=1`. If `s_k > 0`, set `b_k=0`, `e_k=d_k`;
    else, set `b_k=d_k-1`, `e_k=-1`.

    The storage type of ``slice`` output depends on storage types of inputs

    - slice(csr) = csr
    - otherwise, ``slice`` generates output with default storage

    .. note:: When input data storage type is csr, it only supports
       step=(), or step=(None,), or step=(1,) to generate a csr output.
       For other step parameter values, it falls back to slicing
       a dense tensor.

    Example::

      x = [[  1.,   2.,   3.,   4.],
           [  5.,   6.,   7.,   8.],
           [  9.,  10.,  11.,  12.]]

      slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],
                                         [ 6.,  7.,  8.]]
      slice(x, begin=(None, 0), end=(None, 3), step=(-1, 2)) = [[9., 11.],
                                                                [5.,  7.],
                                                                [1.,  3.]]


    Defined in src/operator/tensor/matrix_op.cc:L414

    Parameters
    ----------
    data : Symbol
        Source input
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

def slice_axis(data=None, axis=_Null, begin=_Null, end=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Slices along a given axis.

    Returns an array slice along a given `axis` starting from the `begin` index
    to the `end` index.

    Examples::

      x = [[  1.,   2.,   3.,   4.],
           [  5.,   6.,   7.,   8.],
           [  9.,  10.,  11.,  12.]]

      slice_axis(x, axis=0, begin=1, end=3) = [[  5.,   6.,   7.,   8.],
                                               [  9.,  10.,  11.,  12.]]

      slice_axis(x, axis=1, begin=0, end=2) = [[  1.,   2.],
                                               [  5.,   6.],
                                               [  9.,  10.]]

      slice_axis(x, axis=1, begin=-3, end=-1) = [[  2.,   3.],
                                                 [  6.,   7.],
                                                 [ 10.,  11.]]


    Defined in src/operator/tensor/matrix_op.cc:L501

    Parameters
    ----------
    data : Symbol
        Source input
    axis : int, required
        Axis along which to be sliced, supports negative indexes.
    begin : int, required
        The beginning index along the axis to be sliced,  supports negative indexes.
    end : int or None, required
        The ending index along the axis to be sliced,  supports negative indexes.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def slice_like(data=None, shape_like=None, axes=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Slices a region of the array like the shape of another array.

    This function is similar to ``slice``, however, the `begin` are always `0`s
    and `end` of specific axes are inferred from the second input `shape_like`.

    Given the second `shape_like` input of ``shape=(d_0, d_1, ..., d_n-1)``,
    a ``slice_like`` operator with default empty `axes`, it performs the
    following operation:

    `` out = slice(input, begin=(0, 0, ..., 0), end=(d_0, d_1, ..., d_n-1))``.

    When `axes` is not empty, it is used to speficy which axes are being sliced.

    Given a 4-d input data, ``slice_like`` operator with ``axes=(0, 2, -1)``
    will perform the following operation:

    `` out = slice(input, begin=(0, 0, 0, 0), end=(d_0, None, d_2, d_3))``.

    Note that it is allowed to have first and second input with different dimensions,
    however, you have to make sure the `axes` are specified and not exceeding the
    dimension limits.

    For example, given `input_1` with ``shape=(2,3,4,5)`` and `input_2` with
    ``shape=(1,2,3)``, it is not allowed to use:

    `` out = slice_like(a, b)`` because ndim of `input_1` is 4, and ndim of `input_2`
    is 3.

    The following is allowed in this situation:

    `` out = slice_like(a, b, axes=(0, 2))``

    Example::

      x = [[  1.,   2.,   3.,   4.],
           [  5.,   6.,   7.,   8.],
           [  9.,  10.,  11.,  12.]]

      y = [[  0.,   0.,   0.],
           [  0.,   0.,   0.]]

      slice_like(x, y) = [[ 1.,  2.,  3.]
                          [ 5.,  6.,  7.]]
      slice_like(x, y, axes=(0, 1)) = [[ 1.,  2.,  3.]
                                       [ 5.,  6.,  7.]]
      slice_like(x, y, axes=(0)) = [[ 1.,  2.,  3.,  4.]
                                    [ 5.,  6.,  7.,  8.]]
      slice_like(x, y, axes=(-1)) = [[  1.,   2.,   3.]
                                     [  5.,   6.,   7.]
                                     [  9.,  10.,  11.]]


    Defined in src/operator/tensor/matrix_op.cc:L570

    Parameters
    ----------
    data : Symbol
        Source input
    shape_like : Symbol
        Shape like input
    axes : Shape(tuple), optional, default=[]
        List of axes on which input data will be sliced according to the corresponding size of the second input. By default will slice on all axes. Negative axes are supported.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def smooth_l1(data=None, scalar=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Calculate Smooth L1 Loss(lhs, scalar) by summing

    .. math::

        f(x) =
        \begin{cases}
        (\sigma x)^2/2,& \text{if }x < 1/\sigma^2\\
        |x|-0.5/\sigma^2,& \text{otherwise}
        \end{cases}

    where :math:`x` is an element of the tensor *lhs* and :math:`\sigma` is the scalar.

    Example::

      smooth_l1([1, 2, 3, 4]) = [0.5, 1.5, 2.5, 3.5]
      smooth_l1([1, 2, 3, 4], scalar=1) = [0.5, 1.5, 2.5, 3.5]



    Defined in src/operator/tensor/elemwise_binary_scalar_op_extended.cc:L104

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

def softmax(data=None, axis=_Null, temperature=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Applies the softmax function.

    The resulting array contains elements in the range (0,1) and the elements along the given axis sum up to 1.

    .. math::
       softmax(\mathbf{z/t})_j = \frac{e^{z_j/t}}{\sum_{k=1}^K e^{z_k/t}}

    for :math:`j = 1, ..., K`

    t is the temperature parameter in softmax function. By default, t equals 1.0

    Example::

      x = [[ 1.  1.  1.]
           [ 1.  1.  1.]]

      softmax(x,axis=0) = [[ 0.5  0.5  0.5]
                           [ 0.5  0.5  0.5]]

      softmax(x,axis=1) = [[ 0.33333334,  0.33333334,  0.33333334],
                           [ 0.33333334,  0.33333334,  0.33333334]]



    Defined in src/operator/nn/softmax.cc:L93

    Parameters
    ----------
    data : Symbol
        The input array.
    axis : int, optional, default='-1'
        The axis along which to compute softmax.
    temperature : double or None, optional, default=None
        Temperature parameter in softmax

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def softmax_cross_entropy(data=None, label=None, name=None, attr=None, out=None, **kwargs):
    r"""Calculate cross entropy of softmax output and one-hot label.

    - This operator computes the cross entropy in two steps:
      - Applies softmax function on the input array.
      - Computes and returns the cross entropy loss between the softmax output and the labels.

    - The softmax function and cross entropy loss is given by:

      - Softmax Function:

      .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}

      - Cross Entropy Function:

      .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i \log(\text{output}_i)

    Example::

      x = [[1, 2, 3],
           [11, 7, 5]]

      label = [2, 0]

      softmax(x) = [[0.09003057, 0.24472848, 0.66524094],
                    [0.97962922, 0.01794253, 0.00242826]]

      softmax_cross_entropy(data, label) = - log(0.66524084) - log(0.97962922) = 0.4281871



    Defined in src/operator/loss_binary_op.cc:L59

    Parameters
    ----------
    data : Symbol
        Input data
    label : Symbol
        Input label

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def softmin(data=None, axis=_Null, temperature=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Applies the softmin function.

    The resulting array contains elements in the range (0,1) and the elements along the given axis sum
    up to 1.

    .. math::
       softmin(\mathbf{z/t})_j = \frac{e^{-z_j/t}}{\sum_{k=1}^K e^{-z_k/t}}

    for :math:`j = 1, ..., K`

    t is the temperature parameter in softmax function. By default, t equals 1.0

    Example::

      x = [[ 1.  2.  3.]
           [ 3.  2.  1.]]

      softmin(x,axis=0) = [[ 0.88079703,  0.5,  0.11920292],
                           [ 0.11920292,  0.5,  0.88079703]]

      softmin(x,axis=1) = [[ 0.66524094,  0.24472848,  0.09003057],
                           [ 0.09003057,  0.24472848,  0.66524094]]



    Defined in src/operator/nn/softmax.cc:L137

    Parameters
    ----------
    data : Symbol
        The input array.
    axis : int, optional, default='-1'
        The axis along which to compute softmax.
    temperature : double or None, optional, default=None
        Temperature parameter in softmax

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def softsign(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Computes softsign of x element-wise.

    .. math::
       y = x / (1 + abs(x))

    The storage type of ``softsign`` output is always dense



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L145

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

def sort(data=None, axis=_Null, is_ascend=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Returns a sorted copy of an input array along the given axis.

    Examples::

      x = [[ 1, 4],
           [ 3, 1]]

      // sorts along the last axis
      sort(x) = [[ 1.,  4.],
                 [ 1.,  3.]]

      // flattens and then sorts
      sort(x) = [ 1.,  1.,  3.,  4.]

      // sorts along the first axis
      sort(x, axis=0) = [[ 1.,  1.],
                         [ 3.,  4.]]

      // in a descend order
      sort(x, is_ascend=0) = [[ 4.,  1.],
                              [ 3.,  1.]]



    Defined in src/operator/tensor/ordering_op.cc:L127

    Parameters
    ----------
    data : Symbol
        The input array
    axis : int or None, optional, default='-1'
        Axis along which to choose sort the input tensor. If not given, the flattened array is used. Default is -1.
    is_ascend : boolean, optional, default=1
        Whether to sort in ascending or descending order.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def space_to_depth(data=None, block_size=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Rearranges(permutes) blocks of spatial data into depth.
    Similar to ONNX SpaceToDepth operator:
    https://github.com/onnx/onnx/blob/master/docs/Operators.md#SpaceToDepth 

    The output is a new tensor where the values from height and width dimension are 
    moved to the depth dimension. The reverse of this operation is ``depth_to_space``.

    .. math::

        \begin{gather*}
        x \prime = reshape(x, [N, C, H / block\_size, block\_size, W / block\_size, block\_size]) \\
        x \prime \prime = transpose(x \prime, [0, 3, 5, 1, 2, 4]) \\
        y = reshape(x \prime \prime, [N, C * (block\_size ^ 2), H / block\_size, W / block\_size])
        \end{gather*}

    where :math:`x` is an input tensor with default layout as :math:`[N, C, H, W]`: [batch, channels, height, width] 
    and :math:`y` is the output tensor of layout :math:`[N, C * (block\_size ^ 2), H / block\_size, W / block\_size]`

    Example::

      x = [[[[0, 6, 1, 7, 2, 8],
             [12, 18, 13, 19, 14, 20],
             [3, 9, 4, 10, 5, 11],
             [15, 21, 16, 22, 17, 23]]]]
  
  
      space_to_depth(x, 2) = [[[[0, 1, 2],
                                [3, 4, 5]],
                               [[6, 7, 8],
                                [9, 10, 11]],
                               [[12, 13, 14],
                                [15, 16, 17]],
                               [[18, 19, 20],
                                [21, 22, 23]]]]


    Defined in src/operator/tensor/matrix_op.cc:L1000

    Parameters
    ----------
    data : Symbol
        Input ndarray
    block_size : int, required
        Blocks of [block_size. block_size] are moved

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def split(data=None, num_outputs=_Null, axis=_Null, squeeze_axis=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Splits an array along a particular axis into multiple sub-arrays.

    .. note:: ``SliceChannel`` is deprecated. Use ``split`` instead.

    **Note** that `num_outputs` should evenly divide the length of the axis
    along which to split the array.

    Example::

       x  = [[[ 1.]
              [ 2.]]
             [[ 3.]
              [ 4.]]
             [[ 5.]
              [ 6.]]]
       x.shape = (3, 2, 1)

       y = split(x, axis=1, num_outputs=2) // a list of 2 arrays with shape (3, 1, 1)
       y = [[[ 1.]]
            [[ 3.]]
            [[ 5.]]]

           [[[ 2.]]
            [[ 4.]]
            [[ 6.]]]

       y[0].shape = (3, 1, 1)

       z = split(x, axis=0, num_outputs=3) // a list of 3 arrays with shape (1, 2, 1)
       z = [[[ 1.]
             [ 2.]]]

           [[[ 3.]
             [ 4.]]]

           [[[ 5.]
             [ 6.]]]

       z[0].shape = (1, 2, 1)

    `squeeze_axis=1` removes the axis with length 1 from the shapes of the output arrays.
    **Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 only
    along the `axis` which it is split.
    Also `squeeze_axis` can be set to true only if ``input.shape[axis] == num_outputs``.

    Example::

       z = split(x, axis=0, num_outputs=3, squeeze_axis=1) // a list of 3 arrays with shape (2, 1)
       z = [[ 1.]
            [ 2.]]

           [[ 3.]
            [ 4.]]

           [[ 5.]
            [ 6.]]
       z[0].shape = (2 ,1 )



    Defined in src/operator/slice_channel.cc:L107

    Parameters
    ----------
    data : Symbol
        The input
    num_outputs : int, required
        Number of splits. Note that this should evenly divide the length of the `axis`.
    axis : int, optional, default='1'
        Axis along which to split.
    squeeze_axis : boolean, optional, default=0
        If true, Removes the axis with length 1 from the shapes of the output arrays. **Note** that setting `squeeze_axis` to ``true`` removes axis with length 1 only along the `axis` which it is split. Also `squeeze_axis` can be set to ``true`` only if ``input.shape[axis] == num_outputs``.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def sqrt(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise square-root value of the input.

    .. math::
       \textrm{sqrt}(x) = \sqrt{x}

    Example::

       sqrt([4, 9, 16]) = [2, 3, 4]

    The storage type of ``sqrt`` output depends upon the input storage type:

       - sqrt(default) = default
       - sqrt(row_sparse) = row_sparse
       - sqrt(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L840

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

def square(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns element-wise squared value of the input.

    .. math::
       square(x) = x^2

    Example::

       square([2, 3, 4]) = [4, 9, 16]

    The storage type of ``square`` output depends upon the input storage type:

       - square(default) = default
       - square(row_sparse) = row_sparse
       - square(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L817

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

def squeeze(*data, **kwargs):
    r"""Remove single-dimensional entries from the shape of an array.
    Same behavior of defining the output tensor shape as numpy.squeeze for the most of cases.
    See the following note for exception.

    Examples::

      data = [[[0], [1], [2]]]
      squeeze(data) = [0, 1, 2]
      squeeze(data, axis=0) = [[0], [1], [2]]
      squeeze(data, axis=2) = [[0, 1, 2]]
      squeeze(data, axis=(0, 2)) = [0, 1, 2]

    .. Note::
      The output of this operator will keep at least one dimension not removed. For example,
      squeeze([[[4]]]) = [4], while in numpy.squeeze, the output will become a scalar.


    Parameters
    ----------
    data : Symbol[]
        data to squeeze
    axis : Shape or None, optional, default=None
        Selects a subset of the single-dimensional entries in the shape. If an axis is selected with shape entry greater than one, an error is raised.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def stack(*data, **kwargs):
    r"""Join a sequence of arrays along a new axis.

    The axis parameter specifies the index of the new axis in the dimensions of the
    result. For example, if axis=0 it will be the first dimension and if axis=-1 it
    will be the last dimension.

    Examples::

      x = [1, 2]
      y = [3, 4]

      stack(x, y) = [[1, 2],
                     [3, 4]]
      stack(x, y, axis=1) = [[1, 3],
                             [2, 4]]

    This function support variable length of positional input.

    Parameters
    ----------
    data : Symbol[]
        List of arrays to stack
    axis : int, optional, default='0'
        The axis in the result array along which the input arrays are stacked.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def stop_gradient(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Stops gradient computation.

    Stops the accumulated gradient of the inputs from flowing through this operator
    in the backward direction. In other words, this operator prevents the contribution
    of its inputs to be taken into account for computing gradients.

    Example::

      v1 = [1, 2]
      v2 = [0, 1]
      a = Variable('a')
      b = Variable('b')
      b_stop_grad = stop_gradient(3 * b)
      loss = MakeLoss(b_stop_grad + a)

      executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))
      executor.forward(is_train=True, a=v1, b=v2)
      executor.outputs
      [ 1.  5.]

      executor.backward()
      executor.grad_arrays
      [ 0.  0.]
      [ 1.  1.]



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L267

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

def sum(data=None, axis=_Null, keepdims=_Null, exclude=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Computes the sum of array elements over given axes.

    .. Note::

      `sum` and `sum_axis` are equivalent.
      For ndarray of csr storage type summation along axis 0 and axis 1 is supported.
      Setting keepdims or exclude to True will cause a fallback to dense operator.

    Example::

      data = [[[1, 2], [2, 3], [1, 3]],
              [[1, 4], [4, 3], [5, 2]],
              [[7, 1], [7, 2], [7, 3]]]

      sum(data, axis=1)
      [[  4.   8.]
       [ 10.   9.]
       [ 21.   6.]]

      sum(data, axis=[1,2])
      [ 12.  19.  27.]

      data = [[1, 2, 0],
              [3, 0, 1],
              [4, 1, 0]]

      csr = cast_storage(data, 'csr')

      sum(csr, axis=0)
      [ 8.  3.  1.]

      sum(csr, axis=1)
      [ 3.  4.  5.]



    Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L116

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

def sum_axis(data=None, axis=_Null, keepdims=_Null, exclude=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Computes the sum of array elements over given axes.

    .. Note::

      `sum` and `sum_axis` are equivalent.
      For ndarray of csr storage type summation along axis 0 and axis 1 is supported.
      Setting keepdims or exclude to True will cause a fallback to dense operator.

    Example::

      data = [[[1, 2], [2, 3], [1, 3]],
              [[1, 4], [4, 3], [5, 2]],
              [[7, 1], [7, 2], [7, 3]]]

      sum(data, axis=1)
      [[  4.   8.]
       [ 10.   9.]
       [ 21.   6.]]

      sum(data, axis=[1,2])
      [ 12.  19.  27.]

      data = [[1, 2, 0],
              [3, 0, 1],
              [4, 1, 0]]

      csr = cast_storage(data, 'csr')

      sum(csr, axis=0)
      [ 8.  3.  1.]

      sum(csr, axis=1)
      [ 3.  4.  5.]



    Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L116

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

def swapaxes(data=None, dim1=_Null, dim2=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Interchanges two axes of an array.

    Examples::

      x = [[1, 2, 3]])
      swapaxes(x, 0, 1) = [[ 1],
                           [ 2],
                           [ 3]]

      x = [[[ 0, 1],
            [ 2, 3]],
           [[ 4, 5],
            [ 6, 7]]]  // (2,2,2) array

     swapaxes(x, 0, 2) = [[[ 0, 4],
                           [ 2, 6]],
                          [[ 1, 5],
                           [ 3, 7]]]


    Defined in src/operator/swapaxis.cc:L70

    Parameters
    ----------
    data : Symbol
        Input array.
    dim1 : int (non-negative), optional, default=0
        the first axis to be swapped.
    dim2 : int (non-negative), optional, default=0
        the second axis to be swapped.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def take(a=None, indices=None, axis=_Null, mode=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Takes elements from an input array along the given axis.

    This function slices the input array along a particular axis with the provided indices.

    Given data tensor of rank r >= 1, and indices tensor of rank q, gather entries of the axis
    dimension of data (by default outer-most one as axis=0) indexed by indices, and concatenates them
    in an output tensor of rank q + (r - 1).

    Examples::

      x = [4.  5.  6.]

      // Trivial case, take the second element along the first axis.

      take(x, [1]) = [ 5. ]

      // The other trivial case, axis=-1, take the third element along the first axis

      take(x, [3], axis=-1, mode='clip') = [ 6. ]

      x = [[ 1.,  2.],
           [ 3.,  4.],
           [ 5.,  6.]]

      // In this case we will get rows 0 and 1, then 1 and 2. Along axis 0

      take(x, [[0,1],[1,2]]) = [[[ 1.,  2.],
                                 [ 3.,  4.]],

                                [[ 3.,  4.],
                                 [ 5.,  6.]]]

      // In this case we will get rows 0 and 1, then 1 and 2 (calculated by wrapping around).
      // Along axis 1

      take(x, [[0, 3], [-1, -2]], axis=1, mode='wrap') = [[[ 1.,  2.],
                                                           [ 3.,  4.]],

                                                          [[ 3.,  4.],
                                                           [ 5.,  6.]]]

    The storage type of ``take`` output depends upon the input storage type:

       - take(default, default) = default
       - take(csr, default, axis=0) = csr



    Defined in src/operator/tensor/indexing_op.cc:L692

    Parameters
    ----------
    a : Symbol
        The input array.
    indices : Symbol
        The indices of the values to be extracted.
    axis : int, optional, default='0'
        The axis of input array to be taken.For input tensor of rank r, it could be in the range of [-r, r-1]
    mode : {'clip', 'raise', 'wrap'},optional, default='clip'
        Specify how out-of-bound indices bahave. Default is "clip". "clip" means clip to the range. So, if all indices mentioned are too large, they are replaced by the index that addresses the last element along an axis.  "wrap" means to wrap around.  "raise" means to raise an error, not supported yet.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def tan(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Computes the element-wise tangent of the input array.

    The input should be in radians (:math:`2\pi` rad equals 360 degrees).

    .. math::
       tan([0, \pi/4, \pi/2]) = [0, 1, -inf]

    The storage type of ``tan`` output depends upon the input storage type:

       - tan(default) = default
       - tan(row_sparse) = row_sparse
       - tan(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L83

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

def tanh(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Returns the hyperbolic tangent of the input array, computed element-wise.

    .. math::
       tanh(x) = sinh(x) / cosh(x)

    The storage type of ``tanh`` output depends upon the input storage type:

       - tanh(default) = default
       - tanh(row_sparse) = row_sparse
       - tanh(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L234

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

def tile(data=None, reps=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Repeats the whole array multiple times.

    If ``reps`` has length *d*, and input array has dimension of *n*. There are
    three cases:

    - **n=d**. Repeat *i*-th dimension of the input by ``reps[i]`` times::

        x = [[1, 2],
             [3, 4]]

        tile(x, reps=(2,3)) = [[ 1.,  2.,  1.,  2.,  1.,  2.],
                               [ 3.,  4.,  3.,  4.,  3.,  4.],
                               [ 1.,  2.,  1.,  2.,  1.,  2.],
                               [ 3.,  4.,  3.,  4.,  3.,  4.]]

    - **n>d**. ``reps`` is promoted to length *n* by pre-pending 1's to it. Thus for
      an input shape ``(2,3)``, ``repos=(2,)`` is treated as ``(1,2)``::


        tile(x, reps=(2,)) = [[ 1.,  2.,  1.,  2.],
                              [ 3.,  4.,  3.,  4.]]

    - **n<d**. The input is promoted to be d-dimensional by prepending new axes. So a
      shape ``(2,2)`` array is promoted to ``(1,2,2)`` for 3-D replication::

        tile(x, reps=(2,2,3)) = [[[ 1.,  2.,  1.,  2.,  1.,  2.],
                                  [ 3.,  4.,  3.,  4.,  3.,  4.],
                                  [ 1.,  2.,  1.,  2.,  1.,  2.],
                                  [ 3.,  4.,  3.,  4.,  3.,  4.]],

                                 [[ 1.,  2.,  1.,  2.,  1.,  2.],
                                  [ 3.,  4.,  3.,  4.,  3.,  4.],
                                  [ 1.,  2.,  1.,  2.,  1.,  2.],
                                  [ 3.,  4.,  3.,  4.,  3.,  4.]]]


    Defined in src/operator/tensor/matrix_op.cc:L753

    Parameters
    ----------
    data : Symbol
        Input data array
    reps : Shape(tuple), required
        The number of times for repeating the tensor a. Each dim size of reps must be a positive integer. If reps has length d, the result will have dimension of max(d, a.ndim); If a.ndim < d, a is promoted to be d-dimensional by prepending new axes. If a.ndim > d, reps is promoted to a.ndim by pre-pending 1's to it.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def topk(data=None, axis=_Null, k=_Null, ret_typ=_Null, is_ascend=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Returns the top *k* elements in an input array along the given axis.
     The returned elements will be sorted.

    Examples::

      x = [[ 0.3,  0.2,  0.4],
           [ 0.1,  0.3,  0.2]]

      // returns an index of the largest element on last axis
      topk(x) = [[ 2.],
                 [ 1.]]

      // returns the value of top-2 largest elements on last axis
      topk(x, ret_typ='value', k=2) = [[ 0.4,  0.3],
                                       [ 0.3,  0.2]]

      // returns the value of top-2 smallest elements on last axis
      topk(x, ret_typ='value', k=2, is_ascend=1) = [[ 0.2 ,  0.3],
                                                   [ 0.1 ,  0.2]]

      // returns the value of top-2 largest elements on axis 0
      topk(x, axis=0, ret_typ='value', k=2) = [[ 0.3,  0.3,  0.4],
                                               [ 0.1,  0.2,  0.2]]

      // flattens and then returns list of both values and indices
      topk(x, ret_typ='both', k=2) = [[[ 0.4,  0.3], [ 0.3,  0.2]] ,  [[ 2.,  0.], [ 1.,  2.]]]



    Defined in src/operator/tensor/ordering_op.cc:L64

    Parameters
    ----------
    data : Symbol
        The input array
    axis : int or None, optional, default='-1'
        Axis along which to choose the top k indices. If not given, the flattened array is used. Default is -1.
    k : int, optional, default='1'
        Number of top elements to select, should be always smaller than or equal to the element number in the given axis. A global sort is performed if set k < 1.
    ret_typ : {'both', 'indices', 'mask', 'value'},optional, default='indices'
        The return type.
     "value" means to return the top k values, "indices" means to return the indices of the top k values, "mask" means to return a mask array containing 0 and 1. 1 means the top k values. "both" means to return a list of both values and indices of top k elements.
    is_ascend : boolean, optional, default=0
        Whether to choose k largest or k smallest elements. Top K largest elements will be chosen if set to false.
    dtype : {'float16', 'float32', 'float64', 'int32', 'uint8'},optional, default='float32'
        DType of the output indices when ret_typ is "indices" or "both". An error will be raised if the selected data type cannot precisely represent the indices.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def transpose(data=None, axes=_Null, name=None, attr=None, out=None, **kwargs):
    r"""Permutes the dimensions of an array.

    Examples::

      x = [[ 1, 2],
           [ 3, 4]]

      transpose(x) = [[ 1.,  3.],
                      [ 2.,  4.]]

      x = [[[ 1.,  2.],
            [ 3.,  4.]],

           [[ 5.,  6.],
            [ 7.,  8.]]]

      transpose(x) = [[[ 1.,  5.],
                       [ 3.,  7.]],

                      [[ 2.,  6.],
                       [ 4.,  8.]]]

      transpose(x, axes=(1,0,2)) = [[[ 1.,  2.],
                                     [ 5.,  6.]],

                                    [[ 3.,  4.],
                                     [ 7.,  8.]]]


    Defined in src/operator/tensor/matrix_op.cc:L312

    Parameters
    ----------
    data : Symbol
        Source input
    axes : Shape(tuple), optional, default=[]
        Target axis order. By default the axes will be inverted.

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def trunc(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Return the element-wise truncated value of the input.

    The truncated value of the scalar x is the nearest integer i which is closer to
    zero than x is. In short, the fractional part of the signed number x is discarded.

    Example::

       trunc([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  1.,  1.,  2.]

    The storage type of ``trunc`` output depends upon the input storage type:

       - trunc(default) = default
       - trunc(row_sparse) = row_sparse
       - trunc(csr) = csr



    Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L779

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

def uniform(low=_Null, high=_Null, shape=_Null, ctx=_Null, dtype=_Null, name=None, attr=None, out=None, **kwargs):
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

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def unravel_index(data=None, shape=_Null, name=None, attr=None, out=None, **kwargs):
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

def where(condition=None, x=None, y=None, name=None, attr=None, out=None, **kwargs):
    r"""Return the elements, either from x or y, depending on the condition.

    Given three ndarrays, condition, x, and y, return an ndarray with the elements from x or y,
    depending on the elements from condition are true or false. x and y must have the same shape.
    If condition has the same shape as x, each element in the output array is from x if the
    corresponding element in the condition is true, and from y if false.

    If condition does not have the same shape as x, it must be a 1D array whose size is
    the same as x's first dimension size. Each row of the output array is from x's row
    if the corresponding element from condition is true, and from y's row if false.

    Note that all non-zero values are interpreted as ``True`` in condition.

    Examples::

      x = [[1, 2], [3, 4]]
      y = [[5, 6], [7, 8]]
      cond = [[0, 1], [-1, 0]]

      where(cond, x, y) = [[5, 2], [3, 8]]

      csr_cond = cast_storage(cond, 'csr')

      where(csr_cond, x, y) = [[5, 2], [3, 8]]



    Defined in src/operator/tensor/control_flow_op.cc:L57

    Parameters
    ----------
    condition : Symbol
        condition array
    x : Symbol
    y : Symbol

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

def zeros_like(data=None, name=None, attr=None, out=None, **kwargs):
    r"""Return an array of zeros with the same shape, type and storage type
    as the input array.

    The storage type of ``zeros_like`` output depends on the storage type of the input

    - zeros_like(row_sparse) = row_sparse
    - zeros_like(csr) = csr
    - zeros_like(default) = default

    Examples::

      x = [[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]]

      zeros_like(x) = [[ 0.,  0.,  0.],
                       [ 0.,  0.,  0.]]



    Parameters
    ----------
    data : Symbol
        The input

    name : string, optional.
        Name of the resulting symbol.

    Returns
    -------
    Symbol
        The result symbol.
    """
    return (0,)

__all__ = ['Activation', 'BatchNorm', 'BatchNorm_v1', 'BilinearSampler', 'BlockGrad', 'CTCLoss', 'Cast', 'Concat', 'Convolution', 'Convolution_v1', 'Correlation', 'Crop', 'Custom', 'Deconvolution', 'Dropout', 'ElementWiseSum', 'Embedding', 'Flatten', 'FullyConnected', 'GridGenerator', 'IdentityAttachKLSparseReg', 'InstanceNorm', 'L2Normalization', 'LRN', 'LayerNorm', 'LeakyReLU', 'LinearRegressionOutput', 'LogisticRegressionOutput', 'MAERegressionOutput', 'MakeLoss', 'Pad', 'Pooling', 'Pooling_v1', 'RNN', 'ROIPooling', 'Reshape', 'SVMOutput', 'SequenceLast', 'SequenceMask', 'SequenceReverse', 'SliceChannel', 'Softmax', 'SoftmaxActivation', 'SoftmaxOutput', 'SpatialTransformer', 'SwapAxis', 'UpSampling', 'abs', 'adam_update', 'add_n', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctanh', 'argmax', 'argmax_channel', 'argmin', 'argsort', 'batch_dot', 'batch_take', 'broadcast_add', 'broadcast_axes', 'broadcast_axis', 'broadcast_div', 'broadcast_equal', 'broadcast_greater', 'broadcast_greater_equal', 'broadcast_hypot', 'broadcast_lesser', 'broadcast_lesser_equal', 'broadcast_like', 'broadcast_logical_and', 'broadcast_logical_or', 'broadcast_logical_xor', 'broadcast_maximum', 'broadcast_minimum', 'broadcast_minus', 'broadcast_mod', 'broadcast_mul', 'broadcast_not_equal', 'broadcast_plus', 'broadcast_power', 'broadcast_sub', 'broadcast_to', 'cast', 'cast_storage', 'cbrt', 'ceil', 'choose_element_0index', 'clip', 'concat', 'cos', 'cosh', 'crop', 'ctc_loss', 'degrees', 'depth_to_space', 'diag', 'dot', 'elemwise_add', 'elemwise_div', 'elemwise_mul', 'elemwise_sub', 'erf', 'exp', 'expand_dims', 'expm1', 'fill_element_0index', 'fix', 'flatten', 'flip', 'floor', 'ftml_update', 'ftrl_update', 'gamma', 'gammaln', 'gather_nd', 'hard_sigmoid', 'identity', 'khatri_rao', 'linalg_gelqf', 'linalg_gemm', 'linalg_gemm2', 'linalg_potrf', 'linalg_potri', 'linalg_sumlogdiag', 'linalg_syrk', 'linalg_trmm', 'linalg_trsm', 'log', 'log10', 'log1p', 'log2', 'log_softmax', 'logical_not', 'make_loss', 'max', 'max_axis', 'mean', 'min', 'min_axis', 'mp_sgd_mom_update', 'mp_sgd_update', 'nanprod', 'nansum', 'negative', 'norm', 'normal', 'one_hot', 'ones_like', 'pad', 'pick', 'prod', 'radians', 'random_exponential', 'random_gamma', 'random_generalized_negative_binomial', 'random_negative_binomial', 'random_normal', 'random_poisson', 'random_randint', 'random_uniform', 'ravel_multi_index', 'rcbrt', 'reciprocal', 'relu', 'repeat', 'reshape', 'reshape_like', 'reverse', 'rint', 'rmsprop_update', 'rmspropalex_update', 'round', 'rsqrt', 'sample_exponential', 'sample_gamma', 'sample_generalized_negative_binomial', 'sample_multinomial', 'sample_negative_binomial', 'sample_normal', 'sample_poisson', 'sample_uniform', 'scatter_nd', 'sgd_mom_update', 'sgd_update', 'shape_array', 'shuffle', 'sigmoid', 'sign', 'signsgd_update', 'signum_update', 'sin', 'sinh', 'size_array', 'slice', 'slice_axis', 'slice_like', 'smooth_l1', 'softmax', 'softmax_cross_entropy', 'softmin', 'softsign', 'sort', 'space_to_depth', 'split', 'sqrt', 'square', 'squeeze', 'stack', 'stop_gradient', 'sum', 'sum_axis', 'swapaxes', 'take', 'tan', 'tanh', 'tile', 'topk', 'transpose', 'trunc', 'uniform', 'unravel_index', 'where', 'zeros_like']