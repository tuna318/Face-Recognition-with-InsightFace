# File content is auto-generated. Do not modify.
# pylint: skip-file
from ._internal import NDArrayBase
from ..base import _Null

def AdaptiveAvgPooling2D(data=None, output_size=_Null, out=None, name=None, **kwargs):
    r"""
    Applies a 2D adaptive average pooling over a 4D input with the shape of (NCHW).
    The pooling kernel and stride sizes are automatically chosen for desired output sizes.

    - If a single integer is provided for output_size, the output size is \
      (N x C x output_size x output_size) for any input (NCHW).

    - If a tuple of integers (height, width) are provided for output_size, the output size is \
      (N x C x height x width) for any input (NCHW).



    Defined in src/operator/contrib/adaptive_avg_pooling.cc:L214

    Parameters
    ----------
    data : NDArray
        Input data
    output_size : Shape(tuple), optional, default=[]
        int (output size) or a tuple of int for output (height, width).

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def BilinearResize2D(data=None, height=_Null, width=_Null, out=None, name=None, **kwargs):
    r"""
    Perform 2D resizing (upsampling or downsampling) for 4D input using bilinear interpolation.

    Expected input is a 4 dimensional NDArray (NCHW) and the output
    with the shape of (N x C x height x width). 
    The key idea of bilinear interpolation is to perform linear interpolation
    first in one direction, and then again in the other direction. See the wikipedia of
    `Bilinear interpolation  <https://en.wikipedia.org/wiki/Bilinear_interpolation>`_
    for more details.


    Defined in src/operator/contrib/bilinear_resize.cc:L175

    Parameters
    ----------
    data : NDArray
        Input data
    height : int, required
        output height (required)
    width : int, required
        output width (required)

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def CTCLoss(data=None, label=None, data_lengths=None, label_lengths=None, use_data_lengths=_Null, use_label_lengths=_Null, blank_label=_Null, out=None, name=None, **kwargs):
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
    data : NDArray
        Input ndarray
    label : NDArray
        Ground-truth labels for the loss.
    data_lengths : NDArray
        Lengths of data for each of the samples. Only required when use_data_lengths is true.
    label_lengths : NDArray
        Lengths of labels for each of the samples. Only required when use_label_lengths is true.
    use_data_lengths : boolean, optional, default=0
        Whether the data lenghts are decided by `data_lengths`. If false, the lengths are equal to the max sequence length.
    use_label_lengths : boolean, optional, default=0
        Whether the label lenghts are decided by `label_lengths`, or derived from `padding_mask`. If false, the lengths are derived from the first occurrence of the value of `padding_mask`. The value of `padding_mask` is ``0`` when first CTC label is reserved for blank, and ``-1`` when last label is reserved for blank. See `blank_label`.
    blank_label : {'first', 'last'},optional, default='first'
        Set the label that is reserved for blank label.If "first", 0-th label is reserved, and label values for tokens in the vocabulary are between ``1`` and ``alphabet_size-1``, and the padding mask is ``-1``. If "last", last label value ``alphabet_size-1`` is reserved for blank label instead, and label values for tokens in the vocabulary are between ``0`` and ``alphabet_size-2``, and the padding mask is ``0``.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def DeformableConvolution(data=None, offset=None, weight=None, bias=None, kernel=_Null, stride=_Null, dilate=_Null, pad=_Null, num_filter=_Null, num_group=_Null, num_deformable_group=_Null, workspace=_Null, no_bias=_Null, layout=_Null, out=None, name=None, **kwargs):
    r"""Compute 2-D deformable convolution on 4-D input.

    The deformable convolution operation is described in https://arxiv.org/abs/1703.06211

    For 2-D deformable convolution, the shapes are

    - **data**: *(batch_size, channel, height, width)*
    - **offset**: *(batch_size, num_deformable_group * kernel[0] * kernel[1], height, width)*
    - **weight**: *(num_filter, channel, kernel[0], kernel[1])*
    - **bias**: *(num_filter,)*
    - **out**: *(batch_size, num_filter, out_height, out_width)*.

    Define::

      f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1

    then we have::

      out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
      out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])

    If ``no_bias`` is set to be true, then the ``bias`` term is ignored.

    The default data ``layout`` is *NCHW*, namely *(batch_size, channle, height,
    width)*.

    If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
    evenly into *g* parts along the channel axis, and also evenly split ``weight``
    along the first dimension. Next compute the convolution on the *i*-th part of
    the data with the *i*-th weight part. The output is obtained by concating all
    the *g* results.

    If ``num_deformable_group`` is larger than 1, denoted by *dg*, then split the
    input ``offset`` evenly into *dg* parts along the channel axis, and also evenly
    split ``out`` evenly into *dg* parts along the channel axis. Next compute the
    deformable convolution, apply the *i*-th part of the offset part on the *i*-th
    out.


    Both ``weight`` and ``bias`` are learnable parameters.




    Defined in src/operator/contrib/deformable_convolution.cc:L100

    Parameters
    ----------
    data : NDArray
        Input data to the DeformableConvolutionOp.
    offset : NDArray
        Input offset to the DeformableConvolutionOp.
    weight : NDArray
        Weight matrix.
    bias : NDArray
        Bias parameter.
    kernel : Shape(tuple), required
        Convolution kernel size: (h, w) or (d, h, w)
    stride : Shape(tuple), optional, default=[]
        Convolution stride: (h, w) or (d, h, w). Defaults to 1 for each dimension.
    dilate : Shape(tuple), optional, default=[]
        Convolution dilate: (h, w) or (d, h, w). Defaults to 1 for each dimension.
    pad : Shape(tuple), optional, default=[]
        Zero pad for convolution: (h, w) or (d, h, w). Defaults to no padding.
    num_filter : int (non-negative), required
        Convolution filter(channel) number
    num_group : int (non-negative), optional, default=1
        Number of group partitions.
    num_deformable_group : int (non-negative), optional, default=1
        Number of deformable group partitions.
    workspace : long (non-negative), optional, default=1024
        Maximum temperal workspace allowed for convolution (MB).
    no_bias : boolean, optional, default=0
        Whether to disable bias parameter.
    layout : {None, 'NCDHW', 'NCHW', 'NCW'},optional, default='None'
        Set layout for input, output and weight. Empty for
        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def DeformablePSROIPooling(data=None, rois=None, trans=None, spatial_scale=_Null, output_dim=_Null, group_size=_Null, pooled_size=_Null, part_size=_Null, sample_per_part=_Null, trans_std=_Null, no_trans=_Null, out=None, name=None, **kwargs):
    r"""Performs deformable position-sensitive region-of-interest pooling on inputs.
    The DeformablePSROIPooling operation is described in https://arxiv.org/abs/1703.06211 .batch_size will change to the number of region bounding boxes after DeformablePSROIPooling

    Parameters
    ----------
    data : Symbol
        Input data to the pooling operator, a 4D Feature maps
    rois : Symbol
        Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]]. (x1, y1) and (x2, y2) are top left and down right corners of designated region of interest. batch_index indicates the index of corresponding image in the input data
    trans : Symbol
        transition parameter
    spatial_scale : float, required
        Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal of total stride in convolutional layers
    output_dim : int, required
        fix output dim
    group_size : int, required
        fix group size
    pooled_size : int, required
        fix pooled size
    part_size : int, optional, default='0'
        fix part size
    sample_per_part : int, optional, default='1'
        fix samples per part
    trans_std : float, optional, default=0
        fix transition std
    no_trans : boolean, optional, default=0
        Whether to disable trans parameter.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def MultiBoxDetection(cls_prob=None, loc_pred=None, anchor=None, clip=_Null, threshold=_Null, background_id=_Null, nms_threshold=_Null, force_suppress=_Null, variances=_Null, nms_topk=_Null, out=None, name=None, **kwargs):
    r"""Convert multibox detection predictions.

    Parameters
    ----------
    cls_prob : NDArray
        Class probabilities.
    loc_pred : NDArray
        Location regression predictions.
    anchor : NDArray
        Multibox prior anchor boxes
    clip : boolean, optional, default=1
        Clip out-of-boundary boxes.
    threshold : float, optional, default=0.01
        Threshold to be a positive prediction.
    background_id : int, optional, default='0'
        Background id.
    nms_threshold : float, optional, default=0.5
        Non-maximum suppression threshold.
    force_suppress : boolean, optional, default=0
        Suppress all detections regardless of class_id.
    variances : tuple of <float>, optional, default=[0.1,0.1,0.2,0.2]
        Variances to be decoded from box regression output.
    nms_topk : int, optional, default='-1'
        Keep maximum top k detections before nms, -1 for no limit.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def MultiBoxPrior(data=None, sizes=_Null, ratios=_Null, clip=_Null, steps=_Null, offsets=_Null, out=None, name=None, **kwargs):
    r"""Generate prior(anchor) boxes from data, sizes and ratios.

    Parameters
    ----------
    data : NDArray
        Input data.
    sizes : tuple of <float>, optional, default=[1]
        List of sizes of generated MultiBoxPriores.
    ratios : tuple of <float>, optional, default=[1]
        List of aspect ratios of generated MultiBoxPriores.
    clip : boolean, optional, default=0
        Whether to clip out-of-boundary boxes.
    steps : tuple of <float>, optional, default=[-1,-1]
        Priorbox step across y and x, -1 for auto calculation.
    offsets : tuple of <float>, optional, default=[0.5,0.5]
        Priorbox center offsets, y and x respectively

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def MultiBoxTarget(anchor=None, label=None, cls_pred=None, overlap_threshold=_Null, ignore_label=_Null, negative_mining_ratio=_Null, negative_mining_thresh=_Null, minimum_negative_samples=_Null, variances=_Null, out=None, name=None, **kwargs):
    r"""Compute Multibox training targets

    Parameters
    ----------
    anchor : NDArray
        Generated anchor boxes.
    label : NDArray
        Object detection labels.
    cls_pred : NDArray
        Class predictions.
    overlap_threshold : float, optional, default=0.5
        Anchor-GT overlap threshold to be regarded as a positive match.
    ignore_label : float, optional, default=-1
        Label for ignored anchors.
    negative_mining_ratio : float, optional, default=-1
        Max negative to positive samples ratio, use -1 to disable mining
    negative_mining_thresh : float, optional, default=0.5
        Threshold used for negative mining.
    minimum_negative_samples : int, optional, default='0'
        Minimum number of negative samples.
    variances : tuple of <float>, optional, default=[0.1,0.1,0.2,0.2]
        Variances to be encoded in box regression target.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def MultiProposal(cls_prob=None, bbox_pred=None, im_info=None, rpn_pre_nms_top_n=_Null, rpn_post_nms_top_n=_Null, threshold=_Null, rpn_min_size=_Null, scales=_Null, ratios=_Null, feature_stride=_Null, output_score=_Null, iou_loss=_Null, out=None, name=None, **kwargs):
    r"""Generate region proposals via RPN

    Parameters
    ----------
    cls_prob : NDArray
        Score of how likely proposal is object.
    bbox_pred : NDArray
        BBox Predicted deltas from anchors for proposals
    im_info : NDArray
        Image size and scale.
    rpn_pre_nms_top_n : int, optional, default='6000'
        Number of top scoring boxes to keep after applying NMS to RPN proposals
    rpn_post_nms_top_n : int, optional, default='300'
        Overlap threshold used for non-maximumsuppresion(suppress boxes with IoU >= this threshold
    threshold : float, optional, default=0.7
        NMS value, below which to suppress.
    rpn_min_size : int, optional, default='16'
        Minimum height or width in proposal
    scales : tuple of <float>, optional, default=[4,8,16,32]
        Used to generate anchor windows by enumerating scales
    ratios : tuple of <float>, optional, default=[0.5,1,2]
        Used to generate anchor windows by enumerating ratios
    feature_stride : int, optional, default='16'
        The size of the receptive field each unit in the convolution layer of the rpn,for example the product of all stride's prior to this layer.
    output_score : boolean, optional, default=0
        Add score to outputs
    iou_loss : boolean, optional, default=0
        Usage of IoU Loss

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def PSROIPooling(data=None, rois=None, spatial_scale=_Null, output_dim=_Null, pooled_size=_Null, group_size=_Null, out=None, name=None, **kwargs):
    r"""Performs region-of-interest pooling on inputs. Resize bounding box coordinates by spatial_scale and crop input feature maps accordingly. The cropped feature maps are pooled by max pooling to a fixed size output indicated by pooled_size. batch_size will change to the number of region bounding boxes after PSROIPooling

    Parameters
    ----------
    data : Symbol
        Input data to the pooling operator, a 4D Feature maps
    rois : Symbol
        Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]]. (x1, y1) and (x2, y2) are top left and down right corners of designated region of interest. batch_index indicates the index of corresponding image in the input data
    spatial_scale : float, required
        Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal of total stride in convolutional layers
    output_dim : int, required
        fix output dim
    pooled_size : int, required
        fix pooled size
    group_size : int, optional, default='0'
        fix group size

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def Proposal(cls_prob=None, bbox_pred=None, im_info=None, rpn_pre_nms_top_n=_Null, rpn_post_nms_top_n=_Null, threshold=_Null, rpn_min_size=_Null, scales=_Null, ratios=_Null, feature_stride=_Null, output_score=_Null, iou_loss=_Null, out=None, name=None, **kwargs):
    r"""Generate region proposals via RPN

    Parameters
    ----------
    cls_prob : NDArray
        Score of how likely proposal is object.
    bbox_pred : NDArray
        BBox Predicted deltas from anchors for proposals
    im_info : NDArray
        Image size and scale.
    rpn_pre_nms_top_n : int, optional, default='6000'
        Number of top scoring boxes to keep after applying NMS to RPN proposals
    rpn_post_nms_top_n : int, optional, default='300'
        Overlap threshold used for non-maximumsuppresion(suppress boxes with IoU >= this threshold
    threshold : float, optional, default=0.7
        NMS value, below which to suppress.
    rpn_min_size : int, optional, default='16'
        Minimum height or width in proposal
    scales : tuple of <float>, optional, default=[4,8,16,32]
        Used to generate anchor windows by enumerating scales
    ratios : tuple of <float>, optional, default=[0.5,1,2]
        Used to generate anchor windows by enumerating ratios
    feature_stride : int, optional, default='16'
        The size of the receptive field each unit in the convolution layer of the rpn,for example the product of all stride's prior to this layer.
    output_score : boolean, optional, default=0
        Add score to outputs
    iou_loss : boolean, optional, default=0
        Usage of IoU Loss

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def ROIAlign(data=None, rois=None, pooled_size=_Null, spatial_scale=_Null, sample_ratio=_Null, out=None, name=None, **kwargs):
    r"""
    This operator takes a 4D feature map as an input array and region proposals as `rois`,
    then align the feature map over sub-regions of input and produces a fixed-sized output array.
    This operator is typically used in Faster R-CNN & Mask R-CNN networks.

    Different from ROI pooling, ROI Align removes the harsh quantization, properly aligning
    the extracted features with the input. RoIAlign computes the value of each sampling point
    by bilinear interpolation from the nearby grid points on the feature map. No quantization is
    performed on any coordinates involved in the RoI, its bins, or the sampling points.
    Bilinear interpolation is used to compute the exact values of the
    input features at four regularly sampled locations in each RoI bin.
    Then the feature map can be aggregated by avgpooling.


    References
    ----------

    He, Kaiming, et al. "Mask R-CNN." ICCV, 2017


    Defined in src/operator/contrib/roi_align.cc:L522

    Parameters
    ----------
    data : NDArray
        Input data to the pooling operator, a 4D Feature maps
    rois : NDArray
        Bounding box coordinates, a 2D array
    pooled_size : Shape(tuple), required
        ROI Align output roi feature map height and width: (h, w)
    spatial_scale : float, required
        Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal of total stride in convolutional layers
    sample_ratio : int, optional, default='-1'
        Optional sampling ratio of ROI align, using adaptive size by default.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def SparseEmbedding(data=None, weight=None, input_dim=_Null, output_dim=_Null, dtype=_Null, sparse_grad=_Null, out=None, name=None, **kwargs):
    r"""Maps integer indices to vector representations (embeddings).

    note:: ``contrib.SparseEmbedding`` is deprecated, use ``Embedding`` instead.

    This operator maps words to real-valued vectors in a high-dimensional space,
    called word embeddings. These embeddings can capture semantic and syntactic properties of the words.
    For example, it has been noted that in the learned embedding spaces, similar words tend
    to be close to each other and dissimilar words far apart.

    For an input array of shape (d1, ..., dK),
    the shape of an output array is (d1, ..., dK, output_dim).
    All the input values should be integers in the range [0, input_dim).

    If the input_dim is ip0 and output_dim is op0, then shape of the embedding weight matrix must be
    (ip0, op0).

    The storage type of the gradient will be `row_sparse`.

    .. Note::

        `SparseEmbedding` is designed for the use case where `input_dim` is very large (e.g. 100k).
        The operator is available on both CPU and GPU.
        When `deterministic` is set to `True`, the accumulation of gradients follows a
        deterministic order if a feature appears multiple times in the input. However, the
        accumulation is usually slower when the order is enforced on GPU.
        When the operator is used on the GPU, the recommended value for `deterministic` is `True`.

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
      SparseEmbedding(x, y, 4, 5) = [[[  5.,   6.,   7.,   8.,   9.],
                                     [ 15.,  16.,  17.,  18.,  19.]],

                                    [[  0.,   1.,   2.,   3.,   4.],
                                     [ 10.,  11.,  12.,  13.,  14.]]]



    Defined in src/operator/tensor/indexing_op.cc:L595

    Parameters
    ----------
    data : NDArray
        The input array to the embedding operator.
    weight : NDArray
        The embedding weight matrix.
    input_dim : int, required
        Vocabulary size of the input indices.
    output_dim : int, required
        Dimension of the embedding vectors.
    dtype : {'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'},optional, default='float32'
        Data type of weight.
    sparse_grad : boolean, optional, default=0
        Compute row sparse gradient in the backward calculation. If set to True, the grad's storage type is row_sparse.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def SyncBatchNorm(data=None, gamma=None, beta=None, moving_mean=None, moving_var=None, eps=_Null, momentum=_Null, fix_gamma=_Null, use_global_stats=_Null, output_mean_var=_Null, ndev=_Null, key=_Null, out=None, name=None, **kwargs):
    r"""Batch normalization.

    Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
    well as offset ``beta``.
    Standard BN [1]_ implementation only normalize the data within each device.
    SyncBN normalizes the input within the whole mini-batch.
    We follow the sync-onece implmentation described in the paper [2]_.

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

    Reference:
      .. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating \
        deep network training by reducing internal covariate shift." *ICML 2015*
      .. [2] Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, \
        Ambrish Tyagi, and Amit Agrawal. "Context Encoding for Semantic Segmentation." *CVPR 2018*


    Defined in src/operator/contrib/sync_batch_norm.cc:L97

    Parameters
    ----------
    data : NDArray
        Input data to batch normalization
    gamma : NDArray
        gamma array
    beta : NDArray
        beta array
    moving_mean : NDArray
        running mean of input
    moving_var : NDArray
        running variance of input
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
    ndev : int, optional, default='1'
        The count of GPU devices
    key : string, optional, default=''
        Hash key for synchronization, please set the same hash key for same layer, Block.prefix is typically used as in :class:`gluon.nn.contrib.SyncBatchNorm`.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def backward_index_copy(out=None, name=None, **kwargs):
    r"""

    Parameters
    ----------


    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def backward_quadratic(out=None, name=None, **kwargs):
    r"""

    Parameters
    ----------


    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def bipartite_matching(data=None, is_ascend=_Null, threshold=_Null, topk=_Null, out=None, name=None, **kwargs):
    r"""Compute bipartite matching.
      The matching is performed on score matrix with shape [B, N, M]
      - B: batch_size
      - N: number of rows to match
      - M: number of columns as reference to be matched against.

      Returns:
      x : matched column indices. -1 indicating non-matched elements in rows.
      y : matched row indices.

      Note::

        Zero gradients are back-propagated in this op for now.

      Example::

        s = [[0.5, 0.6], [0.1, 0.2], [0.3, 0.4]]
        x, y = bipartite_matching(x, threshold=1e-12, is_ascend=False)
        x = [1, -1, 0]
        y = [2, 0]



    Defined in src/operator/contrib/bounding_box.cc:L176

    Parameters
    ----------
    data : NDArray
        The input
    is_ascend : boolean, optional, default=0
        Use ascend order for scores instead of descending. Please set threshold accordingly.
    threshold : float, required
        Ignore matching when score < thresh, if is_ascend=false, or ignore score > thresh, if is_ascend=true.
    topk : int, optional, default='-1'
        Limit the number of matches to topk, set -1 for no limit

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def boolean_mask(data=None, index=None, axis=_Null, out=None, name=None, **kwargs):
    r"""
    Experimental CPU-only support for boolean masking.
    Given an n-d NDArray data, and a 1-d NDArray index,
    the operator produces an un-predeterminable shaped n-d NDArray out,
    which stands for the rows in x where the corresonding element in index is non-zero.

    >>> data = mx.nd.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
    >>> index = mx.nd.array([0, 1, 0])
    >>> out = mx.nd.contrib.boolean_mask(data, index)
    >>> out

    [[4. 5. 6.]]
    <NDArray 1x3 @cpu(0)>



    Defined in src/operator/contrib/boolean_mask.cc:L93

    Parameters
    ----------
    data : NDArray
        Data
    index : NDArray
        Mask
    axis : int, optional, default='0'
        An integer that represents the axis in NDArray to mask from.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def box_iou(lhs=None, rhs=None, format=_Null, out=None, name=None, **kwargs):
    r"""Bounding box overlap of two arrays.
      The overlap is defined as Intersection-over-Union, aka, IOU.
      - lhs: (a_1, a_2, ..., a_n, 4) array
      - rhs: (b_1, b_2, ..., b_n, 4) array
      - output: (a_1, a_2, ..., a_n, b_1, b_2, ..., b_n) array

      Note::

        Zero gradients are back-propagated in this op for now.

      Example::

        x = [[0.5, 0.5, 1.0, 1.0], [0.0, 0.0, 0.5, 0.5]]
        y = [[0.25, 0.25, 0.75, 0.75]]
        box_iou(x, y, format='corner') = [[0.1428], [0.1428]]



    Defined in src/operator/contrib/bounding_box.cc:L130

    Parameters
    ----------
    lhs : NDArray
        The first input
    rhs : NDArray
        The second input
    format : {'center', 'corner'},optional, default='corner'
        The box encoding type. 
     "corner" means boxes are encoded as [xmin, ymin, xmax, ymax], "center" means boxes are encodes as [x, y, width, height].

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def box_nms(data=None, overlap_thresh=_Null, valid_thresh=_Null, topk=_Null, coord_start=_Null, score_index=_Null, id_index=_Null, force_suppress=_Null, in_format=_Null, out_format=_Null, out=None, name=None, **kwargs):
    r"""Apply non-maximum suppression to input.

    The output will be sorted in descending order according to `score`. Boxes with
    overlaps larger than `overlap_thresh` and smaller scores will be removed and
    filled with -1, the corresponding position will be recorded for backward propogation.

    During back-propagation, the gradient will be copied to the original
    position according to the input index. For positions that have been suppressed,
    the in_grad will be assigned 0.
    In summary, gradients are sticked to its boxes, will either be moved or discarded
    according to its original index in input.

    Input requirements::

      1. Input tensor have at least 2 dimensions, (n, k), any higher dims will be regarded
      as batch, e.g. (a, b, c, d, n, k) == (a*b*c*d, n, k)
      2. n is the number of boxes in each batch
      3. k is the width of each box item.

    By default, a box is [id, score, xmin, ymin, xmax, ymax, ...],
    additional elements are allowed.

    - `id_index`: optional, use -1 to ignore, useful if `force_suppress=False`, which means
      we will skip highly overlapped boxes if one is `apple` while the other is `car`.

    - `coord_start`: required, default=2, the starting index of the 4 coordinates.
      Two formats are supported:

        - `corner`: [xmin, ymin, xmax, ymax]

        - `center`: [x, y, width, height]

    - `score_index`: required, default=1, box score/confidence.
      When two boxes overlap IOU > `overlap_thresh`, the one with smaller score will be suppressed.

    - `in_format` and `out_format`: default='corner', specify in/out box formats.

    Examples::

      x = [[0, 0.5, 0.1, 0.1, 0.2, 0.2], [1, 0.4, 0.1, 0.1, 0.2, 0.2],
           [0, 0.3, 0.1, 0.1, 0.14, 0.14], [2, 0.6, 0.5, 0.5, 0.7, 0.8]]
      box_nms(x, overlap_thresh=0.1, coord_start=2, score_index=1, id_index=0,
          force_suppress=True, in_format='corner', out_typ='corner') =
          [[2, 0.6, 0.5, 0.5, 0.7, 0.8], [0, 0.5, 0.1, 0.1, 0.2, 0.2],
           [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]]
      out_grad = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                  [0.3, 0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]]
      # exe.backward
      in_grad = [[0.2, 0.2, 0.2, 0.2, 0.2, 0.2], [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]



    Defined in src/operator/contrib/bounding_box.cc:L89

    Parameters
    ----------
    data : NDArray
        The input
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

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def box_non_maximum_suppression(data=None, overlap_thresh=_Null, valid_thresh=_Null, topk=_Null, coord_start=_Null, score_index=_Null, id_index=_Null, force_suppress=_Null, in_format=_Null, out_format=_Null, out=None, name=None, **kwargs):
    r"""Apply non-maximum suppression to input.

    The output will be sorted in descending order according to `score`. Boxes with
    overlaps larger than `overlap_thresh` and smaller scores will be removed and
    filled with -1, the corresponding position will be recorded for backward propogation.

    During back-propagation, the gradient will be copied to the original
    position according to the input index. For positions that have been suppressed,
    the in_grad will be assigned 0.
    In summary, gradients are sticked to its boxes, will either be moved or discarded
    according to its original index in input.

    Input requirements::

      1. Input tensor have at least 2 dimensions, (n, k), any higher dims will be regarded
      as batch, e.g. (a, b, c, d, n, k) == (a*b*c*d, n, k)
      2. n is the number of boxes in each batch
      3. k is the width of each box item.

    By default, a box is [id, score, xmin, ymin, xmax, ymax, ...],
    additional elements are allowed.

    - `id_index`: optional, use -1 to ignore, useful if `force_suppress=False`, which means
      we will skip highly overlapped boxes if one is `apple` while the other is `car`.

    - `coord_start`: required, default=2, the starting index of the 4 coordinates.
      Two formats are supported:

        - `corner`: [xmin, ymin, xmax, ymax]

        - `center`: [x, y, width, height]

    - `score_index`: required, default=1, box score/confidence.
      When two boxes overlap IOU > `overlap_thresh`, the one with smaller score will be suppressed.

    - `in_format` and `out_format`: default='corner', specify in/out box formats.

    Examples::

      x = [[0, 0.5, 0.1, 0.1, 0.2, 0.2], [1, 0.4, 0.1, 0.1, 0.2, 0.2],
           [0, 0.3, 0.1, 0.1, 0.14, 0.14], [2, 0.6, 0.5, 0.5, 0.7, 0.8]]
      box_nms(x, overlap_thresh=0.1, coord_start=2, score_index=1, id_index=0,
          force_suppress=True, in_format='corner', out_typ='corner') =
          [[2, 0.6, 0.5, 0.5, 0.7, 0.8], [0, 0.5, 0.1, 0.1, 0.2, 0.2],
           [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]]
      out_grad = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                  [0.3, 0.3, 0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]]
      # exe.backward
      in_grad = [[0.2, 0.2, 0.2, 0.2, 0.2, 0.2], [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]



    Defined in src/operator/contrib/bounding_box.cc:L89

    Parameters
    ----------
    data : NDArray
        The input
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

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def count_sketch(data=None, h=None, s=None, out_dim=_Null, processing_batch_size=_Null, out=None, name=None, **kwargs):
    r"""Apply CountSketch to input: map a d-dimension data to k-dimension data"

    .. note:: `count_sketch` is only available on GPU.

    Assume input data has shape (N, d), sign hash table s has shape (N, d),
    index hash table h has shape (N, d) and mapping dimension out_dim = k,
    each element in s is either +1 or -1, each element in h is random integer from 0 to k-1.
    Then the operator computs:

    .. math::
       out[h[i]] += data[i] * s[i]

    Example::

       out_dim = 5
       x = [[1.2, 2.5, 3.4],[3.2, 5.7, 6.6]]
       h = [[0, 3, 4]]
       s = [[1, -1, 1]]
       mx.contrib.ndarray.count_sketch(data=x, h=h, s=s, out_dim = 5) = [[1.2, 0, 0, -2.5, 3.4],
                                                                         [3.2, 0, 0, -5.7, 6.6]]



    Defined in src/operator/contrib/count_sketch.cc:L67

    Parameters
    ----------
    data : NDArray
        Input data to the CountSketchOp.
    h : NDArray
        The index vector
    s : NDArray
        The sign vector
    out_dim : int, required
        The output dimension.
    processing_batch_size : int, optional, default='32'
        How many sketch vectors to process at one time.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def ctc_loss(data=None, label=None, data_lengths=None, label_lengths=None, use_data_lengths=_Null, use_label_lengths=_Null, blank_label=_Null, out=None, name=None, **kwargs):
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
    data : NDArray
        Input ndarray
    label : NDArray
        Ground-truth labels for the loss.
    data_lengths : NDArray
        Lengths of data for each of the samples. Only required when use_data_lengths is true.
    label_lengths : NDArray
        Lengths of labels for each of the samples. Only required when use_label_lengths is true.
    use_data_lengths : boolean, optional, default=0
        Whether the data lenghts are decided by `data_lengths`. If false, the lengths are equal to the max sequence length.
    use_label_lengths : boolean, optional, default=0
        Whether the label lenghts are decided by `label_lengths`, or derived from `padding_mask`. If false, the lengths are derived from the first occurrence of the value of `padding_mask`. The value of `padding_mask` is ``0`` when first CTC label is reserved for blank, and ``-1`` when last label is reserved for blank. See `blank_label`.
    blank_label : {'first', 'last'},optional, default='first'
        Set the label that is reserved for blank label.If "first", 0-th label is reserved, and label values for tokens in the vocabulary are between ``1`` and ``alphabet_size-1``, and the padding mask is ``-1``. If "last", last label value ``alphabet_size-1`` is reserved for blank label instead, and label values for tokens in the vocabulary are between ``0`` and ``alphabet_size-2``, and the padding mask is ``0``.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def dequantize(data=None, min_range=None, max_range=None, out_type=_Null, out=None, name=None, **kwargs):
    r"""Dequantize the input tensor into a float tensor.
    min_range and max_range are scalar floats that specify the range for
    the output data.

    When input data type is `uint8`, the output is calculated using the following equation:

    `out[i] = in[i] * (max_range - min_range) / 255.0`,

    When input data type is `int8`, the output is calculate using the following equation
    by keep zero centered for the quantized value:

    `out[i] = in[i] * MaxAbs(min_range, max_range) / 127.0`,

    .. Note::
        This operator only supports forward propogation. DO NOT use it in training.


    Defined in src/operator/quantization/dequantize.cc:L67

    Parameters
    ----------
    data : NDArray
        A ndarray/symbol of type `uint8`
    min_range : NDArray
        The minimum scalar value possibly produced for the input in float32
    max_range : NDArray
        The maximum scalar value possibly produced for the input in float32
    out_type : {'float32'},optional, default='float32'
        Output data type.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def dgl_adjacency(data=None, out=None, name=None, **kwargs):
    r"""This operator converts a CSR matrix whose values are edge Ids
    to an adjacency matrix whose values are ones. The output CSR matrix always has
    the data value of float32.
    Example::

      x = [[ 1, 0, 0 ],
           [ 0, 2, 0 ],
           [ 0, 0, 3 ]]
      dgl_adjacency(x) =
          [[ 1, 0, 0 ],
           [ 0, 1, 0 ],
           [ 0, 0, 1 ]]



    Defined in src/operator/contrib/dgl_graph.cc:L513

    Parameters
    ----------
    data : NDArray
        Input ndarray

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def dgl_subgraph(*data, **kwargs):
    r"""This operator constructs an induced subgraph for
    a given set of vertices from a graph. The operator accepts multiple
    sets of vertices as input. For each set of vertices, it returns a pair
    of CSR matrices if return_mapping is True: the first matrix contains edges
    with new edge Ids, the second matrix contains edges with the original
    edge Ids.
    Example::
      x=[[1, 0, 0, 2],
         [3, 0, 4, 0],
         [0, 5, 0, 0],
         [0, 6, 7, 0]]
      v = [0, 1, 2]
      dgl_subgraph(x, v, return_mapping=True) =
        [[1, 0, 0],
         [2, 0, 3],
         [0, 4, 0]],
        [[1, 0, 0],
         [3, 0, 4],
         [0, 5, 0]]


    Defined in src/operator/contrib/dgl_graph.cc:L267

    Parameters
    ----------
    graph : NDArray
        Input graph where we sample vertices.
    data : NDArray[]
        The input arrays that include data arrays and states.
    return_mapping : boolean, required
        Return mapping of vid and eid between the subgraph and the parent graph.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def div_sqrt_dim(data=None, out=None, name=None, **kwargs):
    r"""Rescale the input by the square root of the channel dimension.

       out = data / sqrt(data.shape[-1])



    Defined in src/operator/contrib/transformer.cc:L38

    Parameters
    ----------
    data : NDArray
        The input array.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def edge_id(data=None, u=None, v=None, out=None, name=None, **kwargs):
    r"""This operator implements the edge_id function for a graph
    stored in a CSR matrix (the value of the CSR stores the edge Id of the graph).
    output[i] = input[u[i], v[i]] if there is an edge between u[i] and v[i]],
    otherwise output[i] will be -1. Both u and v should be 1D vectors.
    Example::
      x = [[ 1, 0, 0 ],
           [ 0, 2, 0 ],
           [ 0, 0, 3 ]]
      u = [ 0, 0, 1, 1, 2, 2 ]
      v = [ 0, 1, 1, 2, 0, 2 ]
      edge_id(x, u, v) = [ 1, -1, 2, -1, -1, 3 ]

    The storage type of ``edge_id`` output depends on storage types of inputs
      - edge_id(csr, default, default) = default
      - default and rsp inputs are not supported



    Defined in src/operator/contrib/dgl_graph.cc:L444

    Parameters
    ----------
    data : NDArray
        Input ndarray
    u : NDArray
        u ndarray
    v : NDArray
        v ndarray

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def fft(data=None, compute_size=_Null, out=None, name=None, **kwargs):
    r"""Apply 1D FFT to input"

    .. note:: `fft` is only available on GPU.

    Currently accept 2 input data shapes: (N, d) or (N1, N2, N3, d), data can only be real numbers.
    The output data has shape: (N, 2*d) or (N1, N2, N3, 2*d). The format is: [real0, imag0, real1, imag1, ...].

    Example::

       data = np.random.normal(0,1,(3,4))
       out = mx.contrib.ndarray.fft(data = mx.nd.array(data,ctx = mx.gpu(0)))



    Defined in src/operator/contrib/fft.cc:L56

    Parameters
    ----------
    data : NDArray
        Input data to the FFTOp.
    compute_size : int, optional, default='128'
        Maximum size of sub-batch to be forwarded at one time

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def getnnz(data=None, axis=_Null, out=None, name=None, **kwargs):
    r"""Number of stored values for a sparse tensor, including explicit zeros.

    This operator only supports CSR matrix on CPU.



    Defined in src/operator/contrib/nnz.cc:L177

    Parameters
    ----------
    data : NDArray
        Input
    axis : int or None, optional, default='None'
        Select between the number of values across the whole matrix, in each column, or in each row.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def group_adagrad_update(weight=None, grad=None, history=None, lr=_Null, rescale_grad=_Null, clip_gradient=_Null, epsilon=_Null, out=None, name=None, **kwargs):
    r"""Update function for Group AdaGrad optimizer.

    Referenced from *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*,
    and available at http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf but
    uses only a single learning rate for every row of the parameter array.

    Updates are applied by::

        grad = clip(grad * rescale_grad, clip_gradient)
        history += mean(square(grad), axis=1, keepdims=True)
        div = grad / sqrt(history + float_stable_eps)
        weight -= div * lr

    Weights are updated lazily if the gradient is sparse.

    Note that non-zero values for the weight decay option are not supported.



    Defined in src/operator/contrib/optimizer_op.cc:L71

    Parameters
    ----------
    weight : NDArray
        Weight
    grad : NDArray
        Gradient
    history : NDArray
        History
    lr : float, required
        Learning rate
    rescale_grad : float, optional, default=1
        Rescale gradient to grad = rescale_grad*grad.
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
    epsilon : float, optional, default=1e-05
        Epsilon for numerical stability

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def ifft(data=None, compute_size=_Null, out=None, name=None, **kwargs):
    r"""Apply 1D ifft to input"

    .. note:: `ifft` is only available on GPU.

    Currently accept 2 input data shapes: (N, d) or (N1, N2, N3, d). Data is in format: [real0, imag0, real1, imag1, ...].
    Last dimension must be an even number.
    The output data has shape: (N, d/2) or (N1, N2, N3, d/2). It is only the real part of the result.

    Example::

       data = np.random.normal(0,1,(3,4))
       out = mx.contrib.ndarray.ifft(data = mx.nd.array(data,ctx = mx.gpu(0)))



    Defined in src/operator/contrib/ifft.cc:L58

    Parameters
    ----------
    data : NDArray
        Input data to the IFFTOp.
    compute_size : int, optional, default='128'
        Maximum size of sub-batch to be forwarded at one time

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def index_copy(old_tensor=None, index_vector=None, new_tensor=None, out=None, name=None, **kwargs):
    r"""Copies the elements of a `new_tensor` into the `old_tensor`.

    This operator copies the elements by selecting the indices in the order given in `index`.
    The output will be a new tensor containing the rest elements of old tensor and
    the copied elements of new tensor.
    For example, if `index[i] == j`, then the `i` th row of `new_tensor` is copied to the
    `j` th row of output.

    The `index` must be a vector and it must have the same size with the `0` th dimension of
    `new_tensor`. Also, the `0` th dimension of old_tensor must `>=` the `0` th dimension of
    `new_tensor`, or an error will be raised.

    Examples::

        x = mx.nd.zeros((5,3))
        t = mx.nd.array([[1,2,3],[4,5,6],[7,8,9]])
        index = mx.nd.array([0,4,2])

        mx.nd.contrib.index_copy(x, index, t)

        [[1. 2. 3.]
         [0. 0. 0.]
         [7. 8. 9.]
         [0. 0. 0.]
         [4. 5. 6.]]
        <NDArray 5x3 @cpu(0)>



    Defined in src/operator/contrib/index_copy.cc:L67

    Parameters
    ----------
    old_tensor : NDArray
        Old tensor
    index_vector : NDArray
        Index vector
    new_tensor : NDArray
        New tensor to be copied

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def quadratic(data=None, a=_Null, b=_Null, c=_Null, out=None, name=None, **kwargs):
    r"""This operators implements the quadratic function.

    .. math::
        f(x) = ax^2+bx+c

    where :math:`x` is an input tensor and all operations
    in the function are element-wise.

    Example::

      x = [[1, 2], [3, 4]]
      y = quadratic(data=x, a=1, b=2, c=3)
      y = [[6, 11], [18, 27]]

    The storage type of ``quadratic`` output depends on storage types of inputs
      - quadratic(csr, a, b, 0) = csr
      - quadratic(default, a, b, c) = default



    Defined in src/operator/contrib/quadratic_op.cc:L50

    Parameters
    ----------
    data : NDArray
        Input ndarray
    a : float, optional, default=0
        Coefficient of the quadratic term in the quadratic function.
    b : float, optional, default=0
        Coefficient of the linear term in the quadratic function.
    c : float, optional, default=0
        Constant term in the quadratic function.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def quantize(data=None, min_range=None, max_range=None, out_type=_Null, out=None, name=None, **kwargs):
    r"""Quantize a input tensor from float to `out_type`,
    with user-specified `min_range` and `max_range`.

    min_range and max_range are scalar floats that specify the range for
    the input data.

    When out_type is `uint8`, the output is calculated using the following equation:

    `out[i] = (in[i] - min_range) * range(OUTPUT_TYPE) / (max_range - min_range) + 0.5`,

    where `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`.

    When out_type is `int8`, the output is calculate using the following equation
    by keep zero centered for the quantized value:

    `out[i] = sign(in[i]) * min(abs(in[i] * scale + 0.5f, quantized_range)`,

    where
    `quantized_range = MinAbs(max(int8), min(int8))` and
    `scale = quantized_range / MaxAbs(min_range, max_range).`

    .. Note::
        This operator only supports forward propogation. DO NOT use it in training.

    Defined in src/operator/quantization/quantize.cc:L74

    Parameters
    ----------
    data : NDArray
        A ndarray/symbol of type `float32`
    min_range : NDArray
        The minimum scalar value possibly produced for the input
    max_range : NDArray
        The maximum scalar value possibly produced for the input
    out_type : {'int8', 'uint8'},optional, default='uint8'
        Output data type.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def quantized_concat(*data, **kwargs):
    r"""Joins input arrays along a given axis.

    The dimensions of the input arrays should be the same except the axis along
    which they will be concatenated.
    The dimension of the output array along the concatenated axis will be equal
    to the sum of the corresponding dimensions of the input arrays.
    All inputs with different min/max will be rescaled by using largest [min, max] pairs.
    If any input holds int8, then the output will be int8. Otherwise output will be uint8.



    Defined in src/operator/quantization/quantized_concat.cc:L108

    Parameters
    ----------
    data : NDArray[]
        List of arrays to concatenate
    dim : int, optional, default='1'
        the dimension to be concated.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def quantized_conv(data=None, weight=None, bias=None, min_data=None, max_data=None, min_weight=None, max_weight=None, min_bias=None, max_bias=None, kernel=_Null, stride=_Null, dilate=_Null, pad=_Null, num_filter=_Null, num_group=_Null, workspace=_Null, no_bias=_Null, cudnn_tune=_Null, cudnn_off=_Null, layout=_Null, out=None, name=None, **kwargs):
    r"""Convolution operator for input, weight and bias data type of int8,
    and accumulates in type int32 for the output. For each argument, two more arguments of type
    float32 must be provided representing the thresholds of quantizing argument from data
    type float32 to int8. The final outputs contain the convolution result in int32, and min
    and max thresholds representing the threholds for quantizing the float32 output into int32.

    .. Note::
        This operator only supports forward propogation. DO NOT use it in training.

    Defined in src/operator/quantization/quantized_conv.cc:L137

    Parameters
    ----------
    data : NDArray
        Input data.
    weight : NDArray
        weight.
    bias : NDArray
        bias.
    min_data : NDArray
        Minimum value of data.
    max_data : NDArray
        Maximum value of data.
    min_weight : NDArray
        Minimum value of weight.
    max_weight : NDArray
        Maximum value of weight.
    min_bias : NDArray
        Minimum value of bias.
    max_bias : NDArray
        Maximum value of bias.
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

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def quantized_flatten(data=None, min_data=None, max_data=None, out=None, name=None, **kwargs):
    r"""

    Parameters
    ----------
    data : NDArray
        A ndarray/symbol of type `float32`
    min_data : NDArray
        The minimum scalar value possibly produced for the data
    max_data : NDArray
        The maximum scalar value possibly produced for the data

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def quantized_fully_connected(data=None, weight=None, bias=None, min_data=None, max_data=None, min_weight=None, max_weight=None, min_bias=None, max_bias=None, num_hidden=_Null, no_bias=_Null, flatten=_Null, out=None, name=None, **kwargs):
    r"""Fully Connected operator for input, weight and bias data type of int8,
    and accumulates in type int32 for the output. For each argument, two more arguments of type
    float32 must be provided representing the thresholds of quantizing argument from data
    type float32 to int8. The final outputs contain the convolution result in int32, and min
    and max thresholds representing the threholds for quantizing the float32 output into int32.

    .. Note::
        This operator only supports forward propogation. DO NOT use it in training.

    Defined in src/operator/quantization/quantized_fully_connected.cc:L90

    Parameters
    ----------
    data : NDArray
        Input data.
    weight : NDArray
        weight.
    bias : NDArray
        bias.
    min_data : NDArray
        Minimum value of data.
    max_data : NDArray
        Maximum value of data.
    min_weight : NDArray
        Minimum value of weight.
    max_weight : NDArray
        Maximum value of weight.
    min_bias : NDArray
        Minimum value of bias.
    max_bias : NDArray
        Maximum value of bias.
    num_hidden : int, required
        Number of hidden nodes of the output.
    no_bias : boolean, optional, default=0
        Whether to disable bias parameter.
    flatten : boolean, optional, default=1
        Whether to collapse all but the first axis of the input data tensor.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def quantized_pooling(data=None, min_data=None, max_data=None, kernel=_Null, pool_type=_Null, global_pool=_Null, cudnn_off=_Null, pooling_convention=_Null, stride=_Null, pad=_Null, p_value=_Null, count_include_pad=_Null, out=None, name=None, **kwargs):
    r"""Pooling operator for input and output data type of int8.
    The input and output data comes with min and max thresholds for quantizing
    the float32 data into int8.

    .. Note::
        This operator only supports forward propogation. DO NOT use it in training.
        This operator only supports `pool_type` of `avg` or `max`.

    Defined in src/operator/quantization/quantized_pooling.cc:L142

    Parameters
    ----------
    data : NDArray
        Input data.
    min_data : NDArray
        Minimum value of data.
    max_data : NDArray
        Maximum value of data.
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

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

def requantize(data=None, min_range=None, max_range=None, min_calib_range=_Null, max_calib_range=_Null, out=None, name=None, **kwargs):
    r"""Given data that is quantized in int32 and the corresponding thresholds,
    requantize the data into int8 using min and max thresholds either calculated at runtime
    or from calibration. It's highly recommended to pre-calucate the min and max thresholds
    through calibration since it is able to save the runtime of the operator and improve the
    inference accuracy.

    .. Note::
        This operator only supports forward propogation. DO NOT use it in training.

    Defined in src/operator/quantization/requantize.cc:L60

    Parameters
    ----------
    data : NDArray
        A ndarray/symbol of type `int32`
    min_range : NDArray
        The original minimum scalar value in the form of float32 used for quantizing data into int32.
    max_range : NDArray
        The original maximum scalar value in the form of float32 used for quantizing data into int32.
    min_calib_range : float or None, optional, default=None
        The minimum scalar value in the form of float32 obtained through calibration. If present, it will be used to requantize the int32 data into int8.
    max_calib_range : float or None, optional, default=None
        The maximum scalar value in the form of float32 obtained through calibration. If present, it will be used to requantize the int32 data into int8.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)

__all__ = ['AdaptiveAvgPooling2D', 'BilinearResize2D', 'CTCLoss', 'DeformableConvolution', 'DeformablePSROIPooling', 'MultiBoxDetection', 'MultiBoxPrior', 'MultiBoxTarget', 'MultiProposal', 'PSROIPooling', 'Proposal', 'ROIAlign', 'SparseEmbedding', 'SyncBatchNorm', 'backward_index_copy', 'backward_quadratic', 'bipartite_matching', 'boolean_mask', 'box_iou', 'box_nms', 'box_non_maximum_suppression', 'count_sketch', 'ctc_loss', 'dequantize', 'dgl_adjacency', 'dgl_subgraph', 'div_sqrt_dim', 'edge_id', 'fft', 'getnnz', 'group_adagrad_update', 'ifft', 'index_copy', 'quadratic', 'quantize', 'quantized_concat', 'quantized_conv', 'quantized_flatten', 'quantized_fully_connected', 'quantized_pooling', 'requantize']