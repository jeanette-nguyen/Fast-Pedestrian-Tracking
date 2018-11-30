import numpy as np
from utils import config as cfg

def center_anchor(anchor):
    """
    Get the values for the center the anchor.

    Parameters
    ----------
    anchor: np array
    [y_min, x_min, y_max, x_max]
    
    Returns
    -------
    The values indicating an anchor's width, height, and center.
    """
    w = anchor[:,3] - anchor[:,1] + 1
    h = anchor[:,2] - anchor[:,0] + 1
    x_center = anchor[:,1] + 0.5*w
    y_center = anchor[:,0] + 0.5*h
    return w, h, x_center, y_center

def set_scale(anchor, scales):
    """
    Generate the anchors for different scales using the same center.
    The height and width are obtained by multiplication of the base anchor
    with the corresponding scale.

    Parameters
    ----------
    anchor: np array
        The base anchor
    scales: np array
        The different scales of the base anchor
    
    Returns
    -------
    The anchors derived from the base vector scaled.
    """
    w, h, x_center, y_center = center_anchor(anchor)
    ws = w*scales
    hs = h*scales
    anchors = create_anchors(ws,hs,x_center,y_center)
    return anchors

def set_ratio(anchor, ratios):
    """
    Generate the anchors for different ratios using the same base anchor.

    Parameters
    ----------
    anchor: np array
        base anchor

    ratios: np array
        different ratios

    Returns
    -------
    The anchors derived from the base anchor using different ratios
    """
    w, h, x_center, y_center = center_anchor(anchor)
    ws = w*np.sqrt(ratios)
    hs = h/np.sqrt(ratios)
    anchors = create_anchors(ws,hs,x_center,y_center)
    return anchors


def create_anchors(ws,hs,x_center,y_center):
    """
    Create the anchors of widths, heights, and centers

    Parameters
    ----------
    w: np array
        the widths
    h: np array
        the heights
    x_center: np array 
        x-axis center
    y_center: np array
        y-axis center

    Returns
    -------
    The anchors centered around (x_center,y_center) with corresponding
    widths and heights.
    """
    #make column vectors
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    x_center = x_center[:, np.newaxis]
    y_center = y_center[:, np.newaxis]
    #[y_min, x_min, y_max, x_max]
    anchors = np.hstack((y_center - 0.5*hs,
                         x_center - 0.5*ws,
                         y_center + 0.5*hs,
                         x_center + 0.5*ws))
    return anchors

def generate_base_anchors():
    """
    Function to generate the anchors, given the scales and ratios.

    Parameters
    ----------
    scales: list
        a list of different area sizes of an anchor
    ratios: list
        a list of the ratios of height to width of the anchor

    Returns
    -------
    The anchors for all different ratios and scales.
    """
    base_size = cfg.anchor_base_size
    ratios = np.array(cfg.anchor_ratios)
    scales = np.array(cfg.anchor_scales)

    #[y_min, x_min, y_max, x_max]
    base_anchor = np.asarray([1,1,base_size,base_size]).reshape(1,-1) - 1
    base_anchors = set_ratio(base_anchor, ratios)
    base_anchors = np.vstack([set_scale(base_anchors[i,:].reshape(1,-1), scales) 
                              for i in range(0,base_anchors.shape[0])])
    return base_anchors

def generate_shifted_anchors(base_anchors,height,width):
    """
    Generate all the possible bounding boxes/ region proposals from
    the A base anchors.

    Parameters
    ----------
    base_anchors: np array 
        The len(scales)*len(ratios) base anchors
        [topmost, leftmost, bottommost, rightmost]
        A base anchors
    height: int
        Height of the feature map
    width: int
        Width of the feature map
    
    height*width*feat_stride = how many shifts needed = K

    Returns
    -------
    All possible A*K = R regional proposals
    """
    feat_stride = cfg.rpn_feat_stride

    # 1 shift in feature map = feat_stride shift in input image
    x_shift = np.arange(0,width*feat_stride,feat_stride)
    y_shift = np.arange(0,height*feat_stride,feat_stride)
    x,y = np.meshgrid(x_shift,y_shift)

    shift = np.stack((y.ravel(),x.ravel(),y.ravel(),x.ravel()),axis=1) 

    A = base_anchors.shape[0]
    K = shift.shape[0]

    # apply every shift to every base anchor
    anchors = base_anchors.reshape((1,A,4)) + shift.reshape((1,K,4)).transpose((1,0,2))
    return anchors.reshape((-1,4)).astype(np.float32)

def loc2bbox(anchors, locs):
    """
    Given the base anchors, determine the region proposal from the base anchors
    in the input image.

    Parameters
    ----------
    anchors: np array
        All base anchors in an image
        Shape is (R, 4), 
        The second index is [y_min, x_min, y_max, x_max]
        Note: R=K*A, where A=len(scales)*len(ratios)
    loc: np array
        The adjustment each anchor should have based off the bounding box
        convolutional layer in the RPN.
        Shape is (len(scales)*len(ratios),4)

    Returns
    -------
    np array - shape (R, 4)
    The regional proposals in an input image.

    """
    # dy = locs[:, 0::4]
    # dx = locs[:, 1::4]
    # dh = locs[:, 2::4]
    # dw = locs[:, 3::4]
    dy = locs[:, 0]
    dx = locs[:, 1]
    dh = locs[:, 2]
    dw = locs[:, 3]

    ws, hs, x_centers, y_centers = center_anchor(anchors)

    bbox_heights = np.exp(dh)*hs 
    bbox_widths = np.exp(dw)*ws
    bbox_center_y = dy*hs + y_centers
    bbox_center_x = dx*ws + x_centers
    bbox = create_anchors(bbox_widths,
                          bbox_heights,
                          bbox_center_x,
                          bbox_center_y)
    #print("loc2bbox" + str(bbox.shape))
    return bbox




