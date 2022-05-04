import numpy as np
from scipy import ndimage
import warnings


def MAE(smap, gtImg):
    """
    Compute MAE.

    Arguments:
        smap  (np.ndarray): Binary/Non binary foreground map with values in the range [0 1].
        gtImg (np.ndarray): Binary ground truth. Type: bool
    Return:
        float: A value MAE, Mean Absolute Error
    """
    mae = np.abs(smap - gtImg).sum() / gtImg.size

    return mae


def PRCurve(smapImg, gtImg):
    """
    Compute precision recall curve.

    Arguments:
        smapImg (np.ndarray): Binary/Non binary foreground map with values in the range [0 1].
        gtImg   (np.ndarray): Binary ground truth. Type: bool
    Return:
        (array, array): precision and recall
    """
    gtPxlNum = gtImg.sum()
    if not gtPxlNum:
        raise ValueError('no foreground region is labeled')

    targetHist, _ = np.histogram(smapImg[gtImg], bins=np.arange(0, 257))
    nontargetHist, _ = np.histogram(smapImg[np.logical_not(gtImg)], bins=np.arange(0, 257))

    targetHist = np.flipud(targetHist)
    nontargetHist = np.flipud(nontargetHist)

    targetHist = np.cumsum(targetHist)
    nontargetHist = np.cumsum(nontargetHist)

    precision = targetHist / (targetHist + nontargetHist + np.finfo(np.float64).eps)
    if np.isnan(precision).any():
        warnings.warn('there exists NAN in precision, this is because your saliency map do not range from 0 to 255', RuntimeWarning)
    recall = targetHist / gtPxlNum

    return precision, recall


def Fmeasure(sMap, gtMap, beta=0.3):
    """
    Compute the F measure

    Arguments:
        sMap: Binary/Non binary foreground map with values in the range [0 1].
        gtMap: Binary ground truth. Type: bool
        beta: the parameter beta.
    Return:
        (float, float, float): Precision, Recall and F-Measure.
    """
    sumLabel = np.minimum(2 * sMap.mean(), 1)

    Label3 = np.zeros_like(gtMap)
    Label3[sMap >= sumLabel] = 1

    NumRec = Label3.sum()
    NumAnd = np.logical_and(Label3, gtMap).sum()
    num_obj = gtMap.sum()

    if NumAnd == 0:
        PreFtem = 0
        RecallFtem = 0
        FmeasureF = 0
    else:
        PreFtem = NumAnd / NumRec
        RecallFtem = NumAnd / num_obj
        FmeasureF = (1 + beta) * PreFtem * RecallFtem / (beta * PreFtem + RecallFtem)

    return PreFtem, RecallFtem, FmeasureF


def Smeasure(prediction, GT):
    """
    Smeasure computes the similarity between the foreground map and
    ground truth(as proposed in "Structure-measure: A new way to evaluate
    foreground maps" [Deng-Ping Fan et. al - ICCV 2017])

    Arguments:
        prediction (np.ndarray): Binary/Non binary foreground map with values in the range [0 1].
        GT         (np.ndarray): Binary ground truth. Type: bool
    Return:
        float: The computed similarity score
    """
    y = GT.mean()
    if y == 0:
        x = prediction.mean()
        Q = 1.0 - x
    elif y == 1:
        x = prediction.mean()
        Q = x
    else:
        alpha = 0.5
        Q = np.maximum(0, alpha * _S_object(prediction, GT) + (1 - alpha) * _S_region(prediction, GT))
    return Q


def _S_region(prediction, GT):
    """
    S_region computes the region similarity between the foreground map and
    ground truth(as proposed in "Structure-measure:A new way to evaluate
    foreground maps" [Deng-Ping Fan et. al - ICCV 2017])

    Arguments:
        prediction (np.ndarray): Binary/Non binary foreground map with values in the range [0 1].
        GT         (np.ndarray): Binary ground truth. Type: bool
    Return:
        float: The region similarity score
    """

    # find the centroid of the GT
    X, Y = _centroid(GT)

    # divide GT into 4 regions
    GT_1, GT_2, GT_3, GT_4, w1, w2, w3, w4 = _divide_GT(GT, X, Y)

    # Divide prediction into 4 regions
    prediction_1, prediction_2, prediction_3, prediction_4 = _divide_prediction(prediction, X, Y)

    # Compute the ssim score for each regions
    Q1 = _ssim(prediction_1, GT_1)
    Q2 = _ssim(prediction_2, GT_2)
    Q3 = _ssim(prediction_3, GT_3)
    Q4 = _ssim(prediction_4, GT_4)

    # Sum the 4 scores
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    return Q


def _centroid(GT):
    """
    Centroid Compute the centroid of the ground truth

    Arguments:
        GT (np.ndarray): Binary ground truth. Type: bool
    Return:
        (int, int): The coordinates of centroid.
    """
    rows, cols = GT.shape
    if GT.sum() == 0:
        X = np.round(cols / 2)
        Y = np.round(rows / 2)
    else:
        total = GT.sum()
        i = np.arange(1, cols + 1)
        j = np.arange(1, rows + 1)
        X = int(np.round(GT.sum(axis=0).dot(i) / total))
        Y = int(np.round(GT.sum(axis=1).dot(j) / total))

    return X, Y


def _divide_GT(GT, X, Y):
    """
    divide the GT into 4 regions according to the centroid of the GT and return the weights
    LT - left top;
    RT - right top;
    LB - left bottom;
    RB - right bottom;
    """
    # width and height of the GT
    height, width = GT.shape
    area = width * height

    # copy the 4 regions
    LT = GT[0:Y, 0:X]
    RT = GT[0:Y, X:width]
    LB = GT[Y:height, 0:X]
    RB = GT[Y:height, X:width]

    # The different weight (each block proportional to the GT foreground region).
    w1 = X * Y / area
    w2 = (width - X) * Y / area
    w3 = X * (height - Y) / area
    w4 = 1.0 - w1 - w2 - w3

    return LT, RT, LB, RB, w1, w2, w3, w4


def _divide_prediction(prediction, X, Y):
    """
    Divide the prediction into 4 regions according to the centroid of the GT
    """
    # width and height of the prediction
    height, width = prediction.shape

    # copy the 4 regions
    LT = prediction[0:Y, 0:X]
    RT = prediction[0:Y, X:width]
    LB = prediction[Y:height, 0:X]
    RB = prediction[Y:height, X:width]

    return LT, RT, LB, RB


def _ssim(prediction, GT):
    """
    ssim computes the region similarity between foreground maps and ground
    truth(as proposed in "Structure-measure: A new way to evaluate foreground
    maps" [Deng-Ping Fan et. al - ICCV 2017])

    Arguments:
        prediction (np.ndarray): Binary/Non binary foreground map with values in the range [0 1].
        GT         (np.ndarray): Binary ground truth. Type: bool
    Return:
        float: The region similarity score
    """
    height, width = prediction.shape

    # Compute the mean of SM,GT
    x = prediction.mean()
    y = GT.mean()

    # Compute the variance of SM,GT
    sigma_x2 = prediction.var(ddof=1)
    sigma_y2 = GT.var(ddof=1)

    # Compute the covariance between SM and GT
    sigma_xy = np.cov(prediction.reshape(-1), GT.reshape(-1), ddof=1)[0][1]

    alpha = 4 * x * y * sigma_xy
    beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        Q = alpha / (beta + np.finfo(np.float64).eps)
    elif beta == 0:
        Q = 1.0
    else:
        Q = 0

    return Q


def _S_object(prediction, GT):
    """
    S_object Computes the object similarity between foreground maps and ground
    truth(as proposed in "Structure-measure:A new way to evaluate foreground
    maps" [Deng-Ping Fan et. al - ICCV 2017])

    Arguments:
        prediction (np.ndarray): Binary/Non binary foreground map with values in the range [0 1].
        GT         (np.ndarray): Binary ground truth. Type: bool
    Return:
        float: The object similarity score
    """
    # compute the similarity of the foreground in the object level
    prediction_fg = prediction.copy()
    prediction_fg[np.logical_not(GT)] = 0
    o_fg = _object(prediction_fg, GT)

    # compute the similarity of the background
    prediction_bg = 1.0 - prediction
    prediction_bg[GT] = 0
    o_bg = _object(prediction_bg, np.logical_not(GT))

    # combine the foreground measure and background measure together
    u = GT.mean()
    Q = u * o_fg + (1 - u) * o_bg

    return Q


def _object(prediction, GT):
    # compute the mean of the foreground or background
    x = prediction[GT].mean()

    # compute the standard deviations of the foreground or background in prediction
    sigma_x = prediction[GT].std(ddof=1)

    score = 2.0 * x / (x ** 2 + 1.0 + sigma_x + np.finfo(np.float64).eps)
    return score


def Emeasure(FM, GT):
    """
    Emeasure Compute the Enhanced Alignment measure (as proposed in "Enhanced-alignment
    Measure for Binary Foreground Map Evaluation" [Deng-Ping Fan et. al - IJCAI'18 oral paper])

    Arguments:
        FM (np.ndarray): Binary foreground map.
        GT (np.ndarray): Binary ground truth. Type: bool
    Return:
        float: The Enhanced alignment score
    """
    thd = 2 * FM.mean()
    FM = FM > thd
    # thd = np.minimum(thd, 1)

    # Special case:
    if GT.sum() == 0:  # if the GT is completely black
        enhanced_matrix = 1.0 - FM  # only calculate the black area of intersection
    elif np.logical_not(GT).sum() == 0:  # if the GT is completely white
        enhanced_matrix = FM  # only calcualte the white area of intersection
    else:
        # Normal case:

        # 1.compute alignment matrix
        align_matrix = _alignment_term(FM, GT)
        # 2.compute enhanced alignment matrix
        enhanced_matrix = _enhanced_alignment_term(align_matrix)

    # 3.Emeasure score
    score = enhanced_matrix.sum() / (GT.size - 1 + np.finfo(np.float64).eps)
    return score


def _alignment_term(FM, GT):
    """
    Alignment Term
    """
    # compute global mean
    mu_FM = FM.mean()
    mu_GT = GT.mean()

    # compute the bias matrix
    align_FM = FM - mu_FM
    align_GT = GT - mu_GT

    # compute alignment matrix
    align_matrix = 2 * align_GT * align_FM / (align_GT * align_GT + align_FM * align_FM + np.finfo(np.float64).eps)
    return align_matrix


def _enhanced_alignment_term(align_matrix):
    """
    Enhanced Alignment Term function. f(x) = 1/4*(1 + x)^2)
    """
    enhanced = ((align_matrix) + 1) ** 2 / 4
    return enhanced


def wFmeasure(FG, GT):
    """
    wFmeasure Compute the Weighted F-beta measure (as proposed in "How to Evaluate
    Foreground Maps?" [Margolin et. al - CVPR'14])

    Arguments:
        FG (np.ndarray): Binary/Non binary foreground map with values in the range [0 1].
        GT (np.ndarray): Binary ground truth. Type: bool
    Return:
        float : The Weighted F-beta score
    """
    if not GT.max():
        return 0

    E = np.abs(FG - GT)

    Dst, IDXT = ndimage.distance_transform_edt(1 - GT.astype(np.float64), return_indices=True)
    # Pixel dependency
    Et = E.copy()
    Et[np.logical_not(GT)] = Et[IDXT[0][np.logical_not(GT)], IDXT[1][np.logical_not(GT)]]  # To deal correctly with the edges of the foreground region
    EA = ndimage.gaussian_filter(Et, 5, mode='constant', truncate=0.5)
    MIN_E_EA = E.copy()
    MIN_E_EA[np.logical_and(GT, EA < E)] = EA[np.logical_and(GT, EA < E)]
    # Pixel importance
    B = np.ones(GT.shape)
    B[np.logical_not(GT)] = 2 - 1 * np.exp(np.log(1 - 0.5) / 5 * Dst[np.logical_not(GT)])
    Ew = MIN_E_EA * B

    TPw = GT.sum() - Ew[GT].sum()
    FPw = Ew[np.logical_not(GT)].sum()

    R = 1 - Ew[GT].mean()  # Weighted Recall
    P = TPw / (TPw + FPw + np.finfo(np.float64).eps)  # Weighted Precision

    Q = 2 * R * P / (R + P + np.finfo(np.float64).eps)  # Beta=1
    # Q = (1 + Beta ** 2) * R * P / (R + Beta * P + np.finfo(np.float64).eps)

    return Q
