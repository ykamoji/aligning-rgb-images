import numpy as np
from skimage.transform import rescale
from scipy.ndimage import sobel

def alignChannels(img, max_shift, image_ext, cropFix=False):

    blue = img[:, :, 2]
    green = img[:, :, 1]
    red = img[:, :, 0]

    methods = ["cs","ssd"]
    method = methods[0]

    ratio_crop = 0.1

    if image_ext in ['jpg','jpeg','png']:

        shift_green, xy_g = align(red, green, ratio_crop, max_shift, method=method)
        shift_blue, xy_b = align(red, blue, ratio_crop, max_shift, method=method)

        predShift = np.array([xy_g, xy_b]).reshape(2, 2)

        image_channel = np.dstack((red, shift_green, shift_blue))

    elif image_ext in ['tif']:

        blue_s = np.abs(sobel(blue))
        green_s = np.abs(sobel(green))
        red_s = np.abs(sobel(red))

        _, shift_green = pyramid(red_s, green_s, method, ratio_crop, max_shift)
        _, shift_blue = pyramid(red_s, blue_s, method, ratio_crop, max_shift)

        predShift = np.array([shift_green, shift_blue]).reshape(2, 2)

        green = np.roll(green, shift_green, (0, 1))
        blue = np.roll(blue, shift_blue, (0, 1))

        image_channel = np.dstack((red, green, blue))
        image_channel /= 255

        image_channel[image_channel < 0] = 0
        image_channel[image_channel > 1] = 1

    if cropFix:
        image_channel = trim(image_channel, predShift)
        image_channel = crop(image_channel)

    return image_channel, predShift


def align(channel1, channel2, ratio_crop, max_shifts, method):

    hrange = range(-max_shifts[0], max_shifts[0]+1)
    vrange = range(-max_shifts[1], max_shifts[1]+1)

    mini = 0
    minj = 0

    min_score = evalScore(channel1, channel2, ratio_crop, method=method)

    for i in hrange:
        for j in vrange:
            temp = evalScore(channel1, np.roll(channel2, [i, j], (0, 1)), ratio_crop, method=method)
            if temp < min_score:
                min_score = temp
                mini = i
                minj = j

    shifted_img = np.roll(channel2, [mini, minj], (0, 1))
    return shifted_img, np.array([mini, minj])

def evalScore(channel1, channel2, exclude, method):

    channel1_revised, channel2_revised = crop_channel(exclude, channel1, channel2)

    if method =='cs':
        channel1_flat = np.ndarray.flatten(channel1_revised)
        channel2_flat = np.ndarray.flatten(channel2_revised)
        return -np.dot(channel1_flat / np.linalg.norm(channel1_flat), channel2_flat / np.linalg.norm(channel2_flat))

    elif method =='ssd':
        return np.sum((channel1_revised - channel2_revised) ** 2)

def crop_channel(exclude, channel1, channel2):

    channel1_crop = int(exclude * len(channel1))
    channel2_crop = int(exclude * len(channel2))

    channel1_revised = channel1[channel1_crop:-channel1_crop, channel1_crop:-channel1_crop]
    channel2_revised = channel2[channel2_crop:-channel2_crop, channel2_crop:-channel2_crop]

    return channel1_revised, channel2_revised

# Trimming the edges on computed shift channels (Extra Credit)
def trim(image_channel, predShift):
    shift_channel_1_i = predShift[0, 0]
    shift_channel_1_j = predShift[0, 1]
    shift_channel_2_i = predShift[1, 0]
    shift_channel_2_j = predShift[1, 1]

    ulI = max(1, shift_channel_1_i, shift_channel_2_i)
    ulJ = max(1, shift_channel_1_j, shift_channel_2_j)

    brI, brJ = image_channel.shape[:2]
    brI = min(brI, brI + shift_channel_1_i, brI + shift_channel_2_i)
    brJ = min(brJ, brJ + shift_channel_1_j, brJ + shift_channel_2_j)

    trimmed = np.copy(image_channel)
    color = 255
    for c in range(3):
        trimmed[:ulI, :, c] = color
        trimmed[brI:, :, c] = color
        trimmed[:, :ulJ, c] = color
        trimmed[:, brJ:, c] = color
    return trimmed

# Cropping with 0.05 percent at the edges (Extra Credit)
def crop(image_channel):
    brI, brJ = image_channel.shape[:2]

    ulI = int(0.05 * brI)
    ulJ = int(0.05 * brJ)

    brI = int(0.95 * brI)
    brJ = int(0.95 * brJ)
    cropped = image_channel.copy()
    color = 1
    for c in range(3):
        cropped[0:ulI, :, c] = color
        cropped[brI:, :, c] = color
        cropped[:, 0:ulJ, c] = color
        cropped[:, brJ:, c] = color
    return cropped

## High resolution processing (Extra Credit)
def pyramid(channel1, channel2, method, ratio_crop, max_shifts, depth=5):

    if channel1.shape[0] < 400 or depth == 0:
        return align(channel1, channel2, ratio_crop, max_shifts, method)
    else:
        _, pred_shift = pyramid(rescale(channel1, 0.5), rescale(channel2, 0.5), method, ratio_crop, max_shifts, depth=depth - 1)
        result, new_shift = align(channel1, channel2, ratio_crop, [-5, 5], method)
        pred_shift = pred_shift*2 + new_shift
        return result, pred_shift


