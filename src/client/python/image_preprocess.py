import numpy as np
from PIL import Image
import tensorrtserver.api.model_config_pb2 as model_config


def preprocess(img, format, dtype, c, h, w, specific_scaling=None):
    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:,:,np.newaxis]

    typed = resized.astype(dtype)

    if specific_scaling == 'INCEPTION':
        scaled = (typed / 128) - 1
    elif specific_scaling == 'VGG':
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=dtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=dtype)
    else:
        scaled = typed

    # Swap to CHW if necessary
    if format == model_config.ModelInput.FORMAT_NCHW:
        ordered = np.transpose(scaled, (2, 0, 1))
    else:
        ordered = scaled
    ordered = scaled

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered

def get_byte_data_from_imgs(image_paths, size):
    # size = (height, width)
    imgs = []
    orig_imgs = []
    for image_path in image_paths:
        orig_img = Image.open(image_path)
        img = preprocess(orig_img, model_config.ModelInput.FORMAT_NHWC, np.uint8, 3, size[0], size[1])
        orig_imgs.append(orig_img)
        imgs.append(img)  
    return imgs, orig_imgs
