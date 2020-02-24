from tensorrtserver.api import *
import numpy as np
from util import *
import argparse
from PIL import Image
import tensorrtserver.api.model_config_pb2 as model_config
import json


def main():
    # setup environment
    ctx_param = ContextParameter(model_name="face_recognition_model")
    ctx_param.server_url = "trtis-server-build-container:8000"
    ctx_param.http_headers = None
    ctx_param.verbose = True
    ctx_param.protocol = ProtocolType.from_str("http") # http or grpc
    ctx_param.model_version = -1
    batch_size = 1
    check_health = False
    
    # get parameter from model config
    model_param = get_model_parameter(ctx_param.server_url, ctx_param.protocol, ctx_param.model_name, ctx_param.verbose)
    input_width = model_param.input[0].dims[1]
    input_height = model_param.input[0].dims[0]

    # send request
    image_path = get_image_path_from_args()
    img = get_byte_data_from_img(image_path, size=(input_height, input_width))
    corr_id = 1000
    datas_dict = {corr_id: [img]}
    result_map = infer(ctx_param, img.shape, img.dtype, datas_dict)
    
    print(result_map)
    
    
def get_image_path_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()
    return args.image_path

def get_byte_data_from_img(image_path, size):
    # size = (height, width)
    img = Image.open(image_path)
    img = preprocess(img, model_config.ModelInput.FORMAT_NHWC, np.uint8, 3, size[0], size[1])
    return img

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

def infer(ctx_param, shape, dtype, datas_dict, batch_size=1):
    """
    infer function
    Arguments:
        ctx_param {ContextParameter} -- ContextParameter
        shape -- data shape (ex. (3, 480, 640))
        datas_dict {dict} -- key is correlation id, value is data list
        batch_size {int} -- batch size
    """
    infer_ctx = InferContext(ctx_param.server_url, 
                             ctx_param.protocol, 
                             ctx_param.model_name, 
                             ctx_param.model_version,
                             http_headers=ctx_param.http_headers, 
                             verbose=ctx_param.verbose)
    
    result_map = {}
    for corr_id, img_list in datas_dict.items():
        # send start request
        send_request(infer_ctx, corr_id, np.zeros(shape, dtype=dtype), batch_size=batch_size, start_of_sequence=True)
        
        # send the image data
        for img in img_list:
            print(img.shape)
            send_request(infer_ctx, corr_id, img, batch_size=batch_size) 
        
        # send end request
        result = send_request(infer_ctx, corr_id, np.zeros(shape, dtype=dtype), batch_size=batch_size, end_of_sequence=True)
        
        # postprocess result
        result_map[corr_id] = json.loads(result["OUTPUT"][0][0])

    return result_map

if __name__ == "__main__":
    print("start script")
    main()
    print("end script")
