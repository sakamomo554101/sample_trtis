from tensorrtserver.api import *
import numpy as np
import argparse
import glob
import os


class ContextParameter:
    def __init__(self, model_name):
        self.server_url = "localhost:8000"
        self.http_headers = None
        self.verbose = True
        self.protocol = ProtocolType.from_str("http") # http or grpc
        self.model_name = model_name
        self.model_version = -1
        self.batch_size = 1
        self.check_health = False

def get_input(binary_mode=False):
    text_list = []
    while True:
        text = input("please enter text('q' is exit code) : ").rstrip()
        if text == "q":
            break
        else:
            if binary_mode:
                text = text.encode("utf-8")
            text_list.append(text)
    print(text_list)
    return text_list

def check_health_status(url, model_name, protocol, http_headers, verbose):
    print("Health for model {}".format(model_name))

    # check inference server status
    health_ctx = ServerHealthContext(url, protocol, verbose=verbose)
    is_live = health_ctx.is_live()
    is_ready = health_ctx.is_ready()
    print("Live: {}".format(is_live))
    print("Ready: {}".format(is_ready))
    if not (is_live and is_ready):
        return False

    # check model status
    status_ctx = ServerStatusContext(url, protocol, model_name, verbose=verbose)
    status = status_ctx.get_server_status()
    print("model status is {}".format(status.ready_state))
    if status.ready_state == 2:
        # if ready_state is 2, model status is ready!
        print("model is ready!")
        return True
    else:
        print("model is not ready!")
        return False

def send_request_with_bytes(ctx, corr_id, byte_data, batch_size=1, start_of_sequence=False, end_of_sequence=False):
    flags = InferRequestHeader.FLAG_NONE
    if start_of_sequence:
        flags = flags | InferRequestHeader.FLAG_SEQUENCE_START
    if end_of_sequence:
        flags = flags | InferRequestHeader.FLAG_SEQUENCE_END
    
    result = ctx.run({ 'INPUT' : [np.array([byte_data])]},
                     { 'OUTPUT' : InferContext.ResultFormat.RAW },
                     batch_size=batch_size, flags=flags, corr_id=corr_id)
    return result

def send_request(ctx, corr_id, array, batch_size=1, start_of_sequence=False, end_of_sequence=False):
    flags = InferRequestHeader.FLAG_NONE
    if start_of_sequence:
        flags = flags | InferRequestHeader.FLAG_SEQUENCE_START
    if end_of_sequence:
        flags = flags | InferRequestHeader.FLAG_SEQUENCE_END
    
    result = ctx.run({ 'INPUT' : array},
                     { 'OUTPUT' : InferContext.ResultFormat.RAW },
                     batch_size=batch_size, flags=flags, corr_id=corr_id)
    return result

def get_model_parameter(url, protocol, model_name, verbose=False):
    ctx = ServerStatusContext(url, protocol, model_name, verbose)
    server_status = ctx.get_server_status()
    
    # check model status
    if model_name not in server_status.model_status:
        raise Exception("unable to get status for '" + model_name + "'")
    
    # get config
    status = server_status.model_status[model_name]
    return status.config

def get_image_path_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()
    return args.image_path

def get_image_paths_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder")
    args = parser.parse_args()
    folder_path = args.image_folder
    
    # get image paths
    paths = glob.glob(os.path.join(folder_path, "*.jpg"))
    for path in paths:
        yield path
