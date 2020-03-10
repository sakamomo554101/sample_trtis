from tensorrtserver.api import *
import numpy as np
from util import *


def main():
    # setup environment
    ctx_param = ContextParameter(
        url="trtis-server-container",
        protocol="http",
        model_name="sample_sequence"
    )
    ctx_param.http_headers = None
    ctx_param.verbose = True
    ctx_param.model_version = -1
    batch_size = 1

    # print server health and model status
    check_health_status(url=ctx_param.server_url, 
                        model_name=ctx_param.model_name, 
                        protocol=ctx_param.protocol, 
                        http_headers=ctx_param.http_headers, 
                        verbose=ctx_param.verbose)
    
    # send request
    text_list = get_input()
    corr_id = 1000
    datas_dict = {corr_id: text_list}
    infer(ctx_param, datas_dict)

def infer(ctx_param, datas_dict, batch_size=1):
    """
    infer function
    Arguments:
        ctx_param {ContextParameter} -- ContextParameter
        datas_dict {dict} -- key is correlation id, value is text list
        batch_size {int} -- batch size
    """
    infer_ctx = InferContext(ctx_param.server_url, 
                             ctx_param.protocol, 
                             ctx_param.model_name, 
                             ctx_param.model_version,
                             http_headers=ctx_param.http_headers, 
                             verbose=ctx_param.verbose)
    
    result_map = {}
    for corr_id, text_list in datas_dict.items():
        send_request(infer_ctx, corr_id, "", batch_size=batch_size, start_of_sequence=True)
        for text in text_list:
            send_request(infer_ctx, corr_id, text, batch_size=batch_size)
        result = send_request(infer_ctx, corr_id, "", batch_size=batch_size, end_of_sequence=True)
        result_map[corr_id] = result

    print(result_map)

def send_request(ctx, corr_id, text, batch_size=1, start_of_sequence=False, end_of_sequence=False):
    flags = InferRequestHeader.FLAG_NONE
    if start_of_sequence:
        flags = flags | InferRequestHeader.FLAG_SEQUENCE_START
    if end_of_sequence:
        flags = flags | InferRequestHeader.FLAG_SEQUENCE_END
    
    result = ctx.run({ 'INPUT' : [np.array([text])]},
                     { 'OUTPUT' : InferContext.ResultFormat.RAW },
                     batch_size=batch_size, flags=flags, corr_id=corr_id)
    return result


if __name__ == "__main__":
    print("start sample_sequence_client script")
    main()
    print("end sample_sequence_client script")