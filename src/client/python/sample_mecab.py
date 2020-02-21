from tensorrtserver.api import *
import numpy as np
from util import *


def main():
    # setup environment
    ctx_param = ContextParameter(model_name="mecab_model")
    ctx_param.server_url = "localhost:8000"
    ctx_param.http_headers = None
    ctx_param.verbose = True
    ctx_param.protocol = ProtocolType.from_str("http") # http or grpc
    ctx_param.model_version = -1
    batch_size = 1
    check_health = False

    # print server health and model status if needed
    if check_health:
        check_health_status(url=ctx_param.server_url, 
                        model_name=ctx_param.model_name, 
                        protocol=ctx_param.protocol, 
                        http_headers=ctx_param.http_headers, 
                        verbose=ctx_param.verbose)
    
    # send request
    text_list = get_input(binary_mode=True)
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
        send_request_with_bytes(infer_ctx, corr_id, "", batch_size=batch_size, start_of_sequence=True)
        for text in text_list:
            send_request_with_bytes(infer_ctx, corr_id, text, batch_size=batch_size)
        result = send_request_with_bytes(infer_ctx, corr_id, "", batch_size=batch_size, end_of_sequence=True)
        
        # 形態素解析結果をパースする（|で単語が分割される)
        tmp_text_list = result["OUTPUT"][0][0].decode("utf-8").split("|")
        text_list = []
        for text in tmp_text_list:
            if len(text) == 0:
                continue
            text_list.append(text)
        
        result_map[corr_id] = text_list

    print(result_map)

if __name__ == "__main__":
    print("start script")
    main()
    print("end script")
