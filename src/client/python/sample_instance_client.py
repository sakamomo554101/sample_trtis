from tensorrtserver.api import *
import numpy as np
from util import *


def main():
    # setup environment
    server_url = "localhost:8000"
    http_headers = []
    http_headers.append(b"test1:hoge")
    verbose = True
    protocol = ProtocolType.from_str("http") # http or grpc
    model_name = "sample_instance"
    model_version = -1

    # print server health and model status
    check_health_status(url=server_url, model_name=model_name, protocol=protocol, http_headers=http_headers, verbose=verbose)

    # infer
    infer(url=server_url, model_name=model_name, model_version=model_version, protocol=protocol, http_headers=http_headers, verbose=verbose)


def infer(url, model_name, model_version, protocol, http_headers, verbose):
    infer_ctx = InferContext(url, protocol, model_name, model_version,
                             http_headers=http_headers, verbose=verbose)
    
    batch_size = 1
    input_data = [np.array(["testtesttest"])]
    result = infer_ctx.run(
        {"INPUT" : input_data},
        {"OUTPUT" : InferContext.ResultFormat.RAW},
        batch_size
    )
    print(result)

if __name__ == "__main__":
    print("start sample_instance_client script")
    main()
    print("end sample_instance_client script")
