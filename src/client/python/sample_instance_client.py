from tensorrtserver.api import *
import numpy as np


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
