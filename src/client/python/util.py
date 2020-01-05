from tensorrtserver.api import *


class ContextParameter:
    def __init__(self, model_name):
        self.server_url = "localhost:8000"
        self.http_headers = None
        self.verbose = True
        self.protocol = ProtocolType.from_str("http") # http or grpc
        self.model_name = model_name
        self.model_version = -1


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
