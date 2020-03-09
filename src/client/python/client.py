from abc import ABCMeta, abstractmethod


class AbstractClient:
    __metaclass__ = ABCMeta
    
    def __init__(self):
        # TODO : imp
        pass

    @abstractmethod
    def draw_page(self, context_parameter):
        pass
