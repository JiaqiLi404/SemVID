class BaseReward:
    def __init__(self):
        super().__init__()
        self.__name__ =self.__class__.__name__

    def __call__(self, **kwargs):
        raise NotImplementedError
