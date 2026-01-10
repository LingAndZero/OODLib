class BaseModel:

    def __init__(self, num_classes):
        super().__init__()

    def get_output(self, x):
        pass

    def get_feature(self, x):
        pass