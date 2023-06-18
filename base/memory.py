class Memory:
    def preprocess(self):
        raise NotImplementedError()

    def postprocess(self):
        raise NotImplementedError()

    def push(self):
        raise NotImplementedError()
    
    def sample(self, batch_size: int):
        raise NotImplementedError()