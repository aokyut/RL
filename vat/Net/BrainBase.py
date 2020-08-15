from tfConfig import tfConfig


class BrainBase:
    def __init__(self, config : tfConfig):
        pass

    def act(self, state):
        raise NotImplementedError("You should Implement act function of Brain class")

    def preprocess_learn(self):
        pass

    def update(self):
        pass
