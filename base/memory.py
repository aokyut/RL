class BaseMemory:
    def append(self, obs, reward, done, action):
        raise NotImplementedError()
    def 