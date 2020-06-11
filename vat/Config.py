from types import SimpleNamespace
"""
ここではbrainのパラメータの使用について書く
brainのパラメータとして要求するのは
input_size: int(入力となるデータのサイズ)
＊今の所入力は画像データではないと仮定しておく。
outpu_size: int
is_continuous: bool(連続値かそうでないか)
"""
default = {
    "model": "PPO_discrete",
    "input_size": 0,
    "idden_size": 0,
    "output_size": 0,
    "gamma": 0.99,
    "lambda_value": 0.5,
    "lambda_entropy": 0.01,
    "epsilon_clip": 0.3,
    "K_epoch": 5,
    "n_update": 1000,
    "is_learn": True,
    "device": "cpu",
}
class Config:
    def __init__(self):
        self.__dict__ = default
        self.hash_keys = [
            "model"
            "input_size",
            "output_size",
            "hidden_size",
            "is_learn"
        ]

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    
    def __str__(self):
        hash = ""
        for key in self.hash_keys:
            content = self.__dict__[key]
            hash += key[0] + str(content)
        return hash

    def set(self, arg_dict):
        for key, value in arg_dict.items():
            self.__dict__[key] = value
