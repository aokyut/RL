import numpy as np
import tensorflow as tf
from tfConfig import tfCOnfig
import tensorflow_probability as tfp

"""
ここではPPOのモデルを定義する
学習を行う際のクラスは主に以下の三つ
PPOBrain(一つだけしか作られない)・・・学習時にメモリへの保存。学習。評価。モデルの保存を行えるようにする。
/ act
/ update
PPOMemory
PPOPolicy(エージェントごとに一つと、Brainに一つ)・・・これ単体でエージェントのBrainとして動くようにする。関数として最低限持つ機能は以下の三つ
/ act
PPOTrainer(Brainが使う)・・・学習を行う。関数でもOK
"""

class PPOValue(tf.keras.Model):
    def __init__(self, input_size:int, output_size:int, hidden_layers:int=2, hidden_unit:int=64):
        super(PPOValue, self).__init__()
        self.hidden_dense_layers = [
            tf.keras.layers.Dense(units = hidden_layers, activation="relu")
        ]

        self.value = tf.keras.layers.Dense(units = hidden_layers, activation=None)

    def call(self, x):
        for layers in self.hidden_dense_layers:
            x = layers(x)
        return self.value(x)

class PPOPolicy(tf.keras.Model):
    def __init__(self, input_size:int, output_size:int, hidden_layers:int=2, hidden_unit:int=64):
        super(PPOPolicy, self).__init__()
        
        self.hidden_dense_layers = [
            tf.keras.layers.Dense(units = hidden_layers, activation="relu")
        ]

        self.mu = tf.keras.layers.Dense(units = output_size, activation="tanh")
        self.sig = tf.keras.layers.Dense(units = 1, activation="softplus")
        self.tile_diag = tf.constant([1, output_size]) # sigma の形を mu似合わせるため

    def call(self, x):
        for layer in self.hidden_dense_layers:
            x = layer(x)
        
        mu = self.mu(x)
        sig = self.sig(x)

        norm_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.tile(sig, self.tile_diag))
        
        return norm_dist
    
    def act(self, x):
        norm_dist : tfp.distributions.MultivariateNormalDiag = self.call
        return tf.clip_by_value(norm_dist.sample(), -1, 1)

