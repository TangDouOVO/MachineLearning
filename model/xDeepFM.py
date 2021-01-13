import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CIN(keras.Model):
    def __init__(self, D = 100, CIN_layers_dims = [68, 32, 24]):
        super(CIN, self).__init__()
        self.CIN_layers = CIN_layers_dims
        self.D = D
        
        
    def build(self, input_shape):
        self.m = input_shape[-1]
        #self.D = input_shape[-1]
        self.embedding = layers.Embedding(self.m, self.D)
        #deep
        #[None, 100] -> [None, 64] -> [None,48] -> [None, 24]
        self.fc1 = layers.Dense(64, activation = tf.nn.relu)
        self.bn1 = layers.BatchNormalization(axis = 1)
        self.drop1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(48, activation = tf.tanh)
        self.bn2 = layers.BatchNormalization(axis = 1)
        self.drop2 = layers.Dropout(0.5)
        self.fc3 = layers.Dense(24, activation = tf.tanh)
        self.bn3 = layers.BatchNormalization(axis = 1)
        self.drop3 = layers.Dropout(0.5)
        
        super(CIN, self).build(input_shape)
        
    def hadamard(self, x0, xk_pre, hk_in, hk_out):
        #x0 -> [batch, m, D]  xk_pre -> [batch, hk-1, D]
        #[batch, hk-1, D] & [batch, m, D]  -> [batch, hk-1, m, D]
        #[batch, D, hk-1, 1]
        temp1 = tf.expand_dims(tf.transpose(xk_pre, perm = (0, 2, 1)), axis = 3)
        #[batch, D, 1, m]
        temp2 = tf.expand_dims(tf.transpose(x0, perm = (0, 2, 1)), axis = 2)
        #[batch, D, hk-1, 1] @ [batch, D, 1, m] -> [batch, D, hk-1, m]
        temp = tf.matmul(temp1, temp2)
        #[batch, D, hk-1, m] -> [batch, D, hk-1*m]
        out = tf.reshape(temp, [-1, self.D, hk_in * self.m])
        return out
        
    def call(self, x, training = None):
        raw = x
        #CIN
        x = self.embedding(x)
        self.CIN_layers.insert(0, self.m)
        x0 = x
        xk_pre = x
        cin = []
        for i in range(1, len(self.CIN_layers)):
            hk_in = self.CIN_layers[i-1]
            hk_out = self.CIN_layers[i]
            hadamard1 = self.hadamard(x0, xk_pre, hk_in, hk_out)
            #{hk-1, m, hk}
            wk = self.add_weight(shape = (self.m * hk_in, hk_out), trainable = True)
            #[batch, D, hk-1*m] @  [hk-1,*m, hk] -> [batch,D, hk]
            temp = tf.matmul(hadamard1, wk)
            #[batch, Dm, hk] -> [batch, hk, Dm]
            temp = tf.transpose(temp, perm = [0, 2, 1])
            #[batch, hk, Dm] -> [batch, hk]
            cin.append(tf.reduce_sum(temp, 2))
        #cin -> [batch, h1+h2+h3]
        cin = tf.concat(cin, axis = 1)
        
        
        #linear
        #[batch, m] -> [batch. 1]
        linear = layers.Dense(1, activation = tf.tanh)(raw)
        
        
        #deep
        #[batch, m] -> [batch. 24]
        x = tf.reshape(x, [-1, self.D*self.m])
        deep = self.drop1(self.bn1(self.fc1(x), training), training)
        deep = self.drop2(self.bn2(self.fc2(deep), training), training)
        deep = self.drop3(self.bn3(self.fc3(deep), training), training)
        

        out = tf.concat([deep, linear, cin], axis = 1)
        out = layers.Dense(1, activation = tf.nn.sigmoid)(out)
        
        return out

model = CIN()
model.build((None, 200))
model.summary()
