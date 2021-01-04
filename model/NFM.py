import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class NFM(keras.Model):
    def __init__(self, k = 120):
        super(NFM, self).__init__()
        self.k = k
        
    def build(self, input_shape):
        #linear
        self.linear = layers.Dense(1, bias_regularizer = tf.keras.regularizers.l2(0.01),
                                  kernel_regularizer = tf.keras.regularizers.l1(0.02))
        #NFM
        #[n, k]
        self.v = self.add_weight(shape = (input_shape[-1], self.k),
                            initializer = 'glorot_uniform',
                            trainable = True)
        #Dense
        self.fc1 = layers.Dense(64, activation = tf.tanh)
        self.bn1 = layers.BatchNormalization(axis = 1)
        self.drop1 = layers.Dropout(0.5)
        
        self.fc2 = layers.Dense(32, activation = tf.tanh)
        self.bn2 = layers.BatchNormalization(axis = 1)
        self.drop2 = layers.Dropout(0.5)
        
        self.fc3 = layers.Dense(1, activation = tf.sigmoid)
        self.bn3 = layers.BatchNormalization(axis = 1)
        self.drop3 = layers.Dropout(0.5)
        
        super(NFM, self).build(input_shape)
        
    def call(self, x, training = None):
        #linear
        #spare:[batch, n] -> [batch, 1]
        linear = self.linear(x)
        
        #NFM
        #[batch, n] @ [n, k] -> [batch, k]
        temp1 = tf.matmul(x, self.v)
        #[batch, k] -> [batch, k]
        temp1 = tf.pow(temp1, 2)
        #[batch, n]**2 @ [n, k]**2 - > [batch, k]
        temp2 = tf.matmul(tf.pow(x, 2), tf.pow(self.v, 2))
        #[batch, k] - [batch, k]  -> [batch, k]
        fBI = 0.5 * (temp1 - temp2)
        #Dense
        #[batch, k] -> [batch, 1]
        nfm = self.drop1(self.bn1(self.fc1(fBI), training), training)
        nfm = self.drop2(self.bn2(self.fc2(fBI), training), training)
        nfm = self.drop3(self.bn3(self.fc3(fBI), training), training)
        
        out = linear + nfm
        return out

model = NFM()
model.build((None, 200))
model.summary()
