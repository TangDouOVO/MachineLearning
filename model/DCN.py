import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DCN(keras.Model):
    def __init__(self, cross_layers_number = 3):
        super(DCN, self).__init__()
        self.cross_number = cross_layers_number
        
    def build(self, input_shape):
        self.feature_dims = input_shape[-1]
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
        super(DCN, self).build(input_shape)
        
    def call(self, x, training = None):
        #cross
        x0 = x
        xl = x
        for _ in range(self.cross_number):
            w = self.add_weight(shape = (self.feature_dims, 1), trainable = True)
            b = self.add_weight(shape = (self.feature_dims, 1), trainable = True)
            #[batch, N, 1] @ [batch, 1, N] -> [batch, N, N]
            x0xl = tf.matmul(tf.reshape(x0, [-1, self.feature_dims, 1]), tf.reshape(xl, [-1, 1, self.feature_dims]))
            #https://zhuanlan.zhihu.com/p/138731311
            #[batch, N, N] @ [N, 1] -> [batch, N]
            x0x1w = tf.squeeze(tf.tensordot(x0xl, w, axes=1))##https://blog.csdn.net/bobobe/article/details/98903601
            xl = x0x1w + tf.squeeze(b) + xl
        #return:xl -> [batch, N]
        
        #deep
        deep = self.drop1(self.bn1(self.fc1(x), training), training)
        deep = self.drop2(self.bn2(self.fc2(deep), training), training)
        deep = self.drop3(self.bn3(self.fc3(deep), training), training)
        #return:deep -> [batch, 24]
        
        #[batch, N + 24]
        out = tf.concat([xl, deep], axis = 1)
        out = layers.Dense(1, activation = tf.sigmoid)
        return out

model = DCN()
model.build((None, 100))
model.summary()
