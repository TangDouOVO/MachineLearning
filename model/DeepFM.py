import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
 
class DeepFM(keras.Model):
    def __init__(self, k = 50):
        super(DeepFM, self).__init__()
        self.k = k
        
    def build(self, input_shape):
        #FM -> order1-feature
        self.Dense1 = layers.Dense(1)
        
        #FM -> order2 -feature
        self.N = input_shape[-1]
        #[N, k]
        self.v = self.add_weight(shape = (self.N, self.k), initializer = 'glorot_uniform',
                                 trainable = True)
        
        #DNN
        self.embedding = layers.Embedding(self.N, self.k)
        self.fc1 = layers.Dense(64, activation = tf.tanh)
        self.bn1 = layers.BatchNormalization(axis = 1)
        self.drop1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(32, activation = tf.tanh)
        self.bn2 = layers.BatchNormalization(axis = 1)
        self.drop2 = layers.Dropout(0.5)
        self.fc3 = layers.Dense(1, activation = tf.nn.sigmoid)
        
        super(DeepFM, self).build(input_shape)
        
    def call(self, x, training  = None):
        #FM -> order1-feature
        #[batch, N] -> [batch, 1]
        linear = self.Dense1(x)
        
        #FM -> order2 -feature
        #[batch, N] @ [N, k] -> [batch, k]
        temp1 = tf.pow(tf.matmul(x, self.v), 2)
        #[batch, N] @ [N, k] -> [batch, k]
        temp2 = tf.matmul(tf.pow(x, 2), tf.pow(self.v, 2))
        #[batch, k] -> [batch,1]
        order2 = tf.reduce_sum(temp1 - temp2, axis = 1, keepdims = True) * 0.5
        
        #DNN
        #[batch, N] -> [batch, N, k] -> [batch, N*k]
        dnn = tf.reshape(self.embedding(x), [-1, self.N*self.k])
        #[batch, N*k] -> [batch, 1]
        dnn = self.drop1(self.bn1(self.fc1(dnn), training), training)
        dnn = self.drop2(self.bn2(self.fc2(dnn), training), training)
        dnn = self.fc3(dnn)
        
        out = linear + order2 + dnn
        
        return out
        
        
model = DeepFM()
model.build((None, 100))
model.summary()
