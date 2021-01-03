#Inner Product-based Neural Network
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

class PNN_Inner(keras.Model):
    def __init__(self, D1):
        super(PNN_Inner, self).__init__()
        self.D1 = D1
        
    def build(self, input_shape):
        self.N = input_shape[1]
        self.M = input_shape[-1]
        #[N, M, D1]
        self.Wz = self.add_weight(shape = (self.N, self.M, self.D1),
                                 trainable = True)
        #[D1, N]
        self.theta = self.add_weight(shape = (self.D1, self.N),
                                 trainable = True)
        #[batch, D1] -> [batch, 1]
        self.l1 = layers.Dense(24, activation = tf.nn.leaky_relu)
        self.bn1 = layers.BatchNormalization(axis = 1)
        self.drop1 = layers.Dropout(0.5)
        self.l2 = layers.Dense(12, activation = tf.nn.leaky_relu)
        self.bn2 = layers.BatchNormalization(axis = 1)
        self.drop2 = layers.Dropout(0.5)
        self.l3 = layers.Dense(1, activation = tf.nn.sigmoid)
        super(PNN_Inner, self).build(input_shape)
    
    def call(self, x, training = None):
        #x:[batch, N, M]
        #w:[N, M, D1]
        linear = []
        for i in range(self.D1):
            #[N, M, 1] -> [N, M]
            w = self.Wz[:, :, i]
            #[batch, N. M] * [N, M] -> [batch, N, M]
            temp = tf.multiply(x, w)
            #[batch, N, M] -> [batch, 1]
            temp = tf.expand_dims(tf.reduce_mean(temp, axis = [1, 2]), axis = 1)
            linear.append(temp)
        #linear:[batch, D1]
        linear = tf.concat(linear, axis = 1)
                           
        product = []
        #[batch, N, M] * [batch, N, M] -> [batch, N, M]
        p = tf.multiply(x, x)
        for i in range(self.D1):
            #[N] -> [1, N]
            theta = tf.expand_dims(self.theta[i,:], axis =0)
            #[N, 1] @ [1, N] -> [N, N] 
            w = tf.matmul(tf.transpose(theta), theta)
            #[batch, N, M] -> [batch, M, N]
            f = tf.transpose(p, perm = [0, 2, 1])
            #[batch, M, N] * [N, N] -> [batch, M, N]
            temp = tf.matmul(f, w)
            #[batch, M, N] -> [batch, 1]
            temp = tf.expand_dims(tf.reduce_mean(temp, axis = [1, 2]), axis = 1)             
            product.append(temp)
        #product:[batch, D1]
        product = tf.concat(product, axis = 1)
        #[batch, D1] -> [batch, 2D1]
        out = tf.concat([linear, product],axis = 1)
        out = self.drop1(self.bn1(self.l1(out), training), training)
        out = self.drop2(self.bn2(self.l2(out), training), training)
        out = self.l3(out)
        return out

      
model = PNN_Inner(D1 = 20)
model.build((None,60, 30))
model.summary()



#Outer Product-based Neural Network
import tensorflow as tf
from tensorflow.keras import layers, optimizers, Sequential
from tensorflow import keras

class PNN_Inner(keras.Model):
    def __init__(self, D1):
        super(PNN_Inner, self).__init__()
        self.D1 = D1
        
    def build(self, input_shape):
        self.N = input_shape[1]
        self.M = input_shape[-1]
        #[N, M, D1]
        self.Wz = self.add_weight(shape = (self.N, self.M, self.D1),
                                 trainable = True)
        #[D1, M]
        self.theta = self.add_weight(shape = (self.D1, self.M),
                                 trainable = True)
        #[batch, D1] -> [batch, 1]
        self.l1 = layers.Dense(24, activation = tf.nn.leaky_relu)
        self.bn1 = layers.BatchNormalization(axis = 1)
        self.drop1 = layers.Dropout(0.5)
        self.l2 = layers.Dense(12, activation = tf.nn.leaky_relu)
        self.bn2 = layers.BatchNormalization(axis = 1)
        self.drop2 = layers.Dropout(0.5)
        self.l3 = layers.Dense(1, activation = tf.nn.sigmoid)
        super(PNN_Inner, self).build(input_shape)
    
    def call(self, x, training = None):
        #x:[batch, N, M]
        #w:[N, M, D1]
        linear = []
        for i in range(self.D1):
            #[N, M, 1] -> [N, M]
            w = self.Wz[:, :, i]
            #[batch, N. M] * [N, M] -> [batch, N, M]
            temp = tf.multiply(x, w)
            #[batch, N, M] -> [batch, 1]
            temp = tf.expand_dims(tf.reduce_mean(temp, axis = [1, 2]), axis = 1)
            linear.append(temp)
        #linear:[batch, D1]
        linear = tf.concat(linear, axis = 1)
                           
        product = []
        #[batch, N, M] -> [batch, M]
        fi = tf.reduce_mean(x, axis = 1)
        #[batch, M] !* [batch, M] -> [batch, M, M]
        p = tf.einsum('ai,aj->aij', fi, fi)
        for i in range(self.D1):
            #[M] -> [1, M]
            theta = tf.expand_dims(self.theta[i,:], axis =0)
            #[M, 1] @ [1, M] -> [M, M] 
            w = tf.matmul(tf.transpose(theta), theta)
            #[batch, M, M] * [M, M] -> [batch, M, M]
            temp = tf.matmul(p, w)
            #[batch, M, M] -> [batch, 1]
            temp = tf.expand_dims(tf.reduce_mean(temp, axis = [1, 2]), axis = 1)             
            product.append(temp)
        #product:[batch, D1]
        product = tf.concat(product, axis = 1)
        #[batch, D1] -> [batch, 2D1]
        out = tf.concat([linear, product],axis = 1)
        out = self.drop1(self.bn1(self.l1(out), training), training)
        out = self.drop2(self.bn2(self.l2(out), training), training)
        out = self.l3(out)
        return out

      
model = PNN_Inner(D1 = 20)
model.build((None,60, 30))
model.summary()
