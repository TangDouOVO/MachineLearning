import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
 
class AFM(keras.Model):
    def __init__(self, t = 20):
        super(AFM, self).__init__()
        self.t = t
    
    def build(self, input_shape):
        self.linear = layers.Dense(1, bias_regularizer = keras.regularizers.l2(0.01),
                                  kernel_regularizer = keras.regularizers.l1(0.02))
        self.N = input_shape[1]
        self.M = input_shape[-1]
        #[N, M]
        self.v = self.add_weight(shape = (self.N, self.M),
                                      initializer = 'glorot_uniform',
                                      trainable = True)
        #attention
        self.a1 = layers.Dense(self.t, activation = tf.nn.relu)
        self.a2 = layers.Dense(1, activation = tf.nn.softmax)
        #product
        self.p = layers.Dense(1)
        super(AFM, self).build(input_shape)
    
    def call(self, dense):
        #spare:[batch, N] -> [batch, 1]
        sparse = dense[:, :, 0]
        linear = self.linear(sparse)
        
        #Pair-wise Interaction Layer
        #dense:[batch, N, M]
        fPI = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                #[M] * [M] -> [M]
                vivj = tf.multiply(self.v[i, :], self.v[j, :])
                #[batch, M] * [batch, M] -> [batch, M]
                xixj = tf.multiply(dense[:, i, :], dense[:, j, :])
                #[batch, M] @ [M] -> [batch, M]
                temp = tf.einsum('aj,j->aj', xixj, vivj)#tf.matmul(xixj, vivj)
                fPI.append(temp)
        #[N(N-1)/2, batch, M]
        fPI = tf.stack(fPI, axis = 0)
        #[batch, N(N-1)/2, M]
        fPI = tf.transpose(fPI, [1, 0, 2])
                
        #attention
        #[batch, N(N-1) / 2, M] -> [batch, N(N-1)/ 2, t]
        temp = self.a1(fPI)
        #[batch, N(N-1)/2, 1]
        attention_weight = self.a2(temp)
        
        #Attention-based Pooling Layer
        #attention_weight:[batch, N(N-1)/2, 1]  
        #fPI: [batch, N(N-1)/2, M]
        #[batch, N(N-1)/2, 1] * [batch, N(N-1)/2, M] -> [batch, N(N-1)/2, M]
        store = tf.multiply(attention_weight, fPI)
        #[batch, N(N-1)/2, M] -> [batch, M]
        fAtt_fPI = tf.reduce_mean(store, axis = 1)
        #[batch, M] -> [batch, 1]
        product_ = self.p(fAtt_fPI)
        
        out = linear + product_
        
        return out
        
        

model = AFM()
model.build((None, 50, 30))
model.summary()
