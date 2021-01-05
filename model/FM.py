from tensorflow.keras import layers
from tensorflow import keras

class FM(keras.Model):
    def __init__(self, k = 4):
        super(FM, self).__init__()
        self.k = k
        
    def build(self,input_shape):
        self.fc = tf.keras.layers.Dense(units = 1,
                                  bias_regularizer = tf.keras.regularizers.l2(0.01),
                                  kernel_regularizer = tf.keras.regularizers.l1(0.02))

        self.v = self.add_weight(shape = (input_shape[-1], self.k),
                                      initializer = 'glorot_uniform',
                                      trainable = True)
        super(FM, self).build(input_shape)
        
    def call(self, x,training=True):
        #[1, dim]@[dim, k] = [1, k]
        a = tf.pow(tf.matmul(x, self.v), 2)
        #[1, dim] @[dim, k] = [1, k]
        b = tf.matmul(tf.pow(x, 2), tf.pow(self.v, 2))
        #[1, dim] @[dim, 1] + reduce_mean([1, k] - [1, k])
        linear = self.fc(x)
        add = tf.keras.layers.Add()([linear, tf.reduce_mean(a - b, axis = 1, keepdims = True)*0.5])
        return tf.sigmoid(add) 
    

model = FM()
model.build((None, 30))
model.summary()
