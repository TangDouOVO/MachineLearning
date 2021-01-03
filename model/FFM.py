import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

class FFM(keras.Model):
    def __init__(self, field_num, feature_field_dict, dim_num, k = 2):
        super(FFM, self).__init__()
        self.field_num = field_num
        self.k = k
        self.feature_field_dict = feature_field_dict
        self.dim_num = dim_num

    def build(self, input_shape):
        self.fc = tf.keras.layers.Dense(units = 1,
                                  bias_regularizer = tf.keras.regularizers.l2(0.01),
                                  kernel_regularizer = tf.keras.regularizers.l1(0.02))
        self.w = self.add_weight(shape = (input_shape[-1], self.field_num, self.k),
                                      initializer = 'glorot_uniform',
                                      trainable = True)
        super(FFM, self).build(input_shape)
        
    def call(self, x, training):
        linear = self.fc(x)
        temp = tf.cast(0, tf.float32)
        temp = tf.expand_dims(temp, axis = 0)
        for j1 in range(self.dim_num):
            for j2 in range(j1 + 1, self.dim_num):
                f1 = self.feature_field_dict[j2]
                f2 = self.feature_field_dict[j1]
                #[, , k] * [, , k] = [, , k] -> [1, k]
                ww = tf.expand_dims(tf.multiply(self.w[j1, f2, :], self.w[j2, f1, :]), axis = 0)
                #[x, ] * [x, ] = [x, ] -> [x, 1]
                xx = tf.expand_dims(tf.multiply(x[:, j1], x[:, j2]),axis = 1)
                #[x, 1] @ [1, k] = [x, k]
                store = tf.matmul(xx, ww)
                #[x, k] -> [x]
                temp += tf.reduce_mean(store, keepdims = True, axis = 1)
        out = layers.Add()([linear, temp])
        return tf.sigmoid(out)

store = {}   
for i in range(30):
        store[i] = int(i / 15)
model = FFM(field_num = 2, feature_field_dict = store, dim_num = 30)
model.build((None, 30))
model.summary()
