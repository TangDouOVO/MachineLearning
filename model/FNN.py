class FNN(keras.Model):
    def __init__(self, w_init, v_init):
        super(FNN, self).__init__()
        self.w_init = w_init
        self.v_init = v_init
        
    def build(self, input_shape):
        #[None, k=30] -> [None, 24] -> [None, 8] -> [None, 2] 
        self.fc1 = layers.Dense(24, activation = tf.tanh)
        self.bn1 = layers.BatchNormalization(axis = 1)
        self.drop1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(8, activation = tf.tanh)
        self.bn2 = layers.BatchNormalization(axis = 1)
        self.drop2 = layers.Dropout(0.5)
        self.fc3 = layers.Dense(2, activation = None)
        super(FNN, self).build(input_shape)
        
    def call(self, x, training = None):
        #[None, dim]@[dim, 1]+ [None, dim]@[dim, k] = [None, k]
        out = x @ self.w_init + x @ self.v_init
        out = self.drop1(self.bn1(self.fc1(out), training), training)
        out = self.drop2(self.bn2(self.fc2(out), training), training)
        out = self.fc3(out)
        return out
    
test = FNN(model.trainable_variables[0], model.trainable_variables[2])#Parameters from FM pre-training
test.build((None, 30))
test.summary()
