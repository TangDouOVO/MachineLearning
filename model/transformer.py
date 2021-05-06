#一、导入库
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
import pickle
import os
import datetime


#二、生成数据
PAD_ID = 0
class DateData:
    def __init__(self, n):
        np.random.seed(1)
        self.date_cn = []
        self.date_en = []
        for timestamp in np.random.randint(143835585, 2043835585, n):
            date = datetime.datetime.fromtimestamp(timestamp)
            self.date_cn.append(date.strftime("%y-%m-%d"))
            self.date_en.append(date.strftime("%d/%b/%Y"))
        self.vocab = set(
            [str(i) for i in range(0, 10)] + ["-", "/", "<GO>", "<EOS>"] + [
                i.split("/")[1] for i in self.date_en])
        self.v2i = {v: i for i, v in enumerate(sorted(list(self.vocab)), start=1)}
        self.v2i["<PAD>"] = PAD_ID
        self.vocab.add("<PAD>")
        self.i2v = {i: v for v, i in self.v2i.items()}
        self.x, self.y = [], []
        for cn, en in zip(self.date_cn, self.date_en):
            self.x.append([self.v2i[v] for v in cn])
            self.y.append(
                [self.v2i["<GO>"], ] + [self.v2i[v] for v in en[:3]] + [
                    self.v2i[en[3:6]], ] + [self.v2i[v] for v in en[6:]] + [
                    self.v2i["<EOS>"], ])
        self.x, self.y = np.array(self.x), np.array(self.y)
        self.start_token = self.v2i["<GO>"]
        self.end_token = self.v2i["<EOS>"]

    def sample(self, n=64):
        bi = np.random.randint(0, len(self.x), size=n)
        bx, by = self.x[bi], self.y[bi]
        decoder_len = np.full((len(bx),), by.shape[1] - 1, dtype=np.int32)
        return bx, by, decoder_len

    def idx2str(self, idx):
        x = []
        for i in idx:
            x.append(self.i2v[i])
            if i == self.end_token:
                break
        return "".join(x)

    @property
    def num_word(self):
        return len(self.vocab)
      
      
      
  def pad_zero(seqs, max_len):
    padded = np.full((len(seqs), max_len), fill_value=PAD_ID, dtype=np.int32)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
    return padded
  
  
  def set_soft_gpu(soft_gpu):
    import tensorflow as tf
    if soft_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")#https://blog.csdn.net/pldcat/article/details/108856657
            
            
#三、模型构建
MODEL_DIM = 32
MAX_LEN = 12
N_LAYER = 3
N_HEAD = 4
DROP_RATE = 0.1


class MultiHead(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.n_head = n_head
        self.model_dim = model_dim
        self.head_dim = model_dim // n_head
        self.wq = keras.layers.Dense(n_head * self.head_dim)
        self.wk = keras.layers.Dense(n_head * self.head_dim)
        self.wv = keras.layers.Dense(n_head * self.head_dim)  

        self.o_dense = keras.layers.Dense(model_dim)
        self.o_drop = keras.layers.Dropout(rate=drop_rate)
        self.attention = None

    def call(self, q, k, v, mask, training):
        #[batch_size, max_len, embedding] -> [batch_size, max_len, model_dim]
        _q = self.wq(q)     
        _k, _v = self.wk(k), self.wv(v)   
        #[batch_size, max_len, model_dim] -> [batch_size, n_head, max_len, head_dim]
        _q = self.split_heads(_q)  
        _k, _v = self.split_heads(_k), self.split_heads(_v) 
        #[batch_size, n_head, max_len, head_dim] -> [batch_size, max_len, n_head*head_dim] 
        context = self.scaled_dot_product_attention(_q, _k, _v, mask)   
        #[batch_size, max_len, n_head*head_dim]  -> [batch_size, max_len, model_dim] 
        o = self.o_dense(context)     
        o = self.o_drop(o, training=training)
        return o


    def split_heads(self, x):
        x = tf.reshape(x, (x.shape[0], x.shape[1], self.n_head, self.head_dim)) 
        return tf.transpose(x, perm=[0, 2, 1, 3]) 


    def scaled_dot_product_attention(self, q, k, v, mask=None):
        #scale dk = head_dim
        dk = tf.cast(k.shape[-1], dtype=tf.float32)
        #[batch_size, n_head, max_len, head_dim] * [batch_size, n_head, head_dim, max_len] -> [batch_size, n_head, max_len, max_len]
        score = tf.matmul(q, k, transpose_b=True) / (tf.math.sqrt(dk) + 1e-8)
        if mask is not None:
            score += mask * -1e9
        self.attention = tf.nn.softmax(score, axis=-1)   
        #[batch_size, n_head, max_len, max_len] * [batch_size, n_head, max_len, head_dim] -> [batch_size, n_head, max_len, head_dim]
        context = tf.matmul(self.attention, v)     
        #[batch_size, n_head, max_len, head_dim] -> [batch_size, max_len, n_head, head_dim]   
        context = tf.transpose(context, perm=[0, 2, 1, 3])  
        #[batch_size, max_len, n_head, head_dim]  -> [batch_size, max_len, n_head*head_dim] 
        context = tf.reshape(context, (context.shape[0], context.shape[1], -1))     
        return context


    


class PositionWiseFFN(keras.layers.Layer):
    def __init__(self, model_dim):
        super().__init__()
        dff = model_dim * 4
        #两层全连接层
        self.l = keras.layers.Dense(dff, activation=keras.activations.relu)
        self.o = keras.layers.Dense(model_dim)

    def call(self, x):
        o = self.l(x)
        o = self.o(o)
        return o        


class EncodeLayer(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.ln = [keras.layers.LayerNormalization(axis=-1) for _ in range(2)] 
        self.mh = MultiHead(n_head, model_dim, drop_rate)
        self.ffn = PositionWiseFFN(model_dim)
        self.drop = keras.layers.Dropout(drop_rate)

    def call(self, xz, training, mask):
        attn = self.mh.call(xz, xz, xz, mask, training)       
        #[batch_size, max_len, n_head*head_dim] + [batch_size, max_len, n_head*head_dim] -> [batch_size, max_len, n_head*head_dim] 
        o1 = self.ln[0](attn + xz)
        ffn = self.drop(self.ffn.call(o1), training)
        o = self.ln[1](ffn + o1)        
        return o


class Encoder(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        super().__init__()
        self.ls = [EncodeLayer(n_head, model_dim, drop_rate) for _ in range(n_layer)]

    def call(self, xz, training, mask):
        for l in self.ls:
            xz = l.call(xz, training, mask)
        return xz       


class DecoderLayer(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.ln = [keras.layers.LayerNormalization(axis=-1) for _ in range(3)] 
        self.drop = keras.layers.Dropout(drop_rate)
        self.mh = [MultiHead(n_head, model_dim, drop_rate) for _ in range(2)]
        self.ffn = PositionWiseFFN(model_dim)

    def call(self, yz, xz, training, yz_look_ahead_mask, xz_pad_mask):
        attn = self.mh[0].call(yz, yz, yz, yz_look_ahead_mask, training)       
        o1 = self.ln[0](attn + yz)
        attn = self.mh[1].call(o1, xz, xz, xz_pad_mask, training)       
        o2 = self.ln[1](attn + o1)
        ffn = self.drop(self.ffn.call(o2), training)
        o = self.ln[2](ffn + o2)
        return o


class Decoder(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        super().__init__()
        self.ls = [DecoderLayer(n_head, model_dim, drop_rate) for _ in range(n_layer)]

    def call(self, yz, xz, training, yz_look_ahead_mask, xz_pad_mask):
        for l in self.ls:
            yz = l.call(yz, xz, training, yz_look_ahead_mask, xz_pad_mask)
        return yz


class PositionEmbedding(keras.layers.Layer):
    def __init__(self, max_len, model_dim, n_vocab):
        super().__init__()
        #[max_len, 1]
        pos = np.arange(max_len)[:, None]
        #[max_len, 1] / [1, model_dim] -> [max_len, model_dim]
        pe = pos / np.power(10000, 2. * np.arange(model_dim)[None, :] / model_dim) 
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        #[max_len, model_dim] -> [1, max_len, model_dim]
        pe = pe[None, :, :]
        self.pe = tf.constant(pe, dtype=tf.float32)
        self.embeddings = keras.layers.Embedding(
            input_dim=n_vocab, output_dim=model_dim,
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        )

    def call(self, x):
        #[batch_size, max_len, embedding] + [1, max_len, model_dim] -> [batch_size, max_len, embedding]
        x_embed = self.embeddings(x) + self.pe 
        return x_embed


class Transformer(keras.Model):
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, drop_rate=0.1, padding_idx=0):
        super().__init__()
        self.max_len = max_len
        self.padding_idx = padding_idx

        self.embed = PositionEmbedding(max_len, model_dim, n_vocab)
        self.encoder = Encoder(n_head, model_dim, drop_rate, n_layer)
        self.decoder = Decoder(n_head, model_dim, drop_rate, n_layer)
        self.o = keras.layers.Dense(n_vocab)

        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
        self.opt = keras.optimizers.Adam(0.002)

    def call(self, x, y, training=None):
        #[batch_size, max_len] -> [batch_size, max_len, embedding] 
        x_embed, y_embed = self.embed(x), self.embed(y)
        #eg: [1, 2, 3, 0, 0] -> [[[0, 0, 0, 1, 1]]]
        pad_mask = self._pad_mask(x)
        #[batch_size, max_len, n_head*head_dim] 
        encoded_z = self.encoder.call(x_embed, training, mask = pad_mask)
        decoded_z = self.decoder.call(y_embed, encoded_z, training, yz_look_ahead_mask = self._look_ahead_mask(y), xz_pad_mask=pad_mask)
        #[batch, max_len, n_vocab]
        o = self.o(decoded_z)
        return o

    def step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.call(x, y[:, :-1], training=True)#ignore <EOS>
            pad_mask = tf.math.not_equal(y[:, 1:], self.padding_idx)#ignore <GO>
            loss = tf.reduce_mean(tf.boolean_mask(self.cross_entropy(y[:, 1:], logits), pad_mask))
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss, logits

    def _pad_bool(self, seqs):
        return tf.math.equal(seqs, self.padding_idx)

    def _pad_mask(self, seqs):
        mask = tf.cast(self._pad_bool(seqs), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def _look_ahead_mask(self, seqs):
        mask = 1 - tf.linalg.band_part(tf.ones((self.max_len, self.max_len)), -1, 0)
        mask = tf.where(self._pad_bool(seqs)[:, tf.newaxis, tf.newaxis, :], 1, mask[tf.newaxis, tf.newaxis, :, :])
        return mask  # (step, step)




def train(model, data, step):
    # training
    t0 = time.time()
    for t in range(step):
        #[batch_size, len]
        bx, by, seq_len = data.sample(64)
        #x:[batch_size, len] -> [batch_size, max_len] 长度不够max_len的用0填充
        #y:[batch_size, len] -> [batch_size, max_len+1]#？？？？？？？
        bx, by = pad_zero(bx, max_len=MAX_LEN), pad_zero(by, max_len = MAX_LEN + 1)
        loss, logits = model.step(bx, by)
        if t % 50 == 0:
            #模型训练的预测y的返回
            logits = logits[0].numpy()
            t1 = time.time()
            print(
                "step: ", t,
                "| time: %.2f" % (t1 - t0),
                "| loss: %.4f" % loss.numpy(),
                "| target: ", "".join([data.i2v[i] for i in by[0, 1:10]]),
                "| inference: ", "".join([data.i2v[i] for i in np.argmax(logits, axis=1)[:10]]),
            )
            t0 = t1




if __name__ == "__main__":
    set_soft_gpu(True)
    d = DateData(4000)
    print("Chinese time order: yy/mm/dd ", d.date_cn[:3], "\nEnglish time order: dd/M/yyyy ", d.date_en[:3])
    print("vocabularies: ", d.vocab)
    print("x index sample: \n{}\n{}".format(d.idx2str(d.x[0]), d.x[0]),
          "\ny index sample: \n{}\n{}".format(d.idx2str(d.y[0]), d.y[0]))
    m = Transformer(MODEL_DIM, MAX_LEN, N_LAYER, N_HEAD, d.num_word, DROP_RATE)
    train(m, d, step=1200)
