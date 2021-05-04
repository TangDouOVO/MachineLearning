#一、导入库

import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import tensorflow_addons as tfa
import datetime
from sklearn.model_selection import train_test_split



#二、日期数据生成
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
      
      
#三、模型构建

class seq2seq(keras.Model):
    def __init__(self, source_dict_total_words, source_embedding_size, encoder_num_layers, encoder_rnn_size,
                target_dict_total_words, target_embedding_size, decoder_rnn_size, start_token, batch_size,
                end_token, target_size, attention_layer_size = 2):
        super(seq2seq, self).__init__()
        '''
        encoder参数说明
        --source_dict_total_words：source字典的总单词个数
        --source_embedding_size：souce压缩的长度
        --encoder_num_layers:encoder堆叠的rnn cell数量
        --encoder_rnn_size:encoder中RNN单元的隐层结点数量
        decoder参数说明
        --target_dict_total_words:target字典的总单词个数
        --target_embedding_size:target压缩的长度:
        --decoder_num_layers:decoder堆叠的rnn cell数量
        --decoder_rnn_size:decoder中RNN单元的隐层结点数量
        --target_size:target中句子的长度
        其他参数说明
        --start_token：decoder输入的开始标志<GO>在target字典中的对应数字编号
        --end_token：decoder输入的结束标志<EOS>在target字典中的对应数字编号
        --batch_size：数据的batch_size
        attention的参数说明
        --attention_layer_size：attention层的深度
        '''
        
        self.source_dict_total_words = source_dict_total_words
        self.source_embedding_size = source_embedding_size
        self.encoder_num_layers = encoder_num_layers
        self.encoder_rnn_size = encoder_rnn_size
        self.target_dict_total_words = target_dict_total_words
        self.target_embedding_size = target_embedding_size
        self.decoder_rnn_size = decoder_rnn_size
        self.start_token = start_token
        self.batch_size = batch_size
        self.end_token = end_token
        self.target_size = target_size
        
        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits = True)
        self.optimzer = optimizers.Adam(lr = 1e-2)
        self.seq_len = tf.fill([self.batch_size], self.target_size-1)
        
        self.attention_layer_size = attention_layer_size
        
    
    
    
        #######################Encoder##################################
        #1.embedding
        self.encoder_embedding = layers.Embedding(self.source_dict_total_words, self.source_embedding_size,
                                                  embeddings_initializer = tf.initializers.RandomNormal(0., 0.1))
        #2.单层或多层rnn
        self.encoder_rnn_cells = [layers.LSTMCell(self.encoder_rnn_size, dropout = 0.5) for _ in range(self.encoder_num_layers)]
        self.encoder_stacked_lstm = layers.StackedRNNCells(self.encoder_rnn_cells)
        self.encoder_rnn = layers.RNN(self.encoder_stacked_lstm, return_state = True, return_sequences = True)
        #######################Decoder##################################  
        #1.embedding
        self.decoder_embedding = layers.Embedding(self.target_dict_total_words, 
                                                  self.target_embedding_size,
                                                  embeddings_initializer = 
                                                  tf.initializers.RandomNormal(0., 0.1))
        #2.构造Decoder中的attention_rnn单元
        self.attention = tfa.seq2seq.LuongAttention(self.encoder_rnn_size,
                                                    memory=None, 
                                                    memory_sequence_length=None)
        self.decoder_rnn_attention_cell = tfa.seq2seq.AttentionWrapper(
            cell = keras.layers.LSTMCell(units = self.encoder_rnn_size),
            attention_mechanism = self.attention,
            attention_layer_size = self.attention_layer_size,
            alignment_history = True,                     
        )
        #3.构造Decoder中的dense单元
        self.decoder_dense_layer = layers.Dense(self.target_dict_total_words,
                                                kernel_initializer = tf.compat.v1.truncated_normal_initializer(mean = 0.0,
                                                                                                               stddev = 0.1))
        #3.train
        self.decoder_sampler = tfa.seq2seq.TrainingSampler()
        self.training_decoder = tfa.seq2seq.BasicDecoder(cell = self.decoder_rnn_attention_cell, 
                                                         sampler = self.decoder_sampler,
                                                         output_layer = self.decoder_dense_layer)
        #4.predict     
        self.sampler = tfa.seq2seq.GreedyEmbeddingSampler()
        self.predicting_decoder = tfa.seq2seq.BasicDecoder(cell = self.decoder_rnn_attention_cell,
                                                           sampler = self.sampler,
                                                           output_layer = self.decoder_dense_layer)
        
        
        
    def encode(self, source):
        embedded = self.encoder_embedding(source)
        res_list = self.encoder_rnn(embedded)
        encoder_output = res_list[0]
        encoder_hidden = res_list[1][0]
        encoder_state = res_list[1][1]
        return [encoder_output, encoder_hidden, encoder_state]
    
    def set_attention(self, source):
        attention_output, attention_hidden, attention_state = self.encode(source)
        self.attention.setup_memory(attention_output)
        s = self.decoder_rnn_attention_cell.get_initial_state(batch_size = source.shape[0], 
                                                dtype = tf.float32).clone(cell_state = [attention_hidden, attention_state])
        return s
    
    def train(self, source, target):
        state = self.set_attention(source)
        decoder_input = target[:, :-1]   #ignore <EOS>
        decoder_embeding_input = self.decoder_embedding(decoder_input)
        
        output, _, _ = self.training_decoder(decoder_embeding_input, 
                                             initial_state = state,
                                             sequence_length = self.seq_len)
        return output.rnn_output
    
    
    def predict(self, source, return_align = False):
        state = self.set_attention(source)
        done, inputs, state = self.predicting_decoder.initialize(
            self.decoder_embedding.variables[0],
            start_tokens = tf.fill([source.shape[0], ], self.start_token),
            end_token = self.end_token,
            initial_state = state,
        )

        pred_id = np.zeros((source.shape[0], self.target_size), dtype = np.int32)
        for time in range(self.target_size):
            output, state, inputs, done = self.predicting_decoder.step(
                time = time, inputs = inputs, state = state, training = False)
            pred_id[:, time] = output.sample_id
        if return_align:
            return np.transpose(state.alignment_history.stack().numpy(), (1, 0, 2))
        else:
            state.alignment_history.mark_used()  # otherwise gives warning
            return pred_id
    
    
    def step(self, source, target):
        with tf.GradientTape() as tape:
            logits = self.train(source, target)
            dec_out = target[:, 1:]  # ignore <GO>
            loss = self.cross_entropy(dec_out, logits)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimzer.apply_gradients(zip(grads, self.trainable_variables))
        return loss.numpy()

  #四、数据验证
  epochs = 200
batch_size = 248

data = DateData(4000)
print("1.Chinese time order: yy/mm/dd ", data.date_cn[:3], "\n2.English time order: dd/M/yyyy ", data.date_en[:3])
print("3.vocabularies: \n", data.vocab)
print("4.x index sample: \n{}\n{}".format(data.idx2str(data.x[0]), data.x[0]),
      "\n5.y index sample: \n{}\n{}".format(data.idx2str(data.y[0]), data.y[0]))

train_db = tf.data.Dataset.from_tensor_slices((np.array(data.x), np.array(data.y)))
train_db = train_db.batch(batch_size, drop_remainder=True)

optimizer = optimizers.Adam(lr = 1e-2) 
model = seq2seq(source_dict_total_words = data.num_word, source_embedding_size = 16, encoder_num_layers = 1, encoder_rnn_size = 32,
                target_dict_total_words = data.num_word, target_embedding_size = 16, decoder_rnn_size = 32, 
                start_token = data.start_token, batch_size = batch_size,
                end_token = data.end_token, target_size = 11)#target_size是target单个句子的长度，包括<GO>和<EOS>

for epoch in range(epochs):
    for step, (source, target) in enumerate(train_db):
        loss = model.step(source, target)
    
        if step % 5 == 0:
            target = data.idx2str(np.array(target[0, 1:-1]))
            pred = model.predict(source = source[0:1])
            res = data.idx2str(pred[0])
            src = data.idx2str(np.array(source[0]))
            print(
                "epoch: ", epoch,
                "step:", step,
                "| loss: %.3f" % loss,
                "| input: ", src,
                "| target: ", target,
                "| inference: ", res,
            )
