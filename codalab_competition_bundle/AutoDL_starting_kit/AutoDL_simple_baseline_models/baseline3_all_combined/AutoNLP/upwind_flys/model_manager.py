"""
MIT License

Copyright (c) 2019 Lenovo Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import keras
from keras.layers import Input, LSTM, Dense
from keras.layers import Dropout
from keras.layers import Embedding, Flatten, Conv1D, concatenate
from keras.layers import SeparableConv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalAveragePooling1D
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

EMBEDDING_DIM=300
MAX_VOCAB_SIZE=20000

class ModelGenerator(object):
    def __init__(self ,
                feature_mode,
                load_pretrain_emb = False,
                data_feature = None,
                meta_data_feature = None,
                fasttext_embeddings_index = None):

        self.cnn_model_lib = {'text_cnn': ModelGenerator.text_cnn_model,
                              'sep_cnn_model': ModelGenerator.sep_cnn_model,
                              'lstm_model': ModelGenerator.lstm_model,
                              }

        self.data_feature = data_feature
        self.load_pretrain_emb = load_pretrain_emb
        self.meta_data_feature = meta_data_feature
        # self.model_name = model_name

        if data_feature is not None:
          self.num_features = data_feature['num_features']
          self.word_index = data_feature['word_index']
          self.num_class = data_feature['num_class']
          self.max_length = data_feature['max_length']
          self.input_shape = data_feature['input_shape']

        self.feature_mode = feature_mode
        self.embedding_matrix = None

        if self.feature_mode == 0:
           self.load_pretrain_emb = False
        self.fasttext_embeddings_index = fasttext_embeddings_index


    
    def build_model(self, model_name,data_feature):
        if model_name == 'svm' :
            model = LinearSVC(random_state=0, tol=1e-5, max_iter=500)
            self.model = CalibratedClassifierCV(model)
            #self.model_name = 'svm'
        else:
            #if self.load_pretrain_emb:
            #    self.generate_emb_matrix()
            #else:
            #    self.embedding_matrix = None
            self.num_features = data_feature['num_features']
            self.word_index = data_feature['word_index']
            self.num_class = data_feature['num_class']
            self.max_length = data_feature['max_length']
            self.input_shape = data_feature['input_shape']
            if self.load_pretrain_emb:
                self.generate_emb_matrix()
            else:
                self.embedding_matrix = None

            kwargs = {'embedding_matrix':self.embedding_matrix,
                      'input_shape':data_feature['input_shape'],
                      'max_length':data_feature['max_length'],
                      'num_features':data_feature['num_features'],
                      'num_classes':data_feature['num_class']}

            #self.model_name = 'text_cnn'
            self.model = self.cnn_model_lib[model_name](**kwargs)
            self.model.compile(loss="categorical_crossentropy",
                           optimizer=keras.optimizers.RMSprop(),
                           metrics=["accuracy"])

            if self.model_name not in self.cnn_model_lib.keys():
              raise Exception('incorrect model name')
        return self.model 

    def model_pre_select(self, call_num):
        if call_num == 0 :
            self.model_name = 'svm'
        elif call_num == 1:
            #self.set_feature_mode(self.meta_data_feature)
            '''if self.load_pretrain_emb:
                self.generate_emb_matrix()
            else:
                self.embedding_matrix = None'''
            self.model_name = 'text_cnn'
            
            if self.model_name not in self.cnn_model_lib.keys():
              raise Exception('incorrect model name')
        return self.model_name

    def generate_emb_matrix(self):

        cnt = 0
        self.embedding_matrix = np.zeros((self.num_features, EMBEDDING_DIM))
        for word, i in self.word_index.items():
            if i >= self.num_features:
                continue
            embedding_vector = self.fasttext_embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector
            else:
                # self.embedding_matrix[i] = np.zeros(300)
                self.embedding_matrix[i] = np.random.uniform(
                    -0.05, 0.05, size=EMBEDDING_DIM)
                cnt += 1

        print('fastText oov words: %s' % cnt)

    @staticmethod
    def _get_last_layer_units_and_activation(num_classes):
        """Gets the # units and activation function for the last network layer.

        Args:
            num_classes: Number of classes.

        Returns:
            units, activation values.
        """
        activation = 'softmax'
        units = num_classes
        return units, activation

    @staticmethod
    def text_cnn_model(input_shape,
                       embedding_matrix,
                       max_length,
                       num_features,
                       num_classes,
                       input_tensor=None,
                       filters=64,
                       emb_size=300,
                       ):

        inputs = Input(name='inputs', shape=[max_length], tensor=input_tensor)
        if embedding_matrix is None:
            layer = Embedding(input_dim=num_features,
                              output_dim=emb_size,
                              input_length=input_shape)(inputs)
        else:
            num_features = MAX_VOCAB_SIZE
            layer = Embedding(input_dim=num_features,
                              output_dim=emb_size,
                              input_length=input_shape,
                              embeddings_initializer=keras.initializers.Constant(
                                  embedding_matrix))(inputs)

        cnns = []
        filter_sizes = [2, 3, 4, 5]
        for size in filter_sizes:
            cnn_l = Conv1D(filters,
                           size,
                           padding='same',
                           strides=1,
                           activation='relu')(layer)
            pooling_l = MaxPooling1D(max_length - size + 1)(cnn_l)
            pooling_l = Flatten()(pooling_l)
            cnns.append(pooling_l)

        cnn_merge = concatenate(cnns, axis=-1)
        out = Dropout(0.2)(cnn_merge)
        main_output = Dense(num_classes, activation='softmax')(out)
        model = keras.models.Model(inputs=inputs, outputs=main_output)
        return model

    @staticmethod
    def sep_cnn_model(input_shape,
                      max_length,
                      num_classes,
                      num_features,
                      embedding_matrix,
                      input_tensor=None,
                      emb_size=300,
                      blocks=1,
                      filters=64,
                      kernel_size=4,
                      dropout_rate=0.25):
        op_units, op_activation = ModelGenerator._get_last_layer_units_and_activation(num_classes)

        inputs = Input(name='inputs', shape=[max_length], tensor=input_tensor)
        if embedding_matrix is None:
            layer = Embedding(input_dim=num_features,
                              output_dim=emb_size,
                              input_length=input_shape)(inputs)
        else:
            num_features = MAX_VOCAB_SIZE
            layer = Embedding(input_dim=num_features,
                              output_dim=emb_size,
                              input_length=input_shape,
                              embeddings_initializer=keras.initializers.Constant(
                                  embedding_matrix))(inputs)

        for _ in range(blocks - 1):
            layer = Dropout(rate=dropout_rate)(layer)
            layer = SeparableConv1D(filters=filters,
                                    kernel_size=kernel_size,
                                    activation='relu',
                                    bias_initializer='random_uniform',
                                    depthwise_initializer='random_uniform',
                                    padding='same')(layer)
            layer = SeparableConv1D(filters=filters,
                                    kernel_size=kernel_size,
                                    activation='relu',
                                    bias_initializer='random_uniform',
                                    depthwise_initializer='random_uniform',
                                    padding='same')(layer)
            layer = MaxPooling1D(pool_size=3)(layer)

        layer = SeparableConv1D(filters=filters * 2,
                                kernel_size=kernel_size,
                                activation='relu',
                                bias_initializer='random_uniform',
                                depthwise_initializer='random_uniform',
                                padding='same')(layer)
        layer = SeparableConv1D(filters=filters * 2,
                                kernel_size=kernel_size,
                                activation='relu',
                                bias_initializer='random_uniform',
                                depthwise_initializer='random_uniform',
                                padding='same')(layer)

        layer = GlobalAveragePooling1D()(layer)
        # model.add(MaxPooling1D())
        layer = Dropout(rate=0.5)(layer)
        layer = Dense(op_units, activation=op_activation)(layer)
        model = keras.models.Model(inputs=inputs, outputs=layer)
        return model

    @staticmethod
    def lstm_model(max_length,
                   num_classes,
                   num_features,
                   embedding_matrix=None,
                   hidden_state_size=128,
                   fc1_size=256,
                   dropout_rate=0.15):
        inputs = Input(name='inputs', shape=[max_length])
        layer = Embedding(num_features, hidden_state_size,
                          input_length=max_length)(inputs)
        # layer = LSTM(hidden_state_size, return_sequences=True)(layer)
        layer = LSTM(hidden_state_size)(layer)
        layer = Dense(fc1_size, activation="relu", name="FC1")(layer)
        layer = Dropout(dropout_rate)(layer)
        layer = Dense(num_classes, activation="softmax", name="FC2")(layer)
        model = keras.models.Model(inputs=inputs, outputs=layer)
        return model






