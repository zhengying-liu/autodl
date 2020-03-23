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

import numpy as np
import random
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing import text
from keras.preprocessing import sequence

os.system("pip install jieba_fast")
import jieba_fast as jieba

CHI_WORD_LENGTH = 2
MAX_CHAR_LENGTH = 96
MAX_VOCAB_SIZE=20000
MAX_SEQ_LENGTH=301
MAX_VALID_PERCLASS_SAMPLE=400
MAX_SAMPLE_TRIAN=18000
MAX_TRAIN_PERCLASS_SAMPLE=800

def clean_en_text(dat, ratio=0.1, is_ratio=True):

    REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')

    ret = []
    for line in dat:
        # text = text.lower() # lowercase text
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        # line = BAD_SYMBOLS_RE.sub('', line)
        line = line.strip()
        line_split = line.split()

        if is_ratio:
            NUM_WORD = max(int(len(line_split) * ratio), MAX_SEQ_LENGTH)
        else:
            NUM_WORD = MAX_SEQ_LENGTH

        if len(line_split) > NUM_WORD:
            line = " ".join(line_split[0:NUM_WORD])
        ret.append(line)
    return ret


def clean_zh_text(dat, ratio=0.1, is_ratio=False):
    REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')

    ret = []
    for line in dat:
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = line.strip()

        if is_ratio:
            NUM_CHAR = max(int(len(line) * ratio), MAX_CHAR_LENGTH)
        else:
            NUM_CHAR = MAX_CHAR_LENGTH

        if len(line) > NUM_CHAR:
            # line = " ".join(line.split()[0:MAX_CHAR_LENGTH])
            line = line[0:NUM_CHAR]
        ret.append(line)
    return ret

def _tokenize_chinese_words(text):
    return ' '.join(jieba.cut(text, cut_all=False))



class DataGenerator(object):
    def __init__(self,
                train_dataset,
                metadata):

        self.meta_data_x,\
        self.meta_data_y = train_dataset
        self.metadata = metadata

        self.num_classes = self.metadata['class_num']
        self.num_samples_train = self.metadata['train_num']
        self.language = metadata['language']

        print("num_samples_train:", self.num_samples_train)
        print("num_class_train:", self.num_classes)

        self.val_index = None
        self.tokenizer = None
        self.max_length = None
        self.sample_num_per_class = None
        self.data_feature = {}

    def set_sample_num_per_class(self, sample_num_per_class):
        self.sample_num_per_class = sample_num_per_class

    #generate validation dataset index
    def sample_valid_index(self):

        all_index = []
        for i in range(self.num_classes):
            all_index.append(
                list(np.where((self.meta_data_y[:, i] == 1) == True)[0]))
        val_index = []
        for i in range(self.num_classes):
            tmp = random.sample(all_index[i],
                                int(len(all_index[i]) * 0.2))
            if len(tmp) > MAX_VALID_PERCLASS_SAMPLE:
                tmp = tmp[:MAX_VALID_PERCLASS_SAMPLE]
            val_index += tmp
            all_index[i] = list(
                set(all_index[i]).difference(set(tmp)))
        self.all_index = all_index
        self.val_index = val_index

    #generate training meta dataset index
    def sample_train_index(self, all_index):

        train_label_distribution = np.sum(np.array(self.meta_data_y), 0)
        print("train_distribution: ", train_label_distribution)
        self.max_sample_num_per_class = int(
            np.max(train_label_distribution) * 4 / 5)

        if self.sample_num_per_class is None:
            if self.num_samples_train < MAX_SAMPLE_TRIAN:
                self.sample_num_per_class = self.max_sample_num_per_class
            else:
                self.sample_num_per_class = min(MAX_TRAIN_PERCLASS_SAMPLE,
                                           self.max_sample_num_per_class)
        meta_train_index = []
        for i in range(self.num_classes):
            if len(all_index[i]) < self.sample_num_per_class:
                tmp = all_index[i] * int(
                    self.sample_num_per_class / len(all_index[i]))
                tmp += random.sample(all_index[i],
                                     self.sample_num_per_class - len(tmp))
                meta_train_index += tmp
            else:
                meta_train_index += random.sample(
                    all_index[i], self.sample_num_per_class)
        random.shuffle(meta_train_index)
        self.meta_train_index = meta_train_index
        return meta_train_index

    def sample_dataset_from_metadataset(self):
        if self.val_index is None:
            self.sample_valid_index()
        self.sample_train_index(self.all_index)

        print("length of sample_index", len(self.meta_train_index))
        print("length of val_index", len(self.val_index))

        train_x = [
            self.meta_data_x[i]
            for i in self.meta_train_index + self.val_index
        ]
        train_y = self.meta_data_y[self.meta_train_index +
                                            self.val_index, :]
        return train_x, train_y


    def dataset_preporocess(self, x_train):
        if self.language == 'ZH':
            print("this is a ZH dataset")
            x_train = clean_zh_text(x_train)
            word_avr = np.mean([len(i) for i in x_train])
            test_num = self.metadata['test_num']
            chi_num_chars_train = int(word_avr * len(x_train) /
                                      CHI_WORD_LENGTH)
            chi_num_chars_test = int(word_avr * test_num / CHI_WORD_LENGTH)

            self.meta_data_feature = {
                'chi_num_chars_train':chi_num_chars_train,
                'chi_num_chars_test':chi_num_chars_test,
                'language':self.language
            }
            self.set_feature_mode()

            if self.feature_mode == 1:
                x_train = list(map(_tokenize_chinese_words, x_train))
        else:
            self.meta_data_feature = {
                 'language':self.language 
            }
            self.set_feature_mode()
            x_train = clean_en_text(x_train)

        return x_train, self.feature_mode

    def set_feature_mode(self):
        if self.meta_data_feature['language'] == 'ZH':
            chi_num_chars_train, chi_num_chars_test = self.meta_data_feature["chi_num_chars_train"],\
                                                      self.meta_data_feature["chi_num_chars_test"]
        cond_word_1 = self.meta_data_feature['language'] == 'EN'
        cond_word_2 = self.meta_data_feature['language'] == 'ZH'\
                      and chi_num_chars_train < 2e5 and chi_num_chars_test < 4e5

        if cond_word_1 or cond_word_2:
            self.feature_mode = 1
        else:
            self.feature_mode = 0
            #self.load_pretrain_emb = False

        print("the feature mode is", self.feature_mode)


    def dataset_postprocess(self, x_train, y_train, model_name):
        if model_name == 'svm':
            self.valid_x = x_train[len(self.meta_train_index):]
            self.valid_y = y_train[len(self.meta_train_index):, :]
            self.x_train = x_train[0:len(self.meta_train_index)]
            self.y_train = y_train[0:len(self.meta_train_index), :]
            self.x_train, self.svm_token = self.vectorize_data(self.x_train)
            #self.data_feature = None

        else:
            x_train, self.word_index, self.num_features, self.tokenizer, self.max_length = self.sequentialize_data(
                x_train, self.feature_mode, tokenizer=self.tokenizer,max_length=self.max_length)

            self.x_train = x_train
            self.y_train = y_train
            print("max_length_training:", self.max_length)
            print("num_featrues_training:", self.num_features)

            self.valid_x = self.x_train[len(self.meta_train_index):]
            self.valid_y = self.y_train[len(self.meta_train_index):, :]

            self.x_train = x_train[:len(self.meta_train_index)]
            self.y_train = y_train[:len(self.meta_train_index), :]


            self.data_feature['num_features'] = self.num_features
            self.data_feature['word_index'] = self.word_index
            self.data_feature['num_class'] = self.num_classes
            self.data_feature['max_length'] = self.max_length
            self.data_feature['input_shape'] = x_train.shape[1:][0] 

    #for svm vectorize data
    def vectorize_data(self, x_train, x_val=None):
        vectorizer = TfidfVectorizer(ngram_range=(1, 1))
        if x_val:
            full_text = x_train + x_val
        else:
            full_text = x_train
        vectorizer.fit(full_text)
        train_vectorized = vectorizer.transform(x_train)
        if x_val:
            val_vectorized = vectorizer.transform(x_val)
            return train_vectorized, val_vectorized, vectorizer
        return train_vectorized, vectorizer

    #Vectorize for cnn
    def sequentialize_data(self,train_contents, feature_mode, val_contents=None,tokenizer=None,max_length=None):
        """Vectorize data into ngram vectors.

        Args:
            train_contents: training instances
            val_contents: validation instances
            y_train: labels of train data.

        Returns:
            sparse ngram vectors of train, valid text inputs.
        """
        if tokenizer is None:
            if feature_mode == 0:
                tokenizer = text.Tokenizer(num_words=MAX_VOCAB_SIZE,
                                           char_level=True,
                                           oov_token="UNK")
            elif feature_mode == 1:
                tokenizer = text.Tokenizer(num_words=MAX_VOCAB_SIZE)
            tokenizer.fit_on_texts(train_contents)
        x_train = tokenizer.texts_to_sequences(train_contents)

        if val_contents:
            x_val = tokenizer.texts_to_sequences(val_contents)

        if max_length == None:
            max_length = len(max(x_train, key=len))
            ave_length = np.mean([len(i) for i in x_train])
            print("max_length_word_training:", max_length)
            print("ave_length_word_training:", ave_length)

        if max_length > MAX_SEQ_LENGTH:
            max_length = MAX_SEQ_LENGTH

        x_train = sequence.pad_sequences(x_train, maxlen=max_length)
        if val_contents:
            x_val = sequence.pad_sequences(x_val, maxlen=max_length)

        word_index = tokenizer.word_index
        num_features = min(len(word_index) + 1, MAX_VOCAB_SIZE)
        print("vacab_word:", len(word_index))
        if val_contents:
            return x_train, x_val, word_index, num_features, tokenizer, max_length
        else:
            return x_train, word_index, num_features, tokenizer, max_length
