import tensorflow as tf
from tensorflow import keras
import numpy as np
import csv
from score import autodl_bac as bac
import pickle
import FileOperator as fo


# get confusion matrix
def confusion_matrix(solution, prediction):

    label_num = solution.shape[1]

    prediction = prediction.tolist()
    solution = solution.tolist()

    conf_matrix = np.zeros((label_num, label_num))

    for i in range(len(solution)):
        row_index = solution[i].index(1)
        col_index = prediction[i].index(max(prediction[i]))
        conf_matrix[row_index, col_index] += 1

    return conf_matrix


def test_confusion_matrix():

    source_file = './20news/tweet_sol_pred_par.pkl'

    f = open(source_file, 'rb')
    solution, prediction = pickle.load(f)
    f.close()

    conf_matrix = confusion_matrix(solution, prediction)

    return


def text_classification_test():

    train_file = './imdb/train_data.pkl'
    f = open(train_file, 'rb')
    train_data = pickle.load(f)
    train_labels = pickle.load(f)
    f.close()

    test_file = './imdb/test_data.pkl'
    f = open(test_file, 'rb')
    test_data = pickle.load(f)
    test_labels = pickle.load(f)
    f.close()

    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=0,
                                                            padding='post',
                                                            maxlen=256)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=0,
                                                           padding='post',
                                                           maxlen=256)

    vocab_size = 10000

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.summary()

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]

    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=40,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        shuffle=True,
                        verbose=1)

    results = model.evaluate(test_data, test_labels)

    print(results)


def test_data_load():

    imdb = keras.datasets.imdb

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    print 'save train'
    train_file = './imdb/train_data.pkl'
    f = open(train_file, 'wb')
    pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_labels, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    print 'save test'
    test_file = './imdb/test_data.pkl'
    f = open(test_file, 'wb')
    pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_labels, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    return


def csv_data_reader(data_file='', label_file=''):

    data_list = []
    with open(data_file) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            data_list.append(row)
    data_list = [[float(x) for x in row] for row in data_list]

    label_list = []
    with open(label_file) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            label_list.append(row)
    label_list = [[int(x) for x in row] for row in label_list]
    # temp_list = []
    # for i in range(len(label_list)):
    #     temp_list.append(label_list[i][0])
    # label_list = temp_list

    return data_list, label_list


def take_num(l):
    return l[0]


def baseline_mlp_test():


    # tweets
    # train_data_file = './tweets/train_text.csv'
    # train_label_file = './tweets/train_label.csv'
    #
    # test_data_file = './tweets/test_text.csv'
    # test_label_file = './tweets/test_label.csv'

    # microblog
    # train_data_file = './microblog/train_text.csv'
    # train_label_file = './microblog/train_label.csv'

    # test_data_file = './microblog/test_text.csv'
    # test_label_file = './microblog/test_label.csv'

    # 20newsgroup
    train_data_file = './20news/shuffled_train_feature.csv'
    train_label_file = './20news/shuffled_train_one_hot_label.csv'

    test_data_file = './20news/test_feature.csv'
    test_label_file = './20news/test_one_hot_label.csv'

    print 'read train data...'
    print train_data_file
    print train_label_file
    train_data, train_label = csv_data_reader(data_file=train_data_file, label_file=train_label_file)
    print 'read test data...'
    print test_data_file
    print test_label_file
    test_data, test_label = csv_data_reader(data_file=test_data_file, label_file=test_label_file)

    len_limit = 400
    max_len = 0
    ave_len = 0.0
    i = 0
    for each_data in train_data:
        i += 1
        ave_len = (float(i-1) / i) * ave_len + float(len(each_data)) / i
        if len(each_data) > max_len:
            max_len = len(each_data)
    for each_data in test_data:
        i += 1
        ave_len = (float(i - 1) / i) * ave_len + float(len(each_data)) / i
        if len(each_data) > max_len:
            max_len = len(each_data)

    for i in range(len(train_data)):
        if len(train_data[i]) > len_limit:
            train_data[i] = train_data[i][:len_limit]
    for i in range(len(test_data)):
        if len(test_data[i]) > len_limit:
            test_data[i] = test_data[i][:len_limit]

    num_words = len(train_data)

    print 'max words len: ', max_len
    print 'ave word len: ', ave_len

    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=0,
                                                            padding='post',
                                                            maxlen=len_limit)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=0,
                                                           padding='post',
                                                           maxlen=len_limit)

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)

    valid_len = int(num_words * 0.1)
    print 'validation len: ', valid_len

    vocab_size = 2000

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 128))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    # model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    # model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    # model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(20, activation=tf.nn.softmax))

    model.summary()

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.005),
                  loss=tf.losses.softmax_cross_entropy,
                  metrics=['accuracy'])

    x_val = train_data[:valid_len]
    partial_x_train = train_data[valid_len:]
    print partial_x_train.shape

    y_val = train_label[:valid_len]
    partial_y_train = train_label[valid_len:]
    print partial_y_train.shape

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=200,
                        batch_size=512,
                        # validation_data=(x_val, y_val),
                        validation_data=(test_data, test_label),
                        shuffle=True,
                        verbose=1)

    print 'testing...'
    predictions = model.predict(test_data)
    # results = model.evaluate(test_data, test_label)
    results = bac(test_label, predictions)
    print 'test balanced acc: ', results
    log_buff = []
    log_buff.append('test balanced acc: ' + str(results))

    print 'saving prediction'
    f = open('./20news/tweet_sol_pred_par.pkl', 'wb')
    pickle.dump((test_label, predictions), f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    print 'saving result...'
    fo.FileWriter('./20news/tweet_result.txt', log_buff, style='w')

    return


def baseline_priv_mlp_test():


    # tweets
    # train_data_file = './tweets/train_text.csv'
    # train_label_file = './tweets/train_label.csv'
    #
    # test_data_file = './tweets/test_text.csv'
    # test_label_file = './tweets/test_label.csv'

    # microblog
    # train_data_file = './microblog/train_text.csv'
    # train_label_file = './microblog/train_label.csv'

    # test_data_file = './microblog/test_text.csv'
    # test_label_file = './microblog/test_label.csv'

    # 20newsgroup
    train_data_file = './20news/shuffled_train_feature.csv'
    train_label_file = './20news/shuffled_train_hyper_one_hot_label.csv'

    test_data_file = './20news/test_feature.csv'
    test_label_file = './20news/test_hyper_one_hot_label.csv'

    print 'read train data...'
    print train_data_file
    print train_label_file
    train_data, train_label = csv_data_reader(data_file=train_data_file, label_file=train_label_file)
    print 'read test data...'
    print test_data_file
    print test_label_file
    test_data, test_label = csv_data_reader(data_file=test_data_file, label_file=test_label_file)

    len_limit = 400
    max_len = 0
    ave_len = 0.0
    i = 0
    for each_data in train_data:
        i += 1
        ave_len = (float(i-1) / i) * ave_len + float(len(each_data)) / i
        if len(each_data) > max_len:
            max_len = len(each_data)
    for each_data in test_data:
        i += 1
        ave_len = (float(i - 1) / i) * ave_len + float(len(each_data)) / i
        if len(each_data) > max_len:
            max_len = len(each_data)

    for i in range(len(train_data)):
        if len(train_data[i]) > len_limit:
            train_data[i] = train_data[i][:len_limit]
    for i in range(len(test_data)):
        if len(test_data[i]) > len_limit:
            test_data[i] = test_data[i][:len_limit]

    num_words = len(train_data)

    print 'max words len: ', max_len
    print 'ave word len: ', ave_len

    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=0,
                                                            padding='post',
                                                            maxlen=len_limit)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=0,
                                                           padding='post',
                                                           maxlen=len_limit)

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)

    valid_len = int(num_words * 0.1)
    print 'validation len: ', valid_len

    vocab_size = 2000

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 128))
    model.add(keras.layers.GlobalAveragePooling1D())
    # model.add(keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    # model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    # model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(7, activation=tf.nn.softmax))

    model.summary()

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.005),
                  loss=tf.losses.softmax_cross_entropy,
                  metrics=['accuracy'])

    x_val = train_data[:valid_len]
    partial_x_train = train_data[valid_len:]
    print partial_x_train.shape

    y_val = train_label[:valid_len]
    partial_y_train = train_label[valid_len:]
    print partial_y_train.shape

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=100,
                        batch_size=512,
                        # validation_data=(x_val, y_val),
                        validation_data=(test_data, test_label),
                        shuffle=True,
                        verbose=1)

    print 'testing...'
    predictions = model.predict(test_data)
    # results = model.evaluate(test_data, test_label)
    results = bac(test_label, predictions)
    print 'test balanced acc: ', results
    log_buff = []
    log_buff.append('test balanced acc: ' + str(results))

    print 'saving prediction'
    f = open('./20news/tsunami_sol_pred_par.pkl', 'wb')
    pickle.dump((test_label, predictions), f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    print 'saving results...'
    fo.FileWriter('./20news/tsunami_result.txt', log_buff, style='w')

    return


def data_trans_2d(lists, shape=(3, 3)):

    data_3d = []
    i = 0
    for each_data in lists:
        if i % 1000 == 0:
            print i
        data_2d = []
        for each_index in each_data:
            data_1d = [0] * shape[1]
            data_1d[each_index] += 1
            data_2d.append(data_1d)
        data_3d.append(data_2d)
        i += 1

    data_3d = tf.expand_dims(np.array(data_3d), 3)
    with tf.Session() as sess:
        data_3d = data_3d.eval()

    return data_3d


class MyExpandLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MyExpandLayer, self).__init__()
        return

    def build(self, input_shape):
        return

    def call(self, input):

        return tf.expand_dims(input, 3)


class MyReshapeLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MyReshapeLayer, self).__init__()
        return

    def build(self, input_shape):
        return

    def call(self, input):

        output_size = 1
        for s in input.shape:
            output_size *= s

        print 'my reshape layer: ', output_size

        return tf.reshape(input, (-1, output_size))


def baseline_cnn_test():

    train_data_file = './tweets/train_text.csv'
    train_label_file = './tweets/train_label.csv'

    test_data_file = './tweets/test_text.csv'
    test_label_file = './tweets/test_label.csv'

    print 'read train data...'
    train_data, train_label = csv_data_reader(data_file=train_data_file, label_file=train_label_file)
    print 'read test data...'
    test_data, test_label = csv_data_reader(data_file=test_data_file, label_file=test_label_file)

    max_index = 501

    print 'process train data...'
    train_len = len(train_data)
    for i in range(len(train_data)):
        if (float(i) / train_len) % 0.1 == 0:
            print 'process ', (float(i) / train_len) * 100, '%'
        for j in range(len(train_data[i])):
            if train_data[i][j] > 499:
                train_data[i][j] = 500
    print 'process test data...'
    test_len = len(test_data)
    for i in range(len(test_data)):
        if (float(i) / test_len) % 0.1 == 0:
            print 'process ', (float(i) / test_len) * 100, '%'
        for j in range(len(test_data[i])):
            if test_data[i][j] > 499:
                test_data[i][j] = 500

    max_len = 0
    for each_data in train_data:
        if len(each_data) > max_len:
            max_len = len(each_data)
    for each_data in test_data:
        if len(each_data) > max_len:
            max_len = len(each_data)

    num_words = len(train_data)

    print 'max words len: ', max_len

    input_shape = (max_len, max_index, 1)

    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=0,
                                                            padding='post',
                                                            maxlen=max_len)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=0,
                                                           padding='post',
                                                           maxlen=max_len)

    train_data = data_trans_2d(train_data, shape=input_shape)
    test_data = data_trans_2d(test_data, shape=input_shape)

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)

    valid_len = int(num_words * 0.1)
    print 'validation len: ', valid_len

    # input_shape = (2, 30)
    # dense_len = 256 * int(int(int(input_shape[0]/2)/2)/2) * int(int(int(input_shape[1]/2)/2)/2)
    # print 'dense len: ', dense_len

    # vocab_size = 7000

    model = keras.Sequential()
    # model.add(keras.layers.Embedding(vocab_size, 16))
    # model.add(MyExpandLayer())
    model.add(keras.layers.Conv2D(256, kernel_size=(2, 2), strides=(1, 1), activation='relu', input_shape=input_shape, padding='same'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(keras.layers.Conv2D(256, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(keras.layers.Conv2D(256, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(256, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(256, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(256, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(keras.layers.Flatten())
    # model.add(MyReshapeLayer())
    model.add(keras.layers.Dense(1024, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1024, activation=tf.nn.relu))
    model.add(keras.layers.Dense(3, activation=tf.nn.softmax))

    model.summary()

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    x_val = train_data[:valid_len]
    partial_x_train = train_data[valid_len:]

    y_val = train_label[:valid_len]
    partial_y_train = train_label[valid_len:]

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=35,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1)

    print 'testing...'
    results = model.evaluate(test_data, test_label)

    print results

    return


def news_data_baseline():

    test_data_file = './data/20newsgroup_data.data/test/sample-tweet.tfrecord'

    with tf.Session() as sess:

        filename_queue = tf.train.string_input_producer([test_data_file], num_epochs=1)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

    return


if __name__ == '__main__':

    # this is the training function of public text dataset
    baseline_mlp_test()
    # this is the training function of private text dataset
    baseline_priv_mlp_test()
