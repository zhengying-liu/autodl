from dataset import AutoDLDataset
import tensorflow as tf


def news_baseline_test():

    dataset = AutoDLDataset('./tweet.data/train/')
    dataset.init()
    iterator = dataset.get_dataset().make_one_shot_iterator()
    next_element = iterator.get_next()

    # features, labels = next_element
    #
    # features = features.eval()
    # labels = labels.eval()
    #
    # print next_element
    data = []
    sess = tf.Session()
    for idx in range(10):
        print("Example " + str(idx))
        data.append(sess.run(next_element))

    for each_data in data:

        print each_data


if __name__ == '__main__':

    news_baseline_test()

