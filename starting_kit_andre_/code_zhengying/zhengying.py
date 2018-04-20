import tensorflow as tf
import dataset

dataset_name = "mnist"
data_set = dataset.AutoDLDataset(dataset_name)
data_set.init()
print(dir(data_set))
