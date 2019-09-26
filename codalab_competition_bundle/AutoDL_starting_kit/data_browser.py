# Author: Zhengying LIU
# Date: 3 Nov 2018
"""Visualize examples and labels for given AutoDL dataset.

Usage:
  `python data_browser.py -dataset_dir=/AutoDL_sample_data/miniciao`

Full usage:
  `python data_browser.py -dataset_dir=/AutoDL_sample_data/miniciao -subset=test -num_examples=7`
"""

import os
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# for wav files
import librosa
from playsound import playsound

def _HERE(*args):
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(h, *args))

tf.logging.set_verbosity(tf.logging.INFO)

# STARTING_KIT_DIR = 'autodl/codalab_competition_bundle/AutoDL_starting_kit'
RELATIVE_STARTING_KIT_DIR = './'
STARTING_KIT_DIR = _HERE(RELATIVE_STARTING_KIT_DIR)
INGESTION_DIR = os.path.join(STARTING_KIT_DIR, 'AutoDL_ingestion_program')
SCORING_DIR = os.path.join(STARTING_KIT_DIR, 'AutoDL_scoring_program')
CODE_DIR = os.path.join(STARTING_KIT_DIR, 'AutoDL_sample_code_submission')
for d in [INGESTION_DIR, SCORING_DIR, CODE_DIR]:
  sys.path.append(d)
from dataset import AutoDLDataset # pylint: disable=wrong-import-position, import-error


class DataBrowser(object):
  """A class for visualizing datasets."""

  def __init__(self, dataset_dir):
    self.dataset_dir = os.path.expanduser(dataset_dir) # Expand the tilde `~/`
    self.domain = self.infer_domain()
    self.d_train, self.d_test, self.other_info = self.read_data()

  def read_data(self):
    """Given a dataset directory, read and return training/test set data as
    `AutoDLDataset` objects, along with other infomation.

    Args:
      dataset_dir: a string indicating the absolute or relative path of a
        formatted AutoDL dataset.
    Returns:
      d_train, d_test: 2 'AutoDLDataset' objects, containing training/test data.
      other_info: a dict containing some additional info on the dataset, e.g.
      the metadata on the column names and class names (contained in
        `label_to_index_map`).
    """
    dataset_dir = self.dataset_dir
    files = os.listdir(dataset_dir)
    data_files = [x for x in files if x.endswith('.data')]
    assert len(data_files) == 1
    dataset_name = data_files[0][:-5]
    solution_files = [x for x in files if x.endswith('.solution')]
    with_solution = None # With or without solution (i.e. training or test)
    if len(solution_files) == 1:
      solution_dataset_name = solution_files[0][:-9]
      if solution_dataset_name == dataset_name:
        with_solution = True
      else:
        raise ValueError("Wrong dataset name. Should be {} but got {}."\
                         .format(dataset_name, solution_dataset_name))
    elif not solution_files:
      with_solution = False
    else:
      return ValueError("Multiple solution files found:" +\
                        " {}".format(solution_files))
    d_train = AutoDLDataset(os.path.join(dataset_dir, dataset_name + '.data',
                                         "train"))
    d_test = AutoDLDataset(os.path.join(dataset_dir, dataset_name + '.data',
                                        "test"))
    other_info = {}
    other_info['dataset_name'] = dataset_name
    other_info['with_solution'] = with_solution
    label_to_index_map = d_train.get_metadata().get_label_to_index_map()
    if label_to_index_map:
      classes_list = [None] * len(label_to_index_map)
      for label in label_to_index_map:
        index = label_to_index_map[label]
        classes_list[index] = label
      other_info['classes_list'] = classes_list
    else:
      tf.logging.info("No label_to_index_map found in metadata. Labels will "
                      "only be represented by integers.")
    self.d_train, self.d_test, self.other_info = d_train, d_test, other_info
    if with_solution:
      solution_path = os.path.join(dataset_dir, solution_files[0])
      self.other_info['Y_test'] = np.loadtxt(solution_path)
    return d_train, d_test, other_info

  def infer_domain(self):
    """Infer the domain from the shape of the 4-D tensor."""
    d_train, _, _ = self.read_data()
    metadata = d_train.get_metadata()
    row_count, col_count = metadata.get_matrix_size(0)
    sequence_size = metadata.get_sequence_size()
    domain = None
    if sequence_size == 1:
      if row_count == 1 or col_count == 1:
        domain = "tabular"
      else:
        domain = "image"
    else:
      if row_count == 1 and col_count == 1:
        domain = "speech"
      elif row_count == 1 or col_count == 1:
        domain = "text"
      else:
        domain = "video"
    self.domain = domain
    tf.logging.info("The inferred domain of the dataset is: {}.".format(domain))
    return domain

  @classmethod
  def show_video(cls, tensor_4d, interval=80, label_confidence_pairs=None):
    """Visualize a video represented by `tensor_4d` using `interval` ms.
    This means that frames per second (fps) is equal to 1000/`interval`.
    """
    fig, _ = plt.subplots()
    image = np.squeeze(tensor_4d[0])
    screen = plt.imshow(image)
    def init():  # only required for blitting to give a clean slate.
      """Initialize the first screen"""
      screen.set_data(np.empty(image.shape))
      return screen,
    def animate(i):
      """Some kind of hooks for `animation.FuncAnimation` I think."""
      if i < len(tensor_4d):
        image = np.squeeze(tensor_4d[i])
        screen.set_data(image)
      return screen,
    _ = animation.FuncAnimation(
        fig, animate, init_func=init, interval=interval,
        blit=True, save_count=50, repeat=False) # interval=40 because 25fps
    plt.title('Labels: ' + str(label_confidence_pairs))
    plt.show()
    return plt

  @classmethod
  def show_image(cls, tensor_4d, label_confidence_pairs=None):
    """Visualize a image represented by `tensor_4d` in RGB or grayscale."""
    num_channels = tensor_4d.shape[-1]
    image = np.squeeze(tensor_4d[0])
    # If the entries are float but in [0,255]
    if not np.issubdtype(image.dtype, np.integer) and np.max(image) > 100:
      image = image / 256
    if num_channels == 1:
      plt.imshow(image, cmap='gray')
    else:
      # if not num_channels == 3:
      #   raise ValueError("Expected num_channels = 3 but got {} instead."\
      #                    .format(num_channels))
      plt.imshow(image)
    plt.title('Labels: ' + str(label_confidence_pairs))
    plt.show()
    return plt

  @classmethod
  def show_speech(cls, tensor_4d, label_confidence_pairs=None):
      data = np.squeeze(tensor_4d)
      print('Playing audio...')
      DataBrowser.play_sound(data)
      print('Done. Now opening labels window.')
      plt.title('Labels: ' + str(label_confidence_pairs))
      plt.show()
      return plt

  @classmethod
  def play_sound(cls, data, nchannels=1, sampwidth=2,
                 framerate=16000, comptype='NONE', compname='not compressed'):
    # Create a tmp file
    tmp_filepath = '/tmp/sound.wav'
    # Write data
    librosa.output.write_wav(tmp_filepath, data, framerate)
    # PLAY
    playsound(tmp_filepath)
    # Delete the tmp file
    os.system('rm ' + tmp_filepath)

  @classmethod
  def get_nth_element(cls, autodl_dataset, num):
    """Get n-th element in `autodl_dataset` using iterator."""
    dataset = autodl_dataset.get_dataset()
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
      for _ in range(num+1):
        try:
          tensor_4d, labels = sess.run(next_element)
        except tf.errors.OutOfRangeError:
          tf.logging.info("Reached the end of dataset. " +
                          "Return the last example.")
          break
    return tensor_4d, labels

  @property
  def show(self):
    """Return corresponding show method according to inferred domain."""
    domain = self.domain
    if domain == 'image':
      return DataBrowser.show_image
    elif domain == 'video':
      return DataBrowser.show_video
    elif domain == 'speech':
      return DataBrowser.show_speech
    else:
      raise NotImplementedError("Show method not implemented for domain: " +\
                                 "{}".format(domain))

  def show_an_example(self, default_max_range=1000, subset='train'):
    """Visualize an example whose index is randomly chosen in the interval
    [0, `max_range`).
    """
    if subset == 'train':
      d = self.d_train
    else:
      d = self.d_test
    max_range = min(d.metadata_.size(), default_max_range)
    idx = np.random.randint(max_range)
    tensor_4d, labels = DataBrowser.get_nth_element(d, idx)
    if subset != 'train':
      if self.other_info['with_solution']:
        labels = self.other_info['Y_test'][idx]
      else:
        tf.logging.info("No solution file found for test set. " +
                        "Only showing examples (without labels).")
    if 'classes_list' in self.other_info:
      c_l = self.other_info['classes_list']
      label_conf_pairs = {c_l[idx]: c for idx, c in enumerate(labels) if c != 0}
    else:
      label_conf_pairs = {idx: c for idx, c in enumerate(labels) if c != 0}
    self.show(tensor_4d, label_confidence_pairs=label_conf_pairs)


def show_examples(dataset_dir, num_examples=5, subset='train'):
      print("Start visualizing process for dataset: {}...".format(dataset_dir))
      data_browser = DataBrowser(dataset_dir)
      num_examples = min(10, int(num_examples))
      for i in range(num_examples):
        print("Visualizing example {}.".format(i+1) +
              " Close the corresponding window to continue...")
        data_browser.show_an_example(subset=subset)

def get_tensor_shape(dataset_dir, bundle_index=0):
    data_browser = DataBrowser(dataset_dir)
    metadata = data_browser.d_train.get_metadata()
    return metadata.get_tensor_shape(bundle_index)

def get_size(dataset_dir):
    data_browser = DataBrowser(dataset_dir)
    num_train = data_browser.d_train.get_metadata().size()
    num_test = data_browser.d_test.get_metadata().size()
    return num_train, num_test

def get_output_dim(dataset_dir):
    data_browser = DataBrowser(dataset_dir)
    output_dim = data_browser.d_train.get_metadata().get_output_size()
    return output_dim

def main(*argv):
  """Do you really need a docstring?"""
  # Actually here dataset_dir should be dataset_dir since dataset_dir/ is the folder
  # that contains all datasets but dataset_dir is the folder that contains the
  # content of one single dataset
  default_dataset_dir = _HERE('AutoDL_sample_data/miniciao')
  tf.flags.DEFINE_string('dataset_dir', default_dataset_dir,
                         "Path to dataset.")
  tf.flags.DEFINE_string('subset', 'train',
                         "Can be 'train' or 'test'.")
  tf.flags.DEFINE_integer('num_examples', 5,
                         "Number of examples to show.")

  FLAGS = tf.flags.FLAGS
  del argv
  dataset_dir = FLAGS.dataset_dir
  subset = FLAGS.subset
  num_examples = FLAGS.num_examples
  num_train, num_test = get_size(dataset_dir)
  print('num_train: {}\nnum_test: {}'.format(num_train, num_test))
  print('tensor shape: {}'.format(get_tensor_shape(dataset_dir)))
  print('output_dim: {}'.format(get_output_dim(dataset_dir)))
  show_examples(dataset_dir, num_examples=num_examples, subset=subset)


if __name__ == '__main__':
  main()
