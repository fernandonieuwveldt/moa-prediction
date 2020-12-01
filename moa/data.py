"""
data loading
"""

import pandas
import tensorflow as tf


class TFDataMapper:
    """Transform pandas data frame to tensorflow data set
    """
    def __init__(self):
        pass

    def data_frame_to_dataset(self, data_frame_features=None, data_frame_labels=None):
        """Transform from data frame to data_set

        Args:
            data_frame_features: Pandas DataFrame
            data_frame_labels: Pandas DataFrame
        """
        if data_frame_labels is not None:
            return tf.data.Dataset.from_tensor_slices((dict(data_frame_features), data_frame_labels.values)).shuffle(1000)
        else:
            return tf.data.Dataset.from_tensor_slices(dict(data_frame_features))

    def transform(self, data_frame_features=None, data_frame_labels=None):
        """transform data

        Args:
            data_frame_features: Pandas DataFrame
            data_frame_labels: Pandas DataFrame
        """
        return self.data_frame_to_dataset(data_frame_features, data_frame_labels)
