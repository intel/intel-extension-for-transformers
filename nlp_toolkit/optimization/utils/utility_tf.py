from collections import OrderedDict, UserDict


TMPPATH = "tmp"
class TFDataloader(object):
    """
       Args:
           dataset (string): Dataset
    """

    def __init__(self, dataset, batch_size=None):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        for inputs, labels in self.dataset:
            if isinstance(inputs, dict) or isinstance(inputs, OrderedDict) \
                  or isinstance(inputs, UserDict):
                for name in inputs.keys():
                    inputs[name] = inputs[name].numpy()
            elif isinstance(inputs, list) or isinstance(inputs, tuple):
                inputs = [input.numpy() for input in inputs]
            else:
                inputs = inputs.numpy()

            if isinstance(labels, dict) or isinstance(labels, OrderedDict) \
                  or isinstance(labels, UserDict):
                for name in labels.keys():
                    labels[name] = labels[name].numpy()
            elif isinstance(labels, list) or isinstance(labels, tuple):
                labels = [label.numpy() for label in labels]
            else:
                labels = labels.numpy()
            yield inputs, labels