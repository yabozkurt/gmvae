import numpy as np

def one_hot(labels, depth):
    one_hot = np.zeros((len(labels), depth))
    labels = np.reshape(labels, (len(labels, )))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


class Trainset(object):
    def __init__(self):
        self.data = None
        self.labels = None
        self.current_index = 0

    def next_batch(self,size):
        assert size < len(self.data), 'train set is too small!'
        batch = self.data[self.current_index:self.current_index+size]
        self.current_index += size
        self.current_index = self.current_index % len(self.data)
        return batch

class Testset(object):
    def __init__(self):
        self.data = None
        self.labels = None

class Dataset(object):
    def __init__(self, k):
        self.k = k
        self.train = Trainset()
        self.test = Testset()

    def setTrainData(self, data, labels = None):
        self.train.data = data
        if type(labels) is np.ndarray:
            self.train.labels = one_hot(labels, self.k)

    def setTestData(self, data, labels = None):
        self.test.data = data
        if type(labels) is np.ndarray:
            self.test.labels = one_hot(labels, self.k)

 
def load_and_mix_data_nolabel(path=None, k=3, data=None, test_ratio=0, randomize=True):
    if data is None:
        data = np.load(path)
    
    m = data.shape[0]

      
    if randomize:
        indices = np.random.permutation(m)
        data = data[indices,:]
        
    train_ratio = 1 - test_ratio
    train_data = data[:int(m * train_ratio)]
    test_data = data[int(m * train_ratio):]
    dataset = Dataset(k)
    dataset.setTrainData(train_data)
    dataset.setTestData(test_data)
    return dataset
