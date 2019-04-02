import numpy as np
import keras
import random
from keras.datasets import mnist
from keras import backend as K

class DataSource(object): # 这好比 一个接口
    def __init__(self):
        raise NotImplementedError()
    def partitioned_by_rows(self, num_workers, test_reserve=.3):
        raise NotImplementedError()
    def sample_single_non_iid(self, weight=None):
        raise NotImplementedError()


class Mnist(DataSource):

    IID = False # 是否 独立同分布
    MAX_NUM_CLASSES_PER_CLIENT = 10 # 每个客户端 持有的 最大（标签）类别数
    
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data() # 获取Mnist数据集
        self.x = np.concatenate([x_train, x_test]).astype('float') # 拼接 变成整个数据集的特征集 
        self.y = np.concatenate([y_train, y_test]) # 整个数据集的 标签集
        n = self.x.shape[0] # 训练集 总数据大小
        idx = np.arange(n) # 生成1个1~n的序列
        np.random.shuffle(idx) # 1～n的随机排列
        self.x = self.x[idx]  # n * 28 * 28  打乱原数据顺序
        self.y = self.y[idx]  # n * 1
        data_split = (0.6, 0.3, 0.1) # 数据分割率
        num_train = int(n * data_split[0]) # 训练集大小
        num_test = int(n * data_split[1]) # 测试集大小
        self.x_train = self.x[0:num_train] # 依次 取出 各个部分的数据
        self.x_test = self.x[num_train:num_train + num_test]
        self.x_valid = self.x[num_train + num_test:]
        self.y_train = self.y[0:num_train]
        self.y_test = self.y[num_train:num_train + num_test]
        self.y_valid = self.y[num_train + num_test:]
        self.classes = np.unique(self.y) # 标签所有的取值
    
    def gen_dummy_non_iid_weights(self): # 生成非 独立同分布 的权重
        self.classes = np.array(range(10)) # 0～9的数组
        print(self.classes)
        num_classes_this_client = 3 #random.randint(1, Mnist.MAX_NUM_CLASSES_PER_CLIENT + 1) # 随机生成 该client的持有 类别总数
        print(num_classes_this_client)
        classes_this_client = random.sample(self.classes.tolist(), num_classes_this_client) # 随机选出 对应的类别
        print(classes_this_client)
        w = np.array([.5, .3, .2]) # 随机生成 每个类别 所占的权重
        print(w)
        weights = np.array([0.] * self.classes.shape[0]) # 10 个 0. 
        for i in range(len(classes_this_client)):
            weights[classes_this_client[i]] = w[i]
        weights /= np.sum(weights) # 除以平均 获得各个类别 的概率
        return weights.tolist() 


    # assuming client server already agreed on data format
    def post_process(self, xi, yi): # 数据处理
        if K.image_data_format() == 'channels_first': # 通道的位置
            xi = xi.reshape(1, xi.shape[0], xi.shape[1])
        else:
            xi = xi.reshape(xi.shape[0], xi.shape[1], 1)

        y_vec = keras.utils.to_categorical(yi, self.classes.shape[0]) # 标签 独热码表示
        return xi / 255., y_vec # 归一化

    # split evenly into exact num_workers chunks, with test_reserve globally
    def partitioned_by_rows(self, num_workers, test_reserve=.3):
        n_test = int(self.x.shape[0] * test_reserve)
        n_train = self.x.shape[0] - n_test
        nums = [n_train // num_workers] * num_workers
        nums[-1] += n_train % num_workers
        idxs = np.array([np.random.choice(np.arange(n_train), num, replace=False) for num in nums])
        return {
            # (size_partition * 28 * 28, size_partition * 1) * num_partitions
            "train": [post_process(self.x[idx], self.y[idx]) for idx in idxs],
            # (n_test * 28 * 28, n_test * 1)
            "test": post_process(self.x[np.arange(n_train, n_train + n_test)], self.y[np.arange(n_train, n_train + n_test)])
        }

    # Generate one sample from all available data, *with replacement*.
    # This is to simulate date generation on a client.
    # weight: [probablity of classes]
    # returns: 28 * 28, 1
    def sample_single_non_iid(self, x, y, weight=None):
        # first pick class, then pick a datapoint at random
        chosen_class = np.random.choice(self.classes, p=weight)
        candidates_idx = np.array([i for i in range(y.shape[0]) if y[i] == chosen_class])
        idx = np.random.choice(candidates_idx)
        return self.post_process(x[idx], y[idx])

    
    # generate t, t, v dataset given distribution and split
    def fake_non_iid_data(self, min_train=100, max_train=1000, data_split=(.6,.3,.1)):        
        # my_class_distr = np.array([np.random.random() for _ in range(self.classes.shape[0])])
        # my_class_distr /= np.sum(my_class_distr)
        my_class_distr = self.gen_dummy_non_iid_weights()

        train_size = random.randint(min_train, max_train)
        test_size = int(train_size / data_split[0] * data_split[1])
        valid_size = int(train_size / data_split[0] * data_split[2])

        train_set = [self.sample_single_non_iid(self.x_train, self.y_train, my_class_distr) for _ in range(train_size)]
        test_set = [self.sample_single_non_iid(self.x_test, self.y_test, my_class_distr) for _ in range(test_size)]
        valid_set = [self.sample_single_non_iid(self.x_valid, self.y_valid, my_class_distr) for _ in range(valid_size)]
        print("done generating client data")

        return ((train_set, test_set, valid_set), my_class_distr)

    def fake_iid_data(self, min_train=100, max_train=1000, data_split=(.6,.3,.1)):        
        # my_class_distr = np.array([np.random.random() for _ in range(self.classes.shape[0])])
        # my_class_distr /= np.sum(my_class_distr)
        my_class_distr = [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1]

        train_size = random.randint(min_train, max_train)
        test_size = int(train_size / data_split[0] * data_split[1])
        valid_size = int(train_size / data_split[0] * data_split[2])

        train_set = [self.sample_single_non_iid(self.x_train, self.y_train, my_class_distr) for _ in range(train_size)]
        test_set = [self.sample_single_non_iid(self.x_test, self.y_test, my_class_distr) for _ in range(test_size)]
        valid_set = [self.sample_single_non_iid(self.x_valid, self.y_valid, my_class_distr) for _ in range(valid_size)]
        print("done generating client data")

        return ((train_set, test_set, valid_set), my_class_distr)

    def get_server_test(self, test_size):
        my_class_distr = [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1]
        test_set = [self.sample_single_non_iid(self.x_test, self.y_test, my_class_distr) for _ in range(test_size)]
        print("done generating server data")

        return test_set


if __name__ == "__main__":
    m = Mnist()
    # res = m.partitioned_by_rows(9)
    # print(res["test"][1].shape)
    for _ in range(10):
        print(m.gen_dummy_non_iid_weights())

