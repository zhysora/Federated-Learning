import numpy as np
import keras
import random
import time
import json
import pickle
import codecs
from keras.models import model_from_json
from socketIO_client import SocketIO, LoggingNamespace
from fl_server import obj_to_pickle_string, pickle_string_to_obj

import datasource
import threading

class LocalModel(object): # 局部模型
    def __init__(self, model_config, data_collected): # self, 模型配置， 本地数据
        # model_config: 模型配置
            # 'model': self.global_model.model.to_json(),
            # 'model_id'
            # 'min_train_size'
            # 'data_split': (0.6, 0.3, 0.1), # train, test, valid
            # 'epoch_per_round'
            # 'batch_size'
        self.model_config = model_config

        self.model = model_from_json(model_config['model_json']) # 获取传过来的全局模型
        # the weights will be initialized on first pull from server

        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy']) # 模型编译

        train_data, test_data, valid_data = data_collected # 收集到的训练集、 测试集、 验证集 数据
        self.x_train = np.array([tup[0] for tup in train_data])
        self.y_train = np.array([tup[1] for tup in train_data]).reshape((-1, 10))
        self.x_test = np.array([tup[0] for tup in test_data])
        self.y_test = np.array([tup[1] for tup in test_data]).reshape((-1, 10))
        self.x_valid = np.array([tup[0] for tup in valid_data])
        self.y_valid = np.array([tup[1] for tup in valid_data]).reshape((-1, 10))

    def get_weights(self): # 返回模型权重
        return self.model.get_weights()

    def set_weights(self, new_weights): # 设定为新的权重
        self.model.set_weights(new_weights)

    # return final weights, train loss, train accuracy
    def train_one_round(self): # 跑一轮
        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy']) # 模型编译

        self.model.fit(self.x_train, self.y_train,
                  epochs=self.model_config['epoch_per_round'],
                  batch_size=self.model_config['batch_size'],
                  verbose=1) # 训练一轮

        score = self.model.evaluate(self.x_train, self.y_train, verbose=0) # 获取评估值
        print('Train loss:', score[0]) # 打印 评估值
        print('Train accuracy:', score[1])
        return self.model.get_weights(), score[0], score[1] # 返回 新的权重， loss, acc

    def validate(self): # 查看在验证集上的表现
        score = self.model.evaluate(self.x_valid, self.y_valid, verbose=0)
        print('Validate loss:', score[0])
        print('Validate accuracy:', score[1])
        return score # loss, acc

    def evaluate(self): # 查看在测试集上的表现
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score # loss, acc


# A federated client is a process that can go to sleep / wake up intermittently
# it learns the global model by communication with the server;
# it contributes to the global model by sending its local gradients.

class FederatedClient(object):
    ##### Client Config
    DATA_MODE = True # true 代表 Non-IDD, false 代表 IDD
    SLEEP_MODE = False # true 代表会随机sleep, false 代表不会

    MAX_DATASET_SIZE_KEPT = 1200 # 最大数据集上限

    def __init__(self, server_host, server_port, datasource):
        self.local_model = None
        self.datasource = datasource() # 获取数据集

        self.sio = SocketIO(server_host, server_port, LoggingNamespace) # SocketIO 服务 客户端
        self.register_handles() # 运行各个 响应机制
        print("sent wakeup")
        self.sio.emit('client_wake_up') # 向 服务端 发送 事件 client_wake_up; 服务端 会回复 init 事件
        self.sio.wait() # 等待

    
    ########## Socket Event Handler ##########
    def on_init(self, *args): # 处理 init 事件
        model_config = args[0] # 获取模型 配置
        print('on init', model_config)
        print('preparing local data based on server model_config')
        # ([(Xi, Yi)], [], []) = train, test, valid
        if FederatedClient.DATA_MODE:
            fake_data, my_class_distr = self.datasource.fake_non_iid_data( # 从数据中心 获取 分配到的数据
                min_train=model_config['min_train_size'],
                max_train=FederatedClient.MAX_DATASET_SIZE_KEPT,
                data_split=model_config['data_split']
            )
            self.local_model = LocalModel(model_config, fake_data) # 用模型配置， 与分配到的局部数据 生成局部模型
        else:
            fake_data, my_class_distr = self.datasource.fake_iid_data( # 从数据中心 获取 分配到的数据
                min_train=model_config['min_train_size'],
                max_train=FederatedClient.MAX_DATASET_SIZE_KEPT,
                data_split=model_config['data_split']
            )
            self.local_model = LocalModel(model_config, fake_data) # 用模型配置， 与分配到的局部数据 生成局部模型

        # ready to be dispatched for training
        self.sio.emit('client_ready', { # 向服务端发送 client_ready 事件
                'train_size': self.local_model.x_train.shape[0],
                'class_distr': my_class_distr  # for debugging, not needed in practice
            })


    def register_handles(self):
        ########## Socket IO messaging ##########
        def on_connect():
            print('connect')

        def on_disconnect():
            print('disconnect')

        def on_reconnect():
            print('reconnect')

        def on_request_update(*args): # 收到 request_update
            req = args[0] # 获得数据 包含当前模型权重
            # req:
            #     'model_id'
            #     'round_number'
            #     'current_weights'
            #     'weights_format'
            #     'run_validation'
            print("update requested")
            if FederatedClient.SLEEP_MODE:
                self.intermittently_sleep(p = 1., low = 10, high = 100) # randowm sleep

            if req['weights_format'] == 'pickle': # 解码 获得模型权重
                weights = pickle_string_to_obj(req['current_weights'])

            self.local_model.set_weights(weights) # 设定权重
            my_weights, train_loss, train_accuracy = self.local_model.train_one_round() # 获得局部训练集上的表现
            resp = {
                'round_number': req['round_number'],
                'weights': obj_to_pickle_string(my_weights),
                'train_size': self.local_model.x_train.shape[0],
                'valid_size': self.local_model.x_valid.shape[0],
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
            }
            if req['run_validation']: # 是否需要测 验证集
                valid_loss, valid_accuracy = self.local_model.validate()
                resp['valid_loss'] = valid_loss
                resp['valid_accuracy'] = valid_accuracy

            self.sio.emit('client_update', resp) # 回复 client_update事件 附带所需的数据


        def on_stop_and_eval(*args): # 收到 stop_and_eval
            req = args[0] # 获得数据 包含当前模型权重
            if req['weights_format'] == 'pickle': # 解码 获得权重
                weights = pickle_string_to_obj(req['current_weights'])
            self.local_model.set_weights(weights) # 更新局部模型
            test_loss, test_accuracy = self.local_model.evaluate() # 获得局部测试集表现
            resp = {
                'test_size': self.local_model.x_test.shape[0],
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            }
            self.sio.emit('client_eval', resp) # 向服务端 发送事件 client_eval 附带数据(局部测试集上的表现)

        # 在这里绑定了 socketIO event与对应的相应函数
        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('init', lambda *args: self.on_init(*args))
        self.sio.on('request_update', on_request_update)
        self.sio.on('stop_and_eval', on_stop_and_eval)




        # TODO: later: simulate datagen for long-running train-serve service
        # i.e. the local dataset can increase while training

        # self.lock = threading.Lock()
        # def simulate_data_gen(self):
        #     num_items = random.randint(10, FederatedClient.MAX_DATASET_SIZE_KEPT * 2)
        #     for _ in range(num_items):
        #         with self.lock:
        #             # (X, Y)
        #             self.collected_data_train += [self.datasource.sample_single_non_iid()]
        #             # throw away older data if size > MAX_DATASET_SIZE_KEPT
        #             self.collected_data_train = self.collected_data_train[-FederatedClient.MAX_DATASET_SIZE_KEPT:]
        #             print(self.collected_data_train[-1][1])
        #         self.intermittently_sleep(p=.2, low=1, high=3)

        # threading.Thread(target=simulate_data_gen, args=(self,)).start()

    
    def intermittently_sleep(self, p=.1, low=10, high=100): # 随机休眠
        if (random.random() < p):
            time.sleep(random.randint(low, high))


# possible: use a low-latency pubsub system for gradient update, and do "gossip"
# e.g. Google cloud pubsub, Amazon SNS
# https://developers.google.com/nearby/connections/overview
# https://pypi.python.org/pypi/pyp2p

# class PeerToPeerClient(FederatedClient):
#     def __init__(self):
#         super(PushBasedClient, self).__init__()    


if __name__ == "__main__":
    server_host = "127.0.0.1"
    server_port = 5000
    FederatedClient(server_host, server_port, datasource.Mnist)
