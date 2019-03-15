import pickle
import keras
import uuid
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import msgpack
import random
import codecs
import numpy as np
import json
import msgpack_numpy
# https://github.com/lebedov/msgpack-numpy

import sys
import time

from flask import *
from flask_socketio import SocketIO
from flask_socketio import *
# https://flask-socketio.readthedocs.io/en/latest/
       

class GlobalModel(object): # 全局模型
    """docstring for GlobalModel"""
    def __init__(self): # 初始化
        self.model = self.build_model() # 模型
        self.current_weights = self.model.get_weights() # 当前权重
        # for convergence check
        self.prev_train_loss = None # 上一个 训练集上的 loss

        # all rounds; losses[i] = [round#, timestamp, loss]
        # round# could be None if not applicable
        self.train_losses = [] # 训练集、验证集上的 loss、acc
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []

        self.training_start_time = int(round(time.time())) # 开始时间
    
    def build_model(self): # 父类不填写 子类会完成
        raise NotImplementedError()

    # client_updates = [(w, n)..]
    def update_weights(self, client_weights, client_sizes): # 更新权重  客户权重的加权平均
        new_weights = [np.zeros(w.shape) for w in self.current_weights] # 新权重 先 全部清零
        total_size = np.sum(client_sizes) # 客户总数据量

        for c in range(len(client_weights)): # 枚举 客户 数量
            for i in range(len(new_weights)): # 枚举 某一个权重
                new_weights[i] += client_weights[c][i] * client_sizes[c] / total_size # 关于数据量的 加权平均
        self.current_weights = new_weights  # 更新权重

    def aggregate_loss_accuracy(self, client_losses, client_accuracies, client_sizes): # 集成 损失值 与 准确度
        total_size = np.sum(client_sizes) # 客户 总数据量
        # weighted sum 依据数据量 的加权 平均
        aggr_loss = np.sum(client_losses[i] / total_size * client_sizes[i]
                for i in range(len(client_sizes)))
        aggr_accuraries = np.sum(client_accuracies[i] / total_size * client_sizes[i]
                for i in range(len(client_sizes)))
        return aggr_loss, aggr_accuraries

    # cur_round coule be None
    def aggregate_train_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round): # 把每一轮的 训练集上的结果 整合起来
        cur_time = int(round(time.time())) - self.training_start_time # 已经历的时间
        aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes) # 平均后的 loss, acc
        self.train_losses += [[cur_round, cur_time, aggr_loss]] # 利用列表， 每一轮的结果整合起来
        self.train_accuracies += [[cur_round, cur_time, aggr_accuraries]]
        with open('stats.txt', 'w') as outfile: # 输出到 stats.txt 中
            json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    # cur_round coule be None
    def aggregate_valid_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round): # 把每一轮 验证集上的结果 整合起来
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        self.valid_losses += [[cur_round, cur_time, aggr_loss]]
        self.valid_accuracies += [[cur_round, cur_time, aggr_accuraries]]
        with open('stats.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    def get_stats(self): # 返回 一个字典 各个域存储 当前训练集和验证集上的结果
        return {
            "train_loss": self.train_losses,
            "valid_loss": self.valid_losses,
            "train_accuracy": self.train_accuracies,
            "valid_accuracy": self.valid_accuracies
        }
        

class GlobalModel_MNIST_CNN(GlobalModel): # 继承至全局模型 实现 MNIST数据集上的 CNN 实例
    def __init__(self): # 调用父类 初始化
        super(GlobalModel_MNIST_CNN, self).__init__()

    def build_model(self): # 创建 CNN 模型 识别手写数字
        # ~5MB worth of parameters
        model = Sequential() # 28*28*1 的 图片 -> 10 的概率分布
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(28, 28, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model

        
######## Flask server with Socket IO ########

# Federated Averaging algorithm with the server pulling from clients

class FLServer(object):  # 服务端
    
    MIN_NUM_WORKERS = 1 # 最少工人数
    MAX_NUM_ROUNDS = 50 # 最大训练轮数
    NUM_CLIENTS_CONTACTED_PER_ROUND = 1 # 每轮连接的客户端 数
    ROUNDS_BETWEEN_VALIDATIONS = 2 # 验证的间隔论述

    def __init__(self, global_model, host, port): # 初始化
        self.global_model = global_model() # 加载全局模型 MINIST_CNN

        self.ready_client_sids = set() # 就绪的 客户端 id 集合

        self.app = Flask(__name__) # 搭建 
        self.socketio = SocketIO(self.app)
        self.host = host # ip地址
        self.port = port # 端口

        # UUID 是 通用唯一识别码（Universally Unique Identifier）的缩写
        # 其目的，是让分布式系统中的所有元素，都能有唯一的辨识信息，而不需要通过中央控制端来做辨识信息的指定。
        self.model_id = str(uuid.uuid4()) # 申请一个独立 的uuid

        #####
        # training states 训练状态
        self.current_round = -1  # -1 for not yet started 当前轮数
        self.current_round_client_updates = [] # 当前轮客户端的更新们
        self.eval_client_updates = [] # 客户的评估更新  也就是测试集上的表现
        #####

        # socket io messages
        self.register_handles() # 运行socket服务


        @self.app.route('/') # 一些网页可视化
        def dashboard():
            return render_template('dashboard.html')

        @self.app.route('/stats') # 一些网页可视化
        def status_page(): # 返回当前全局模型的
            return json.dumps(self.global_model.get_stats()) # 当前训练集和验证集上的结果

        
    def register_handles(self):# socket.io 服务 相应 各个socketIO事件(event)
        # single-threaded async, no need to lock

        @self.socketio.on('connect') # 收到 connect
        def handle_connect(): # 打印 xx connected
            print(request.sid, "connected")

        @self.socketio.on('reconnect') # 收到 reconnect
        def handle_reconnect(): # 打印 xx reconnected
            print(request.sid, "reconnected")

        @self.socketio.on('disconnect') # 收到 xx disconnected
        def handle_reconnect(): # 打印 xx disconnected
            print(request.sid, "disconnected")
            if request.sid in self.ready_client_sids: # 并且从 就绪客户端id集合中 将其删除
                self.ready_client_sids.remove(request.sid)

        @self.socketio.on('client_wake_up') # 收到 client_wake_up
        def handle_wake_up():
            print("client wake_up: ", request.sid) # 打印 client id
            emit('init', { # 向客户端回复 发出事件 init 后面接一个字典 代表传输的数据
                    'model_json': self.global_model.model.to_json(), # 全局模型端json形式
                    'model_id': self.model_id, # 模型端 uuid
                    'min_train_size': 1200, # 最小训练数据量
                    'data_split': (0.6, 0.3, 0.1), # train, test, valid  数据分割比例
                    'epoch_per_round': 1,  # 每批次论述
                    'batch_size': 10 # 批训练大小
                })

        @self.socketio.on('client_ready') # 收到 client_ready
        def handle_client_ready(data): # 这里客户端 传了个 data 过来
            print("client ready for training", request.sid, data) # 打印 客户端id 数据
            self.ready_client_sids.add(request.sid) # 加入到客户端就绪队列
            # 如果就绪队列中的客户端数量 >= 服务端要求的最小值 而且服务端还处于未开始训练的状态
            if len(self.ready_client_sids) >= FLServer.MIN_NUM_WORKERS and self.current_round == -1: # 开始第一轮
                self.train_next_round()
                self.begin_time = int(round(time.time()))
                print("begin_time(s):", self.begin_time())

        @self.socketio.on('client_update') # 收到 client_update
        def handle_client_update(data):
            print("received client update of bytes: ", sys.getsizeof(data)) # 打印收到的数据大小
            print("handle client_update", request.sid) # 打印 客户端 id
            for x in data: # 这里的data是个字典 {key : value}  枚举的x是枚举的key
                if x != 'weights': #打印出除了'weights'以外的域 下方描述了data的构成
                    print(x, data[x])
            # data: 
            #   weights 权重
            #   train_size 训练集大小
            #   valid_size 验证集大小 
            #   train_loss 训练集上的loss
            #   train_accuracy 训练集上的acc
            #   valid_loss? 验证集上的loss
            #   valid_accuracy? 验证集上的acc

            # discard outdated update 丢弃 过期的更新
            if data['round_number'] == self.current_round: # 判断轮数是否和当前轮数一致
                self.current_round_client_updates += [data] # 数据存起来
                # 将最新添加进来的那一个 权重解压。 这里客户端传来的'weights'应当是用pickle压缩过的
                self.current_round_client_updates[-1]['weights'] = pickle_string_to_obj(data['weights']) 
                
                # tolerate 30% unresponsive clients 容忍30%的无响应客户端
                # 如果 传来更新请求的客户端数量 >= 每轮连接的客户端数量的 70%
                if len(self.current_round_client_updates) > FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND * .7:
                    self.global_model.update_weights( # 将数据交给全局模型 去更新
                        [x['weights'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates], 
                    ) # 集成各个的loss 和acc
                    aggr_train_loss, aggr_train_accuracy = self.global_model.aggregate_train_loss_accuracy(
                        [x['train_loss'] for x in self.current_round_client_updates],
                        [x['train_accuracy'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                        self.current_round
                    )
                    # 打印训练集上的loss 和acc
                    print("aggr_train_loss", aggr_train_loss)
                    print("aggr_train_accuracy", aggr_train_accuracy)
                    # 如果有验证集上的信息
                    if 'valid_loss' in self.current_round_client_updates[0]: # 也给全局模型处理了 并打印出来
                        aggr_valid_loss, aggr_valid_accuracy = self.global_model.aggregate_valid_loss_accuracy(
                            [x['valid_loss'] for x in self.current_round_client_updates],
                            [x['valid_accuracy'] for x in self.current_round_client_updates],
                            [x['valid_size'] for x in self.current_round_client_updates],
                            self.current_round
                        )
                        print("aggr_valid_loss", aggr_valid_loss)
                        print("aggr_valid_accuracy", aggr_valid_accuracy)

                    # 如果前一轮的train_loss和现在的train_loss变化小于1% 判断为收敛 停止训练并开始评估
                    if self.global_model.prev_train_loss is not None and \
                            (self.global_model.prev_train_loss - aggr_train_loss) / self.global_model.prev_train_loss < .001:
                        # converges 收敛
                        print("converges! starting test phase..")
                        self.stop_and_eval() # 服务端结束 并向客户端发出 eval请求 开始测试集上的表现计算
                        return
                    
                    self.global_model.prev_train_loss = aggr_train_loss

                    # 如果超过了训练论数上限则停止训练并开始做评估 否则 继续下一轮
                    if self.current_round >= FLServer.MAX_NUM_ROUNDS: 
                        self.stop_and_eval() # 服务端结束 并向客户端发出 eval请求 开始测试集上的表现计算
                    else:
                        self.train_next_round()

        @self.socketio.on('client_eval') # 收到 client_eval
        def handle_client_eval(data): 
            if self.eval_client_updates is None:
                return
            print("handle client_eval", request.sid)
            print("eval_resp", data)
            self.eval_client_updates += [data] # 数据存入 eval_updates中

            # tolerate 30% unresponsive clients 超过0.7是开始处理 eval_updates 
            if len(self.eval_client_updates) > FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND * .7:
                aggr_test_loss, aggr_test_accuracy = self.global_model.aggregate_loss_accuracy(
                    [x['test_loss'] for x in self.eval_client_updates],
                    [x['test_accuracy'] for x in self.eval_client_updates],
                    [x['test_size'] for x in self.eval_client_updates],
                );
                print("\naggr_test_loss", aggr_test_loss) # 输出测试集上的表现
                print("aggr_test_accuracy", aggr_test_accuracy)
                self.end_time = int(round(time.time()))
                print("end_time(s):", self.end_time)
                print("total_time(s):", self.end_time - self.begin_time)
                print("== done ==")
                self.eval_client_updates = None  # special value, forbid evaling again

    
    # Note: we assume that during training the #workers will be >= MIN_NUM_WORKERS
    def train_next_round(self): # 训练 下一轮
        self.current_round += 1 # 当前轮数 + 1
        # buffers all client updates
        self.current_round_client_updates = [] # 清空收到的客户端更新

        print("### Round ", self.current_round, "###") # 打印论述
        # 从就绪客户队列中 随机选取 所要求的 每轮连接客户端数
        client_sids_selected = random.sample(list(self.ready_client_sids), FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND)
        print("request updates from", client_sids_selected) # 打印 选中的客户端

        # by default each client cnn is in its own "room"
        for rid in client_sids_selected: # 向这些客户端 发送 update请求
            emit('request_update', {
                    'model_id': self.model_id,
                    'round_number': self.current_round,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights), # 用pickle 压缩了

                    'weights_format': 'pickle', # 权重保存形式 pickle
                    'run_validation': self.current_round % FLServer.ROUNDS_BETWEEN_VALIDATIONS == 0, # 每两轮看一下验证集上的表现
                }, room=rid)

    
    def stop_and_eval(self): # 训练结束 进行测试集上的表现
        self.current_round = 999
        self.eval_client_updates = [] # 清空
        for rid in self.ready_client_sids: # 向就绪队列中的客户端 发送 eval请求
            emit('stop_and_eval', {
                    'model_id': self.model_id,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    'weights_format': 'pickle'
                }, room=rid)

    def start(self): # 运行
        self.socketio.run(self.app, host=self.host, port=self.port)



def obj_to_pickle_string(x): # 将对象 用 pickle 保存
    # return x
    # return pickle.dumps(x)
    return codecs.encode(pickle.dumps(x), "base64").decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO
 
def pickle_string_to_obj(s): # 从pickle中 加载 对象
    # return s
    # return pickle.loads(s)
    return pickle.loads(codecs.decode(s.encode(), "base64"))
    # return msgpack.unpackb(s, object_hook=msgpack_numpy.decode)


if __name__ == '__main__':
    # When the application is in debug mode the Werkzeug development server is still used
    # and configured properly inside socketio.run(). In production mode the eventlet web server
    # is used if available, else the gevent web server is used.
    
    server_host = "192.168.0.103"
    server_port = 5000

    server = FLServer(GlobalModel_MNIST_CNN, server_host, server_port) # 服务端
    print("listening on %s:%d"%(server_host, server_port))
    server.start()
