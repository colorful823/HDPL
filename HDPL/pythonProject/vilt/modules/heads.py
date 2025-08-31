import torch
import torch.nn as nn
import torch.nn.functional as F
from pythonProject.utils import hypergraph_utils as hgut
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


import  numpy as np
from models.layers import HGNN_conv

import torch
import torch.nn as nn
import torch.nn.functional as F

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing  # 用来转化为独热编码
from sklearn.model_selection import train_test_split
from scipy import linalg as LA  # 用来求正交基
from openpyxl import load_workbook


# 在第一次求权重时，并未使用岭回归，还是直接求了伪逆，对于小型数据集这种方法足够了




class node_generator(object):
    def __init__(self, isenhance=False):
        self.Wlist = []
        self.blist = []
        self.function_num = 0
        self.isenhance = isenhance

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def relu(self, x):

        x=x.cpu()
        return np.maximum(x, 0)

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def linear(self, x):
        return x

    def generator(self, shape, times):
        # times是多少组mapping nodes
        for i in range(times):
            W = 2 * np.random.random(size=shape) - 1
            if self.isenhance:
                W = LA.orth(W)  # 求正交基，只在增强层使用。也就是原始输入X变成mapping nodes的W和mapping nodes变成enhancement nodes的W要正交
            b = 2 * np.random.random() - 1
            yield (W, b)

    def generator_nodes(self, data, times, batchsize, function_num):
        # 按照bls的理论，mapping layer是输入乘以不同的权重加上不同的偏差之后得到的若干组，
        # 所以，权重是一个列表，每一个元素可作为权重与输入相乘
        data = data.squeeze(1)


        # 将 NumPy 数组转换为 PyTorch 张量

        data = data.to('cuda')
        data = data.to(torch.float32)  # 或者 torch.float16
          # 或者 torch.float16

        self.Wlist = [elem[0] for elem in self.generator((data.shape[1], batchsize), times)]
        self.blist = [elem[1] for elem in self.generator((data.shape[1], batchsize), times)]

        self.function_num = {'linear': self.linear,
                             'sigmoid': self.sigmoid,
                             'tanh': self.tanh,
                             'relu': self.relu}[function_num]  # 激活函数供不同的层选择
        # 下面就是先得到一组mapping nodes，再不断叠加，得到len(Wlist)组mapping nodes
        print(f"data address: {data.device}")
        self.Wlist=torch.tensor(self.Wlist).to('cuda')
        self.blist=torch.tensor(self.blist).to('cuda')
        self.Wlist= self.Wlist.to(torch.float32)
        self.blist=self.blist.to(torch.float32)
        # nodes = self.function_num(data.dot((self.Wlist[0]).squeeze(0)) + (self.blist[0]).unsqueeze(0))
        nodes_list = []
        for i in range(len(self.Wlist)):
            W = self.Wlist[i].squeeze(0)  # (768, 15)
            b = self.blist[i]  # 标量

            nodes = self.function_num(torch.matmul(data, W) + b)  # 计算 (88, 15)

            nodes_list.append(nodes)

        # 将所有节点合并到一个 tensor 中
        nodes = torch.cat(nodes_list, dim=1)  # 合并在特征维度上，得到 (88, 10*15)
        return nodes

    def transform(self, testdata):
        # testnodes = self.function_num(testdata.dot((self.Wlist[0]).squeeze(0)) + (self.blist[0]).unsqueeze(0))
        testdata = testdata.squeeze(1)



        testdata = testdata.to('cuda')
        testdata = testdata.to(torch.float32)  # 或者 torch.float16

        self.Wlist = torch.tensor(self.Wlist).to('cuda')
        self.blist = torch.tensor(self.blist).to('cuda')
        self.Wlist = self.Wlist.to(torch.float32)
        self.blist = self.blist.to(torch.float32)
        # nodes = self.function_num(data.dot((self.Wlist[0]).squeeze(0)) + (self.blist[0]).unsqueeze(0))
        testnodes_list = []
        for i in range(len(self.Wlist)):
            W = self.Wlist[i].squeeze(0)  # (768, 15)
            b = self.blist[i]  # 标量

            testnodes = self.function_num(torch.matmul(testdata, W) + b)  # 计算 (88, 15)
            testnodes_list.append(testnodes)

        # 将所有节点合并到一个 tensor 中
        testnodes = torch.cat(testnodes_list, dim=1)  # 合并在特征维度上，得到 (88, 10*15)

        return testnodes

    def update(self, otherW, otherb):
        # 权重更新
        self.Wlist += otherW
        self.blist += otherb





# class broadNet(object):
#     def __init__(self, map_num=10, enhance_num=10, DESIRED_ACC=0.80, EPOCH=3, STEP=20, map_function='linear',
#                  enhance_function='linear', batchsize='auto'):
#         self.map_num = map_num  # 多少组mapping nodes
#         self.enhance_num = enhance_num  # 多少组engance nodes
#         self.batchsize = batchsize
#         self.map_function = map_function
#         self.enhance_function = enhance_function
#         self.DESIRED_ACC = DESIRED_ACC
#         self.EPOCH = EPOCH
#         self.STEP = STEP
#         self.W = 0
#         self.pseudoinverse = 0
#         self.mapping_generator = node_generator()
#         self.enhance_generator = node_generator(isenhance=True)
#         self.A=0
#         self.B=0
#     def fit(self, data, label):
#         if self.batchsize == 'auto':
#             self.batchsize = data.shape[1]
#
#         mappingdata = self.mapping_generator.generator_nodes(data, self.map_num, self.batchsize, self.map_function)
#         enhancedata = self.enhance_generator.generator_nodes(mappingdata, self.enhance_num, self.batchsize,
#                                                              self.enhance_function)
#
#         print('number of mapping nodes {0}, number of enhance nodes {1}'.format(mappingdata.shape[1],
#                                                                                 enhancedata.shape[1]))
#         # print('mapping nodes maxvalue {0} minvalue {1} '.format(round(np.max(mappingdata), 5),
#         #                                                         round(np.min(mappingdata), 5)))
#         # print('enhance nodes maxvalue {0} minvalue {1} '.format(round(np.max(enhancedata), 5),
#         #                                                         round(np.min(enhancedata), 5)))
#
#         inputdata = np.column_stack((mappingdata, enhancedata))
#         print('input shape ', inputdata.shape)   #352,2000
#         inputdata = torch.tensor(inputdata,dtype=torch.float32)
#
#         # 求伪逆
#         self.pseudoinverse = np.linalg.pinv(inputdata)
#         # 新的输入到输出的权重
#         print('pseudoinverse shape:', self.pseudoinverse.shape) #352,2000
#         self.W = self.pseudoinverse.dot(label.cpu())
#         print(f"W SHAPE before any predict operation:",self.W.shape)
#         # 查看当前的准确率
#         Y = self.predict(data)
#
#         label = np.array(label.cpu())
#
#         # 确保 Y 和 label 是一维数组
#         Y = Y.ravel()  # 让 Y 保持 (352,)
#         label = label.ravel()  # 让 label 保持 (352,)
#
#         # 将 Y 转换为二进制标签 (0 或 1)
#         # predicted_label = (Y > 0.5).astype(int)
#         predicted_label = (Y > 0.4).astype(int)  # 将概率转换为 0 或 1
#         accuracy = self.accuracy(predicted_label, label)
#         print("initial setting, number of mapping nodes {0}, number of enhance nodes {1}, accuracy {2}".format(
#             mappingdata.shape[1], enhancedata.shape[1], round(accuracy, 5)))
#         # 如果准确率达不到要求并且训练次数小于设定次数，重复添加enhance_nodes
#         epoch_now = 0
#         while accuracy < self.DESIRED_ACC and epoch_now < self.EPOCH:
#             #data dataframe 352,4    label list 352
#             Y = self.addingenhance_predict(data, label, self.STEP, self.batchsize)
#             predicted_label = (Y > 0.4).astype(int)  # 将概率转换为 0 或 1
#             accuracy = self.accuracy(predicted_label, label)
#             epoch_now += 1
#             if epoch_now == 3:
#                 out_feats=np.column_stack((self.A,self.B))
#                 return  out_feats
#         return inputdata
#
#     def decode(self, Y_onehot):
#         return np.ravel(Y_onehot)  # 直接返回预测标签
#
#     def accuracy(self, predictlabel, label):
#         predictlabel = np.ravel(predictlabel).tolist()
#         label = np.ravel(label).tolist()
#         count = sum([1 for i in range(len(label)) if label[i] == predictlabel[i]])
#         print(f"w shape after accuracy:",self.W.shape)
#         return round(count / len(label), 5)
#
#     def predict(self, testdata):
#         # testdata = self.normalscaler.transform(testdata)
#         test_inputdata = self.transform(testdata)
#         print(f"W SHAPE pretend to predict :",self.W.shape)
#         predictions = test_inputdata.dot(self.W)
#         # 确保预测值在合理范围内
#         predictions = np.clip(predictions, 0, 1)
#         return self.decode(predictions)
#
#
#     def transform(self, data):
#         mappingdata = self.mapping_generator.transform(data)
#         enhancedata = self.enhance_generator.transform(mappingdata)
#         return np.column_stack((mappingdata, enhancedata))
#
#     def addingenhance_nodes(self, data, label, step=1, batchsize='auto'):
#         if batchsize == 'auto':
#             batchsize = data.shape[1]
#
#         self.A=mappingdata = self.mapping_generator.transform(data)
#         inputdata = self.transform(data)
#         localenhance_generator = node_generator()
#         self.B=extraenhance_nodes = localenhance_generator.generator_nodes(mappingdata, step, batchsize, self.enhance_function)#additional enhanced nodes
#
#         D = self.pseudoinverse.dot(extraenhance_nodes)
#         C = extraenhance_nodes - inputdata.dot(D)
#         BT = np.linalg.pinv(C) if (C == 0).any() else np.mat((D.T.dot(D) + np.eye(D.shape[1]))).I.dot(D.T).dot(
#             self.pseudoinverse)
# #BT ndarray(500,352) C ndarray  (352,500) D ndarray (2000,500)
#         print(f"w shape before w_transformed:",self.W.shape)
#         # 先将一维数组转换为二维数组
#         label = np.array(label)
#         label = label.reshape(-1, 1)
#         W_transformed = (self.W.reshape(-1,1) - D.dot(BT).dot(label)).reshape(-1, 1)  # 变为 (2000, 1)
#         print(f"w shape after W_transformed :")
#         BT_transformed = BT.dot(label).reshape(-1, 1)  # 变为 (500, 1)
#
#         # 使用 np.vstack 或 np.row_stack 堆叠
#         # self.W = np.row_stack((W_transformed, BT_transformed))  # 变为 (2500, 1)
#         # self.W = np.row_stack( ( (self.W - D.dot(BT).dot(label)).reshape(-1,1), (BT.dot(label)).reshape(-1,1)  ))
#         print(f"w shape before np.concatenate:", self.W.shape)
#         self.W = np.concatenate((W_transformed, BT_transformed), axis=0)  # 变为 (2500, 1)
#         print(f"w shape after np.concatenate:", self.W.shape)
#         self.enhance_generator.update(localenhance_generator.Wlist, localenhance_generator.blist)
#         print(f" w shape before pseudoinverse:",self.W.shape)
#         self.pseudoinverse = np.concatenate((self.pseudoinverse - D.dot(BT), BT))
#         print(f" w shape after pseudoinverse:",self.W.shape)
#
#     def addingenhance_predict(self, data, label, step=1, batchsize='auto'):
#         self.addingenhance_nodes(data, label, step, batchsize)
#
#         test_inputdata = self.transform(data)
#         #test_inputdata ndarray 352,2500   self.w (2500,1)
#         print(f"test_inputdata shape,",test_inputdata.shape)
#         print(f"self.w shape,",self.W)
#         predictions = test_inputdata.dot(self.W)
#         predictions = np.clip(predictions, 0, 1)
#         return self.decode(predictions)
import numpy as np
import torch


# 假设node_generator类已定义
class node_generator:
    def __init__(self, isenhance=False):
        self.isenhance = isenhance

    def generator_nodes(self, data, num, batchsize, func_type):
        # 这里仅为示例实现，实际应根据需要实现
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        # 简单生成随机节点作为示例
        return np.random.randn(data.shape[0], num)


class broadNet(object):
    def __init__(self, map_num=10, enhance_num=10, DESIRED_ACC=0.90, EPOCH=3, STEP=20,
                 map_function='linear', enhance_function='linear', batchsize='auto',
                 max_width_expansions=5):  # 新增：最大宽度扩展次数
        self.map_num = map_num
        self.enhance_num = enhance_num
        self.batchsize = batchsize
        self.map_function = map_function
        self.enhance_function = enhance_function
        self.DESIRED_ACC = DESIRED_ACC
        self.EPOCH = EPOCH
        self.STEP = STEP
        self.max_width_expansions = max_width_expansions  # 最大宽度扩展次数
        self.W = 0
        self.pseudoinverse = 0
        self.mapping_generator = node_generator()
        self.enhance_generator = node_generator(isenhance=True)
        self.A = 0
        self.B = 0
        self.layers = []  # 存放每一层 BLS
        self.current_width_expansions = 0  # 当前宽度扩展次数

    def accuracy(self, pred, label):
        """计算准确率的辅助函数"""
        return np.mean(pred == label)

    def predict(self, data):
        """预测函数，假设已实现"""
        # 这里仅为示例实现
        mappingdata = self.mapping_generator.generator_nodes(data, self.map_num, self.batchsize, self.map_function)
        enhancedata = self.enhance_generator.generator_nodes(mappingdata, self.enhance_num,
                                                             self.batchsize, self.enhance_function)
        inputdata = np.column_stack((mappingdata, enhancedata))
        inputdata = torch.tensor(inputdata, dtype=torch.float32)
        return (inputdata.numpy() @ self.W).ravel()

    def addingenhance_predict(self, data, label, step, batchsize):
        """增加增强节点并预测，假设已实现"""
        # 这里仅为示例实现，实际应增加增强节点并重新计算
        self.enhance_num += step
        return self.predict(data)

    def fit(self, data, label):
        if self.batchsize == 'auto' and hasattr(data, 'shape'):
            self.batchsize = data.shape[1]

        # 初始生成映射节点和增强节点
        mappingdata = self.mapping_generator.generator_nodes(data, self.map_num, self.batchsize, self.map_function)
        enhancedata = self.enhance_generator.generator_nodes(mappingdata, self.enhance_num,
                                                             self.batchsize, self.enhance_function)

        inputdata = np.column_stack((mappingdata, enhancedata))
        inputdata = torch.tensor(inputdata, dtype=torch.float32)

        # 计算伪逆和权重
        self.pseudoinverse = np.linalg.pinv(inputdata)
        self.W = self.pseudoinverse.dot(label.cpu())

        # 计算初始准确率
        Y = self.predict(data)
        label_np = np.array(label.cpu()).ravel()
        Y_ravel = Y.ravel()
        predicted_label = (Y_ravel > 0.4).astype(int)
        accuracy = self.accuracy(predicted_label, label_np)

        print(f"Initial accuracy = {accuracy:.4f}")

        total_epochs = 0
        # 先尝试扩展宽度
        while accuracy < self.DESIRED_ACC and total_epochs < self.EPOCH:
            # 检查是否已达到最大宽度扩展次数
            if self.current_width_expansions >= self.max_width_expansions:
                print(f"Reached maximum width expansions ({self.max_width_expansions}), adding new layer...")
                # 获取当前层输出作为下一层输入
                current_output = np.column_stack((mappingdata, enhancedata))
                # 增加新层
                current_output = self.add_layer(current_output, label)
                # 重新计算准确率
                Y = self.predict(data)
                Y_ravel = Y.ravel()
                predicted_label = (Y_ravel > 0.4).astype(int)
                accuracy = self.accuracy(predicted_label, label_np)
                print(f"After adding new layer, accuracy = {accuracy:.4f}")
                # 重置宽度扩展计数
                self.current_width_expansions = 0
            else:
                # 继续扩展宽度
                Y = self.addingenhance_predict(data, label, self.STEP, self.batchsize)
                Y_ravel = Y.ravel()
                predicted_label = (Y_ravel > 0.4).astype(int)
                accuracy = self.accuracy(predicted_label, label_np)
                self.current_width_expansions += 1
                total_epochs += 1
                print(f"Width expansion {self.current_width_expansions}, Epoch {total_epochs}, "
                      f"Accuracy = {accuracy:.4f}, Enhance num = {self.enhance_num}")

        # 存储当前层信息
        out_feats = np.column_stack((mappingdata, enhancedata))
        self.layers.append((self.mapping_generator, self.enhance_generator, self.W, self.pseudoinverse))

        return out_feats

    def add_layer(self, prev_out, label):
        """在已有 BLS 基础上，堆叠新的一层（深度扩展）"""
        print("=== Adding a new layer ===")
        # 创建新层时继承当前网络的参数
        new_net = broadNet(
            map_num=self.map_num,
            enhance_num=self.enhance_num,
            DESIRED_ACC=self.DESIRED_ACC,
            EPOCH=self.EPOCH,
            STEP=self.STEP,
            map_function=self.map_function,
            enhance_function=self.enhance_function,
            batchsize=self.batchsize,
            max_width_expansions=self.max_width_expansions
        )
        # 用前一层的输出作为新层的输入进行训练
        out_feats = new_net.fit(prev_out, label)
        self.layers.append(new_net)
        return out_feats


class InfoNCE_ThreeModal(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def compute_loss(self, x, y):
        """两模态的 InfoNCE Loss"""
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)

        logits = torch.matmul(x, y.t()) / self.temperature
        labels = torch.arange(logits.size(0)).to(logits.device)

        loss_x = F.cross_entropy(logits, labels)
        loss_y = F.cross_entropy(logits.t(), labels)

        return (loss_x + loss_y) / 2, x, y

    def forward(self, text_shared, vision_shared, image_shared):
        """
        text_shared:   [B, D]
        vision_shared: [B, D]
        image_shared:  [B, D]
        return: loss, t''_en, v''_en, i''_en
        """
        # 三对组合
        loss_tv, t_en, v_en = self.compute_loss(text_shared, vision_shared)
        loss_ti, t_en, i_en = self.compute_loss(text_shared, image_shared)
        loss_vi, v_en, i_en = self.compute_loss(vision_shared, image_shared)

        # 总loss = 三个loss的平均
        loss = (loss_tv + loss_ti + loss_vi) / 3

        return loss, t_en, v_en, i_en









class FeatureProcessor(nn.Module):
    def __init__(self, in_ch, hidden_size, m_prob=1.0, K_neigs=[11], is_probH=True, split_diff_scale=False):
        super(FeatureProcessor, self).__init__()
        self.in_ch = in_ch
        self.hidden_size = hidden_size
        self.m_prob = m_prob
        self.K_neigs = K_neigs
        self.is_probH = is_probH
        self.split_diff_scale = split_diff_scale

        # Define the HGNN layers
        self.hgc1 = HGNN_conv(in_ch, hidden_size)
        # self.hgc2 = HGNN_conv(hidden_size, hidden_size)

    def load_feature_construct_H(self, all_outputs, use_cnn_feature=True, use_cnn_feature_for_structure=True):
        cfg = hgut.HOHEConfig(
            bins=10,
            bin_strategy="quantile",
            topk_ig=8,  # 选 IG 前8的特征（或用 ig_threshold=...）
            w1=0.4, w2=0.6,  # 皮尔逊权重大一点
            sim_threshold=0.7,
            standardize=True
        )

        ft = hgut.HOHESelector(cfg).fit(all_outputs, torch.stack(batch['label'])

        fts = None

        if use_cnn_feature:
            fts = hgut.feature_concat(fts, cnn_ft)

        if fts is None:
            raise Exception('No feature used for model!')

        H = None
        if use_cnn_feature_for_structure:
            tmp = hgut.construct_H_with_KNN(cnn_ft, K_neigs=self.K_neigs,
                                            split_diff_scale=self.split_diff_scale,
                                            is_probH=self.is_probH, m_prob=self.m_prob)
            H = hgut.hyperedge_concat(H, tmp)

        return fts, H

    def generate_G(self, H):
        return hgut.generate_G_from_H(H)

    def forward(self, all_outputs):
        # Process the features and construct the incidence matrix H
        fts, H = self.load_feature_construct_H(all_outputs)  #all_outputs 3,1

        # Generate G from H
        G = self.generate_G(H)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        fts = torch.Tensor(fts).to(device)
        G = torch.Tensor(G).to(device)
        # Pass through HGNN layers
        x = F.relu(self.hgc1(fts, G))
        # x = F.dropout(x, 0.5)
        # x = self.hgc2(x, G)

        return x


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class CNN1D(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size, output_size):
        super(CNN1D, self).__init__()
        # 定义批量归一化层
        self.norm = nn.BatchNorm1d(input_channels)
        # 定义一维卷积层
        self.conv1 = nn.Conv1d(input_channels=6, num_filters=64, kernel_size=3,padding=1)
        # 定义池化层
        self.pool = nn.MaxPool1d(kernel_size=2)
        # 定义一个全连接层，将卷积层的输出映射到输出大小
        # 注意：这里需要根据卷积和池化层后的实际输出维度来设置
        self.fc = nn.Linear(num_filters, output_size)

    def forward(self, x):
        # 增加输入通道维度
         # x的维度应该是[batch_size, input_channels, sequence_length]
        # 批量归一化操作

        x = self.norm(x)  #3,6
        # 卷积后使用ReLU激活函数
        x=x.unsqueeze(2) #1,3,6
        # x=x.transpose(1,2)#1,6,3
        # x=x.transpose(1,2)#
        x=self.conv1(x)
        x = F.relu(x)
        # 池化操作
        # x = self.pool(x)
        # 展平操作，为全连接层准备数据
        x = x.view(x.size(0), -1)  # 调整这里的维度以匹配全连接层的输入
        # 全连接层
        x = self.fc(x)
        return x


class ITMHead(nn.Module):#image-text
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x

class IAMHead(nn.Module):#image-audio
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x
class TAMHead(nn.Module):#text-audio
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x
class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class MPPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, 256 * 3)

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x)
        return x


class ModmisBinHead(nn.Module):
    def __init__(self, hidden_size: int, label_count: int):
        super().__init__()
        self.binarizer = nn.Linear(hidden_size, label_count)

    def forward(self, x):
        return self.binarizer(x)


class ModmisClsHead(nn.Module):
    def __init__(self, hidden_size: int, class_num_list: list[int]):
        super().__init__()
        self.classifier_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, class_num),
            )
            for class_num in class_num_list
        ])

    def forward(self, x):
        return [classifier(x) for classifier in self.classifier_list]


class ModmisRegHead(nn.Module):
    def __init__(self, hidden_size: int, regression_num: int):
        super().__init__()
        # 先尝试简单的线性头 之后再考虑加其它操作（除了pooling已经加上去了）
        self.regresser = nn.Linear(hidden_size, regression_num)

    def forward(self, x):
        return self.regresser(x)
