#分类GNN使用的距离判断程序
# #记录每个节点之间的最短路径
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import networkx as nx
import os
import random
from itertools import islice
from collections import Counter
import collections

from sklearn.model_selection import learning_curve
#计算原始图的最短路径，N越大，p越大


# # B = np.load(fname+".npy")#读取固定的图
# # G = nx.Graph(B)
# G = nx.read_edgelist('./'+fname+'_weight.txt',nodetype = int,data=(('weight',float),),create_using=nx.Graph())
# p= dict(nx.shortest_path_length(G))
#np.save("BA1000_short_path.npy",p)

#print(p[3][2])#3
#print(p[2][3])#3
#print(p[15][17]) #BA20.npy最远距离节点，15，17；17，18
""" max_value,max_key = max(((v,k) for inner_p in p.values() for k,v in inner_p.items()))
print(max_value)
print(max_key)
for a,inner_p in p.items():
    for k,v in inner_p.items():
        if v ==max_value and k==max_key:
            print(a,k) 
#max_value是最短路径的最大值，max_key是其中一种key值，(a,k)是此时的key_value对应的多个情况，它们的最短路径都是max_value """


#读取csv文件到list中
labels = []
preds = []
jordancenterb =[]
jordancenterm =[]
jordancenterw = []
N = 500
fname = "WS"+str(N)
#pycharm中的文件名

bmname = fname+'_p0.1_ob1_jc_m1'
#path = os.path.join('/home/iot/zcy/usb/copy/rexying_diffpool/diffpool-master/data/',bmname)
# path1 = os.path.join('/home/iot/zcy/usb/copy/rexying_diffpool/diffpool-master/labels',bmname)
path = os.path.join('/home/iot/zcy/usb/copy/research/graph_python/research/3.18_DW_CNN/data_0.5',bmname)
file_labels = path +'/'+bmname+'_graph_labels.txt'
file_jordancenterb = path + '/'+bmname+'_jordancenter_b.txt'
file_jordancenterm= path + '/'+bmname+'_jordancenter_m.txt'
file_jordancenterw = path +'/'+bmname+ '_jordancenter_w.txt'
#/home/iot/zcy/usb/copy/research/graph_python/research/3.18_DW_CNN/data/ high67_p0.1_ob1_jc_m10/ high67_p0.1_ob1_jc_m10_graph_labels.txt
#读取txt文件到list中：
with open(file_labels,'r') as f1:
    for line in f1.readlines():
        labels.append(int(line.strip('\n')))
with open(file_jordancenterb,'r') as f2:
    for line in f2.readlines():
        longlist = []
        for x in line.strip('\n').split(','):
            if x == '':
                break
            longlist.append(int(x))
        jordancenterb.append(longlist)
with open(file_jordancenterm,'r') as f4:
    for line in f4.readlines():
        longlist = []
        for x in line.strip('\n').split(','):
            #print(x)
            if x == '':
                break
            longlist.append(int(x))
        jordancenterm.append(longlist)
with open(file_jordancenterw ,'r') as f3:
    for line in f3.readlines():
        longlist = []
        for x in line.strip('\n').split(','):
            #print(x)
            if x == '':
                break
            longlist.append(int(x))
            #print(longlist)
        jordancenterw.append(longlist)

print('len(labels):',len(labels))

#计算预测值与真实值之间最短距离出现的次数，其中距离为0的次数为result[0](即准确预测次数)
read_dic = np.load(fname+"_short_path.npy",allow_pickle = True).item()
#print(read_dic[2][3])
def distance(jc):
    distance_jc =[]#多个jc与真实值之间的平均距离
    sum1=0
    for i in range(len(labels)):
        if type(jc[i]) ==int :
            a = read_dic[labels[i]][jc[i]]
        else:
            b = 0
            for j in jc[i]:
                b=b+read_dic[labels[i]][j]
            a=b/len(jc[i])
        distance_jc.append(a)
        sum1=sum1+a
    average_distance_preds = sum1/(len(distance_jc))
    #print('预测值与真实值之间的距离：',distance_preds)
    #print('平均多个jc_b距离：{:.4}'.format(average_distance_preds))
    result = {}
    for i in set(distance_jc):
        result[i] = distance_jc.count(i)
    return average_distance_preds
def distance_X(jc):
    distance_jc =[]#多个jc与真实值之间的平均距离
    sum1=0
    for i in range(len(labels)):
        if type(jc[i]) ==int :
            a = read_dic[labels[i]][jc[i]]
        else:
            # b = 0
            # for j in jc[i]:
            #     b=b+read_dic[labels[i]][j]
            # a=b/len(jc[i])
            a=read_dic[labels[i]][jc[i][-1]]
        distance_jc.append(a)
        sum1=sum1+a
    average_distance_preds = sum1/(len(distance_jc))
    #print('预测值与真实值之间的距离：',distance_preds)
    #print('平均多个jc_b距离：{:.4}'.format(average_distance_preds))
    result = {}
    for i in set(distance_jc):
        result[i] = distance_jc.count(i)
    return average_distance_preds
#print(result[1]+result[0])
print("jc_b:",distance(jordancenterb))
print("jc_m:",distance_X(jordancenterm))
print("jc_w:",distance(jordancenterw))
acc_count=0
for i in range(len(labels)):
    if labels[i]==jordancenterm[i][-1]:
        acc_count = acc_count+1
print("acc:",acc_count/len(labels))
# #画出预测值频率直方图
# distance_preds_sequence = sorted([d for d in distance_preds], reverse=True)  # distance sequence
# distance_predsCount = collections.Counter(distance_preds_sequence)
# dis_preds, cnt = zip(*distance_predsCount.items())
# # dis_preds1=[]
# # for i in dis_preds:
# #     dis_preds1.append(i)
# #     dis_preds1.append(2)
# #     dis_preds1.append(3)
# dis_preds_frequence = []
# for i in cnt:
#     dis_preds_frequence.append(i/(len(labels)))
# #fig, ax = plt.subplots()
# plt.figure()
# ax1 = plt.subplot(3,2,1)
# plt.bar(dis_preds, dis_preds_frequence, width=0.10, color="b")

# #plt.title("Distance_preds Histogram")
# plt.ylabel("Frequence")
# plt.xlabel("Distance_preds")
# plt.ylim((0,1))
# ax1.set_xticks([d for d in dis_preds])
# ax1.set_xticklabels(dis_preds)

# #画出约旦中心频率直方图
# distance_center_sequence = sorted([d for d in distance_center], reverse=True)  # distance_center sequence
# distance_centerCount = collections.Counter(distance_center_sequence)
# dis_center, center_cnt = zip(*distance_centerCount.items())
# dis_center_frequence = []
# for i in center_cnt:
#     dis_center_frequence.append(i/(len(distance_center)))
# ax2 = plt.subplot(3,2,2)
# plt.bar(dis_center, dis_center_frequence, width=0.1, color="g")

# #plt.title("Jordan Center Histogram")
# plt.xlabel("Jordan Center")
# plt.ylim((0,1))
# ax2.set_xticks([d for d in dis_center])
# ax2.set_xticklabels(dis_center)

# #保存图片
# path_fig = os.path.join('/home/iot/zcy/usb/copy/research/graph_python/research/experiments_graph',bmname) + '_results.png'
# plt.tight_layout()
# plt.savefig(path_fig)
# plt.show()