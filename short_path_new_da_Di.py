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
#计算原始图的最短路径，N越大，p越大

N =500
fname = "BA"+str(N)
# B = np.load(fname+".npy")#读取固定的图
# G = nx.Graph(B)
G = nx.read_edgelist('./'+fname+'_weight_2.txt',nodetype = int,data=(('weight',float),),create_using=nx.DiGraph())
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

""" #将list写入csv
labels = [11,22,3,4,5]
with open('labels.csv','w') as f:
    writer = csv.writer(f)
    for i in labels:
        writer.writerow(str(i))#整数i不能直接写入

print(labels) """

#读取csv文件到list中
labels = []
preds = []
da_b = []
da_m =[]
da_w =[]

# unbet = []
# discen = []
# dynage = []
""" #/home/zhang/Documents/pytorch/learn/GraphKernel/rexying_diffpool/diffpool-master
with open('/home/zhang/Documents/pytorch/learn/GraphKernel/rexying_diffpool/diffpool-master/wormpro11_labels.csv','r') as f1:
    context = f1.read()
    list_results = context.split('\n')
    for i in range(len(list_results)-1):
        labels.append(eval(list_results[i]))
with open('/home/zhang/Documents/pytorch/learn/GraphKernel/rexying_diffpool/diffpool-master/wormpro11_preds.csv','r') as f2:
    context = f2.read()
    list_results = context.split('\n')
    for i in range(len(list_results)-1):
        preds.append(eval(list_results[i])) """

#pycharm中的文件名
bmname = fname+'_p0.5_m3_da'
#path = os.path.join('/home/iot/zcy/usb/copy/rexying_diffpool/diffpool-master/data/',bmname)
# path1 = os.path.join('/home/iot/zcy/usb/copy/rexying_diffpool/diffpool-master/labels',bmname)
path = os.path.join('/home/iot/zcy/usb/copy/research/graph_python/research/3.18_DW_CNN/data_0.5',bmname)
file_labels = path +'/'+bmname+'_graph_labels.txt'
#file_jordancenter = path + '/'+bmname+'_jordancenter.txt'
#file_b = path + '/'+bmname+'_dynage_b.txt'
file_m = path + '/'+bmname+'_dynage_m.txt'
#file_w = path + '/'+bmname+'_dynage_w.txt'
#file_jordancenterall = path +'/'+bmname+ '_jordancenterall.txt'

# file_unbet = path + '_unbet.txt'
# file_discen = path + '_discen.txt'
# file_dynage = path + '_dynage.txt'
#读取txt文件到list中：
with open(file_labels,'r') as f1:
    for line in f1.readlines():
        labels.append(int(line.strip('\n')))
# with open(file_b,'r') as f2:
#     for line in f2.readlines():
#         da_b.append(int(line.strip('\n')))
with open(file_m,'r') as f3:
    for line in f3.readlines():
        da_m.append(int(line.strip('\n')))
# with open(file_w,'r') as f4:
#     for line in f4.readlines():
#         da_w.append(int(line.strip('\n')))
# with open(file_jordancenterall ,'r') as f3:
#     for line in f3.readlines():
#         #print(type(line.strip('\n')))  #str
#         #print(len(line.strip('\n')))
#         longlist = []
#         for x in line.strip('\n').split(','):
#             #print(x)
#             if x == '':
#                 break
#             longlist.append(int(x))
#             #print(longlist)
#         jordancenterall.append(longlist)
# ###
# with open(file_eva_labels,'r') as f4:
#     for line in f4.readlines():
#         eva_labels.append(int(line.strip('\n')))
# with open(file_unbet,'r') as f4:
#     for line in f4.readlines():
#         unbet.append(int(line.strip('\n')))
# with open(file_discen,'r') as f5:
#     for line in f5.readlines():
#         discen.append(int(line.strip('\n')))
# with open(file_dynage,'r') as f6:
#     for line in f6.readlines():
#         dynage.append(int(line.strip('\n')))
#print('labels:',labels)
print('len(labels):',len(labels))
#print('preds:',preds)
print('len(preds):',len(preds))      #ctrl+/(#)  或者 ctrl+shift+a(''') 选中内容全部注释  取消同理

#计算预测值与真实值之间最短距离出现的次数，其中距离为0的次数为result[0](即准确预测次数)
read_dic = np.load(fname+"_short_path.npy",allow_pickle = True).item()
#print(read_dic[2][3])
###
# distance_preds =[]#多个jc与真实值之间的平均距离
# sum1=0
# for i in range(len(labels)):
#     if type(jordancenterall[i]) ==int :
#         a = read_dic[labels[i]][jordancenterall[i]]
#     else:
#         b = 0
#         for j in jordancenterall[i]:
#             b=b+read_dic[labels[i]][j]
#         a=b/len(jordancenterall[i])
#     distance_preds.append(a)
#     sum1=sum1+a
# average_distance_preds = sum1/(len(distance_preds))
# #print('预测值与真实值之间的距离：',distance_preds)
# print('平均多个jc距离：{:.2}'.format(average_distance_preds))
# result = {}
# for i in set(distance_preds):
#     result[i] = distance_preds.count(i)
# #print(result[1]+result[0])
def distance(da):
    distance_center=[]#真实值与约旦中心的距离
    #print(nx.eccentricity(G))#G中所有节点的离心率
    #print(nx.radius(G))#G的半径，离心率等于半径的点称为图的约旦中心（中心）
    # G_center = nx.center(G)
    # print('center:{}'.format(G_center))
    sum2= 0
    for i in range(len(labels)):
        b = read_dic[labels[i]][da[i]]
        distance_center.append(b)
        sum2 = sum2+b
    average_distance_center = sum2/(len(distance_center))
    return average_distance_center
#print('约旦中心与真实值之间的距离：',distance_center)
#print('平均daynamic_age_b距离：{:.2}'.format(distance(da_b)))
print('平均daynamic_age_m距离：{:.4}'.format(distance(da_m)))
#print('平均daynamic_age_w距离：{:.2}'.format(distance(da_w)))
count = 0
# for i in range(len(distance_preds)):
#     if distance_preds[i] <= distance_center[i]:
#         count=count+1
# print('预测距离小于等于约旦中心距离个数：',count)
acc_count=0
for i in range(len(labels)):
    if labels[i]==da_m[i]:
        acc_count = acc_count+1
print("acc:",acc_count/len(labels))
# distance_unbet=[]#真实值与无偏中心性的距离
# sum3= 0
# for i in range(len(labels)):
#     b = read_dic[labels[i]][unbet[i]]
#     distance_unbet.append(b)
#     sum3 = sum3+b
# average_distance_unbet = sum3/(len(distance_unbet))
# print('无偏中心性与真实值之间的距离：',distance_unbet)
# print('平均无偏中心性距离：{:.2}'.format(average_distance_unbet))
# count = 0
# for i in range(len(distance_preds)):
#     if distance_preds[i] <= distance_unbet[i]:
#         count=count+1
# print('预测距离小于等于无偏中心性距离个数：',count)

# distance_discen=[]#真实值与距离中心的距离
# sum3= 0
# for i in range(len(labels)):
#     b = read_dic[labels[i]][discen[i]]
#     distance_discen.append(b)
#     sum3 = sum3+b
# average_distance_discen = sum3/(len(distance_discen))
# print('距离中心与真实值之间的距离：',distance_discen)
# print('平均距离中心距离：{:.2}'.format(average_distance_discen))
# count = 0
# for i in range(len(distance_preds)):
#     if distance_preds[i] <= distance_discen[i]:
#         count=count+1
# print('预测距离小于等于距离中心距离个数：',count)

# distance_dynage=[]#真实值与动态年龄的距离
# sum3= 0
# for i in range(len(labels)):
#     b = read_dic[labels[i]][dynage[i]]
#     distance_dynage.append(b)
#     sum3 = sum3+b
# average_distance_dynage = sum3/(len(distance_dynage))
# print('动态年龄与真实值之间的距离：',distance_dynage)
# print('平均动态年龄距离：{:.2}'.format(average_distance_dynage))
# count = 0
# for i in range(len(distance_preds)):
#     if distance_preds[i] <= distance_dynage[i]:
#         count=count+1
# print('预测距离小于等于动态年龄性距离个数：',count)

#画出预测值频率直方图
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
# plt.xlabel("dyamic_age")
# plt.ylim((0,1))
# ax2.set_xticks([d for d in dis_center])
# ax2.set_xticklabels(dis_center)

#画出无偏中心性频率直方图
# distance_unbet_sequence = sorted([d for d in distance_unbet], reverse=True)  # distance_center sequence
# distance_unbetCount = collections.Counter(distance_unbet_sequence)
# dis_unbet, center_cnt = zip(*distance_unbetCount.items())
# dis_unbet_frequence = []
# for i in center_cnt:
#     dis_unbet_frequence.append(i/(len(distance_unbet)))
# ax2 = plt.subplot(3,2,3)
# plt.bar(dis_unbet, dis_unbet_frequence, width=0.1, color="g")

#plt.title("Unbiased Betweenness Histogram")
# plt.xlabel("Distance_unbet")
# plt.ylim((0,1))
# ax2.set_xticks([d for d in dis_unbet])
# ax2.set_xticklabels(dis_unbet)
#画出距离中心频率直方图
# distance_discen_sequence = sorted([d for d in distance_discen], reverse=True)  # distance_center sequence
# distance_discenCount = collections.Counter(distance_discen_sequence)
# dis_discen, center_cnt = zip(*distance_discenCount.items())
# dis_discen_frequence = []
# for i in center_cnt:
#     dis_discen_frequence.append(i/(len(distance_discen)))
# ax2 = plt.subplot(3,2,4)
# plt.bar(dis_discen, dis_discen_frequence, width=0.1, color="g")

#plt.title("Distance Center Histogram")
# plt.xlabel("Distance_discen")
# plt.ylim((0,1))
# ax2.set_xticks([d for d in dis_discen])
# ax2.set_xticklabels(dis_discen)
#画出动态年龄频率直方图

# distance_dynage_sequence = sorted([d for d in distance_dynage], reverse=True)  # distance_center sequence
# distance_dynageCount = collections.Counter(distance_dynage_sequence)
# dis_dynage, center_cnt = zip(*distance_dynageCount.items())
# dis_dynage_frequence = []
# for i in center_cnt:
#     dis_dynage_frequence.append(i/(len(distance_dynage)))
# ax2 = plt.subplot(3,2,5)
# plt.bar(dis_dynage, dis_dynage_frequence, width=0.1, color="g")

# #plt.title("Dynamic Age Histogram")
# plt.xlabel("Distance_dynage")
# plt.ylim((0,1))
# ax2.set_xticks([d for d in dis_dynage])
# ax2.set_xticklabels(dis_dynage)
#保存图片
# path_fig = os.path.join('/home/iot/zcy/usb/copy/research/graph_python/research/experiments_graph',bmname) + '_results.png'
# plt.tight_layout()
# plt.savefig(path_fig)
# plt.show()