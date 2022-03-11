# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 16:57:59 2020

@author: baosh
"""

#线性拟合
import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
#import torchvision
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# pandas and numpy are used for data manipulation
# matplotlib and seaborn are used for plotting graphs
import seaborn
#import yfinance as yf
# fix_yahoo_finance is used to fetch data import fix_yahoo_finance as yf。
#然后我们读取过去10年间每天黄金ETF的价格数据，并将数据储存在Df中。
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, date
import pylab
import yfinance as yf
from collections import  Counter
from sklearn.linear_model import LinearRegression
#import tushare as ts

 

def load_data():
    token = ''
    ts.set_token(token)
    code='600489'
    start_date = '2012-10-12'
    end_date = '2020-12-01' #雅虎财经接口专用日期
    start_date_guozhai='20121012'
    end_date_guozhai='20201201'#tushare接口专用日期
    pro = ts.pro_api()
    Df_gold_price = yf.download('GLD',start_date,end_date) #下载雅虎财经黄金日内数据
    data_gold_price=Df_gold_price.values #雅虎财经黄金日内数据转换成list数据
    days_shift=1 #偏移天数
    Df_gold_price_shift=Df_gold_price.shift(-days_shift) #时间偏移将数据往前挪N天
    Df_DINIW_price = yf.download('UUP',start_date,end_date) #下载雅虎财经美元指数数据
    data_DINIW_price=Df_DINIW_price.values #雅虎财经黄金日内数据转换成list数据
    Df_VIX = yf.download('VIX',start_date,end_date) #下载雅虎财经VIX恐慌指数数据
    #VIX恐慌指数到2018年就结束了
    data_VIX=Df_VIX.values #雅虎财经黄金日内数据转换成list数据
    Df_IRX = yf.download('^IRX',start_date,end_date) #下载雅虎财经13周美元指数数据
    data_IRX=Df_IRX.values #雅虎财经黄金日内数据转换成list数据
    Df_FVX = yf.download('^FVX',start_date,end_date) #下载雅虎财经5年美元指数数据
    data_FVX=Df_FVX.values #雅虎财经黄金日内数据转换成list数据
    Df_TNX = yf.download('^TNX',start_date,end_date) #下载雅虎财经10年美元指数数据
    data_TNX=Df_TNX.values #雅虎财经黄金日内数据转换成list数据
    Df_TYX = yf.download('^TYX',start_date,end_date) #下载雅虎财经30年美元指数数据
    data_TYX=Df_TYX.values #雅虎财经黄金日内数据转换成list数据
    #美国国债收益率曲线利率（日频）获取1月期和1年期数据
    
   #美国国债收益率曲线利率（日频）
    #获取1月期和1年期数据
    #df_tycr = pro.us_tycr(start_date='20180101', end_date='20200327', fields='m1,y1')
    df_tycr = pro.us_tycr(start_date=start_date_guozhai, end_date=end_date_guozhai)
    df_tycr['date'] = pd.to_datetime(df_tycr['date']) #将数据类型转换为日期类型
    df_tycr = df_tycr.set_index('date') # 将date设置为index
    #美国国债收益率曲线利率（日频）结束
    
    #美国国债实际收益率曲线利率
    #获取5年期和20年期数据
    #df_trycr = pro.us_trycr(start_date='20180101', end_date='20200327', fields='y5,y20')
    #美国国债实际收益率曲线利率结束
    
    #美国短期国债利率
    #获取指定字段数据
    df_us_tbr = pro.us_tbr(start_date=start_date_guozhai, end_date=end_date_guozhai)#, fields='w4_bd,w52_ce')
    df_us_tbr['date'] = pd.to_datetime(df_us_tbr['date']) #将数据类型转换为日期类型
    df_us_tbr = df_us_tbr.set_index('date') # 将date设置为index
    #美国短期国债利率结束
    
    #国债长期利率
    #获取5年期和20年期数据
    df_us_tltr = pro.us_tltr(start_date=start_date_guozhai, end_date=end_date_guozhai)#, fields='ltc,cmt')
    df_us_tltr['date'] = pd.to_datetime(df_us_tltr['date']) #将数据类型转换为日期类型
    df_us_tltr = df_us_tltr.set_index('date') # 将date设置为index
    #国债长期利率结束
    
    #国债实际长期利率平均值
    #获取指定字段
    df_us_trltr = pro.us_trltr(start_date=start_date_guozhai, end_date=end_date_guozhai)#, fields='ltr_avg')
    df_us_trltr['date'] = pd.to_datetime(df_us_trltr['date']) #将数据类型转换为日期类型
    df_us_trltr = df_us_trltr.set_index('date') # 将date设置为index
    #国债实际长期利率平均值结束

    #数据整合部分
    Df_tushare=pd.concat([df_tycr, df_us_tbr,df_us_tltr,df_us_trltr],axis=1) #将所有的数据归到同一个矩阵中
    Df_GLD=pd.concat([Df_gold_price_shift['Open'],Df_tushare,Df_DINIW_price,Df_IRX,Df_FVX,Df_TNX,Df_TYX,Df_VIX],axis=1)
    Df_GLD = Df_GLD.fillna(0) #具有NaN的点换成0
    
    seq=Df_GLD['2020':'2020-11'].values
    seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)
    seq = np.nan_to_num(seq) #具有NaN的点换成0
    return seq

threshold_LR=0.0001 #拟合阈值
learning_rate_LR=0.001 #迭代步长
#数据分测试集与训练集
rate_division=0.75 #75%的数据用作训练集
LR_data=load_data()
date_num,vector_num=np.shape(LR_data)
temp_num=int(date_num*rate_division)
train_set=LR_data[0:temp_num]
test_set=LR_data[temp_num:date_num]
#数据分测试集与训练集结束
weight_vector=np.ones(vector_num-1)
date_num_train,vector_num_train=np.shape(train_set)
for i_train in range(date_num_train):
    prediction=sum(train_set[i_train,1:vector_num]*weight_vector)
    loss_train=train_set[i_train,0]-prediction
    if loss_train<threshold_LR:
        break
    weight_vector=weight_vector+learning_rate_LR*train_set[i_train,1:vector_num]*weight_vector #更新权值
    print('第%d次的训练损失为：%d'%(i_train,loss_train))
    
##Prediction
date_num_test,vector_num_test=np.shape(test_set)
test_result=np.ones(date_num_test)
for i_test in range(date_num_test):
    prediction=sum(test_set[i_test,1:vector_num]*weight_vector)
    if abs(prediction)<0.1*test_set[i_test,0]:
        test_result[i_test]=0
    print('第%d次的预测值为：%d'%(i_test,prediction))
result_set=Counter(test_result) #统计结果向量中的正确个数，0代表预测成功
correct_num=result_set[0.0]
correct_rate=correct_num/date_num_test #计算预测正确率
print('The accuracy of test set is',correct_rate,'')
