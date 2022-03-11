# -*- coding: utf-8 -*-
"""
Created on Sun May 17 20:42:41 2020

@author: SHUAI BAO
"""
import pandas as pd
import random #Import shuffle function
import numpy as np
from collections import  Counter
import tushare as ts
from datetime import datetime, date
import pylab
import yfinance as yf
from collections import  Counter
from sklearn.linear_model import LinearRegression
import time
import matplotlib.pyplot as plt
##Define subfunction part
token = ''
ts.set_token(token)
code='600489'
start_date = '2012-10-12'
end_date = '2020-12-01' #雅虎财经接口专用日期
start_date_guozhai='20121012'
end_date_guozhai='20211201'#tushare接口专用日期
pro = ts.pro_api()
Df_gold_price = yf.download('GLD',start_date,end_date) #下载雅虎财经黄金日内数据
data_gold_price=Df_gold_price.values #雅虎财经黄金日内数据转换成list数据
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
Df_tushare=pd.concat([df_tycr, df_us_tbr,df_us_tltr,df_us_trltr],axis=1) #将所有的数据归到同一个矩阵中
#Df_GLD=pd.concat([Df_gold_price['Open'],Df_DINIW_price['Open'],Df_IRX,Df_FVX['Open']],axis=1)
#数据预处理：所有的黄金数据都按照时间偏移往前挪N天，N=1：20，随后训练时候做1：20天的预测
days_shift=1 #偏移天数
Df_gold_price_shift=Df_gold_price.shift(-days_shift) #时间偏移将数据往前挪N
Df_GLD=pd.concat([Df_tushare['ltc'],Df_tushare['cmt'],Df_tushare['e_factor'],Df_tushare['ltr_avg'],Df_DINIW_price['Open'],Df_IRX['Open'],Df_FVX['Open'],Df_TNX['Open'],Df_TYX['Open'],Df_VIX['Open'],Df_gold_price_shift['Open']],axis=1)
Df_GLD = Df_GLD.fillna(0) #具有NaN的点换成0
    

def load_data(subset_num):
    seq=Df_GLD.values
    data_num,data_vector=np.shape(seq)
    label_seq=np.mat(np.zeros(data_num,))
    for i_num in range(1,data_num):
        if seq[i_num-1,10]-seq[i_num,10]<0:
            label_seq[0,i_num]=4 #0代表价格升高
        else:
            label_seq[0,i_num]=2 #1代表价格降低
    seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)
    seq = np.nan_to_num(seq) #具有NaN的点换成0
    return seq


#Sample block
def split_sample(divide_num,sample_mat):
    sample_shape=np.shape(sample_mat) #Take out the matrix size in sample
    sample_num=sample_shape[0] #Take out the number of samples in the sample, and assign the value to sample
    divide_range=int(sample_num/divide_num) #Divide sample in verage, round down
    a=int(divide_range*(divide_num-1)) #Take out the lower bound of sample
    b=a+ divide_range#Take the upper bound of sample
    train_set=sample_mat[0:a]
    test_set=sample_mat[a:b]
    return train_set,test_set
#End of sample block
   
#Augment vector part:
def Augment_feature(augmenting_data):
    (sample_num,sample_dim)=np.shape(augmenting_data)
    ones=np.ones(sample_num) #Construct row vector 1
    ones_column =ones.reshape(-1, 1) #Convert row vector to column vector
    augmented_data=np.column_stack((ones_column,augmenting_data[:,0:9],augmenting_data[:,9])) #Add a column 1 to the right of the matrix
    return augmented_data
#End of enhancement vector part
    
    
#Linear scaling part, the function name is scale ﹣ linear
#Because the data obtained here is in str format, 
#the exception value will be output at the beginning of index. When calculating formula, STR format needs to be converted to number format
def  Scale_linearly(subset):
    dimensional_vector=subset[:,0:10] 
    #Here, there is a class column by default when inputting subset. The class column is in the 11th column, and the 0 column is the enhancement vector 1
    sample_shape=np.shape(dimensional_vector) #Take out the matrix size in sample
    sample_dimension=sample_shape[1] #Take out the number of dimensions in the sample
    for j in range(sample_dimension): 
        vector=dimensional_vector[:,j]
        #vector=[int(i) for i in vector_str] #Convert STR data to number type
        min_j=min(vector)#Find min value in vector
        min_j=float(min_j)
        max_j=max(vector)#Find max value in vector
        max_j=float(max_j)
        vector=np.mat(vector)
        dimensional_vector[:,j]=2*(vector-min_j+10e-6)/(max_j-min_j+10e-6)-1
    Xij=np.hstack((dimensional_vector,subset[:,10:11])) 
    #The returned data contains the combination of 10 dimensions after linear scaling and the last class column, a total of 11 columns
    return Xij
#End of linear scaling section
    
#Reset sample vector x according to label y
def Reset_example_vector(subset):
    subset_num_mat=np.shape(subset)
    subset_num=subset_num_mat[0]
    Xij=subset
    for k in range(subset_num):
        if subset[k,10]==4: #k=ln defined as 1
            Xij[k,10]=1
        else:
            Xij[k,10]=0 #k!=ln defined as 0
    x=Xij
    return x
#End of Reset sample vector x according to label y
    
#定义tanh（）函数
def tanh(x):
    Xmat = np.mat(x, dtype=float) #将mat数据后面的参数定义为float，不然会计算报错
    s1 = np.exp(Xmat) - np.exp(-Xmat)
    s2 = np.exp(Xmat) + np.exp(-Xmat)
    s = s1 / s2
    return s
#tanh()函数定义结束
    
#定义隐层函数,weight_w_input是本层权重矩阵,input_xi是底层节点值向量，hidden_node_vector是本层节点值向量，
def hidden_layer(input_xi,weight_w_input): 
    #输入层，下一层隐藏层的神经元节点数为H=60
    (hidden_node_num,input_dimension)=np.shape(weight_w_input) #取出权重矩阵的节点数与输入向量维度
    hidden_node_vector=np.zeros(hidden_node_num) #初始化本层节点值向量
    input_xi=np.array(input_xi) #把数据格式转换成array以便于后续相乘   
    for node_num_i in range(hidden_node_num):#计算本层节点序列的向量
        w_i=weight_w_input[node_num_i] #取出第node_num_i个权重向量，计算下一层第node_num_i个节点的值
        input_weight_mat=input_xi*w_i #计算输入层与权重向量的元素乘积得到input_weight_mat
        input_weight_num=np.sum(input_weight_mat) #计算输入层乘积和，input_weight_num为输入层与权重向量乘积之和
        input_weight_num_activate=tanh(input_weight_num) #输入层激活，激活函数为tanh()
        hidden_node_vector[node_num_i]=input_weight_num_activate #计算节点值并赋值到本层点向量中
    #hidden_node_sum=np.sum(hidden_node_vector) #所有节点向量值求和
    return hidden_node_vector #输出本层节点值向量
#隐层函数定义结束

#定义BP函数,weight_w_now_layer为当前层的权重向量，delta_k 为总计算误差，默认计算函数为tanh()，now_layer_node_num为底层的计算和,也就是当前节点的数值
def BP_error(weight_w_now_layer,delta_k,now_layer_node_num):#定义BP反向传播计算误差值
    weight_error_mat=weight_w_now_layer*delta_k
    weight_error_sum=np.sum(weight_error_mat,axis=1) #按每一行求和，求完之后是列向量
    deuction_delta=1-now_layer_node_num**2
    BP_delta_j=(deuction_delta)*weight_error_sum
    return BP_delta_j
#BP函数定义结束

#定义更新权重值向量函数
#xi是底一层节点值向量，delta_layer_1是顶层每个节点的误差向量，是一个数列，eta_learning_rate是学习率，自定义设置的超参数
#previous_weight_vector是未更新前的权重向量矩阵
def update_weight_vector(xi,delta_layer_1,eta_learning_rate,previous_weight_vector):
    delta_layer_1=np.mat(delta_layer_1)
    [layer1_col,layer1_node_num]=np.shape(delta_layer_1)
    xi_dimension=np.size(xi)
    new_weight_vector=np.zeros((layer1_node_num,xi_dimension))
    for layer1_node_num_i in range(layer1_node_num): #在每个节点偏差值基础上更新此节点对应的底层权重向量
        layer1_delta_i=delta_layer_1[0,layer1_node_num_i] #取出layer1层第layer1_node_num_i个节点的误差
        new_weight_vector[layer1_node_num_i]=previous_weight_vector[layer1_node_num_i]-eta_learning_rate*layer1_delta_i*xi
    return  new_weight_vector
#定义更新权重值向量函数结束
    
def forward_calculate(input_xi_label,weight_w_input,weight_w_1,weight_w_output):
    #前向计算开始
    #weight_w_input=weight_wij_input_temp #输入层权重向量
    #weight_w_1=weight_wij_2_temp #隐层第一层输出到第二层的权重向量
    #weight_w_output=weight_wij_output_temp #隐层到输出层权重向量
    input_xi=input_xi_label[0:10] #取出样本中0到10个数作为数据向量，第11个数作为结果判断，x矩阵为输入
    #前向计算开始
    #输入层，下一层隐藏层的神经元节点数为H=9
    (hidden_node_num,input_dimension)=np.shape(weight_w_input) #取出权重矩阵的节点数与输入向量维度
    
    input_xi=np.array(input_xi) #把数据格式转换成array以便于后续相乘   
    
    for node_num_i in range(hidden_node_num):#计算下一层节点序列的向量
        w_i=weight_w_input[node_num_i] #取出第node_num_i个权重向量，计算下一层第node_num_i个节点的值
        input_weight_mat=input_xi*w_i #计算输入层与权重向量的元素乘积得到input_weight_mat
        input_weight_num=np.sum(input_weight_mat) #计算输入层乘积和，input_weight_num为输入层与权重向量乘积之和
        input_weight_num_activate=tanh(input_weight_num) #输入层激活，激活函数为tanh()
        hidden_node_vector[node_num_i]=input_weight_num_activate #计算节点值并赋值到下一节点向量中
    hidden_layer_1_node_num=hidden_node_vector #将下一层节点值更新,激活函数为tanh()
    #输入层结束
    
    #隐层第一层
    #输入计算,input_weight_num_activate作为隐层第一层输入向量,hidden_layer_num1为第一层隐层输出向量
    hidden_layer_num1_activate=hidden_layer(hidden_layer_1_node_num,weight_w_1)
    #hidden_layer_num1_activate=tanh(hidden_layer_num1) #隐层第一层激活，激活函数为tanh()
    #隐层第一层结束
    
    #隐层第二层
    #输入计算,hidden_layer_num1_activate作为隐层第二层输入向量,hidden_layer_2_node_num为第二层隐层输出向量
    hidden_layer_2_node_num=hidden_layer_num1_activate
    hidden_layer_num2_activate=hidden_layer(hidden_layer_2_node_num,weight_w_output)
    #hidden_layer_num2_activate=tanh(hidden_layer_num2) #隐层第二层激活，激活函数为tanh()
    #隐层第二层结束
    #输出层
    reult_yk=hidden_layer_num2_activate #得到输出result_yk
    #前向计算结束
    delta_k=reult_yk-input_xi_label[10]
    return reult_yk,delta_k
    #前向计算结束
def load_data():
    token = '4166534b2fed0241aec0cfd025bcc577fe8053aef9202e88740213ec'
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
    Df_GLD=pd.concat([Df_tushare['ltc'],Df_tushare['cmt'],Df_tushare['e_factor'],Df_tushare['ltr_avg'],Df_DINIW_price['Open'],Df_IRX['Open'],Df_FVX['Open'],Df_TNX['Open'],Df_TYX['Open'],Df_VIX['Open'],Df_gold_price_shift['Open']],axis=1)
    Df_GLD = Df_GLD.fillna(0) #具有NaN的点换成0
    
    seq=Df_GLD.values
    data_num,data_vector=np.shape(seq)
    label_seq=np.mat(np.zeros(data_num,))
    for i_num in range(1,data_num):
        if seq[i_num-1,10]-seq[i_num,10]<0:
            label_seq[0,i_num]=4 #0代表价格升高
        else:
            label_seq[0,i_num]=2 #1代表价格降低
    seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)
    seq = np.nan_to_num(seq) #具有NaN的点换成0
    return seq
BP_data=load_data()

def run_BP():
    #Import data
    #df = pd.read_csv(r'C:\Users\SHUAI BAO\OneDrive\2020年\西交利物浦大学\EEE418高级特征识别Advanced Pattern Recognition\Lab\lab5-neural-network\breast-cancer-wisconsin.data') # 读取数据集
    df=BP_data
    #When using the import statement, add r before the path to indicate that the string is a non escaped original string
    #End of importing
        
    #problem5.2.1：Random shuffle data
    list_mat=np.mat(df) #Convert the data in DF parameter from panda.core.frame.dataframe to matrix type, and then name it list
    list=np.array(list_mat) #Mat format to array format
    list[list == '?'] = 0 #For data cleaning, the data inside is'? ' Replace the data of with 0
    (row_num,col_num)=np.shape(list)
    for mat_num in range(row_num):
        list[mat_num,6]=float(list[mat_num,6]) #Change STR data in data to num format
    list=list[:,1:11]
    random.seed(17) #Define the integer value at the beginning of the algorithm before calling the random module
    #导入后的数据命名为list
    np.random.shuffle(list) #Arrange the list data randomly. Random is only valid for list type and not for array types
    ##problem1：End of Random shuffle data
    
    #problem5.2.2：Split data
    divide_num=5 #Define the number of splitting
    (train_set,test_set)=split_sample(divide_num,list)
    #problem2：End of splitting
    
    #problem5.3.1：Augment the data
    X_bar_temp=Augment_feature(train_set)
    X_augmented=np.mat(X_bar_temp)
    #problem5.3.1：End of Augment the data5
    
    #problem5.3.2：Linearly scale
    X_data_scaling=X_bar_temp 
    #The default input data for linear scaling is 10 columns of data plus the 11th column of label, 11 columns in total, and the first column is the enhanced vector column 1
    Xij_scale=Scale_linearly(X_data_scaling) #The output data after linear scaling is 10 columns of data plus the 11th column of label, 11 columns in total
    #problem5.3.2：End of Linearly scale
    
    #problem5.3.3： Reset the example vector x according its label y
    Xij_scale_input=np.array(Xij_scale)
    x=Reset_example_vector(Xij_scale_input)
    #problem5.3.3： End of Reset the example vector x according its label y
    
    
    #初始化wij，D是输入节点数，K是输出节点数
    (sample_num,sample_dimension)= np.shape(x) #取出x矩阵的维度数与样本数
    D_inputnode=sample_dimension #输入节点为特征维度数量
    K_outputnode=1 #输出节点数为1
    random_area=np.sqrt(6/(D_inputnode+K_outputnode+1)) #随机数范围计算
    #初始化权重向量wij矩阵
    H=63 #隐层节点数
    weight_wij_input=-random_area + 2*random_area*np.random.random((H,sample_dimension-1)) #用随机数产生wij1，初始化为60*10的矩阵，60为节点数，10为输入数据的维度数
    weight_wij_2=-random_area + 2*random_area*np.random.random((H,H)) #用随机数产生wij2，初始化为矩阵，隐层1与隐层2之间的链接
    weight_wij_output=-random_area + 2*random_area*np.random.random((1,H)) #随机数产生隐层与输出层之间的权重矩阵,权重为向量
    #用随机数产生wij2，初始化为数列
    #weight_wij_2=np.array(weight_wij_2) #把数据格式转换成array以便于后续相乘 
    weight_wij_input_temp=weight_wij_input #初始化随后更新节点的暂存值
    weight_wij_2_temp=weight_wij_2
    weight_wij_output_temp=weight_wij_output
    hidden_node_vector=np.zeros(H) #初始化隐层节点向量
    #初始化结束
    
    #BP更新节点权重,x为输入值，隐层层数网络为2
    for sample_i in range(sample_num):
        weight_w_input=weight_wij_input_temp #输入层权重向量
        weight_w_1=weight_wij_2_temp #隐层第一层输出到第二层的权重向量
        weight_w_output=weight_wij_output_temp #隐层到输出层权重向量
        
        #前向计算开始
        #输入层，下一层隐藏层的神经元节点数为H=9
        (hidden_node_num,input_dimension)=np.shape(weight_w_input) #取出权重矩阵的节点数与输入向量维度
        input_xi=x[sample_i,0:10] #取出样本中0到10个数作为数据向量，第11个数作为结果判断，x矩阵为输入
        input_xi=np.array(input_xi) #把数据格式转换成array以便于后续相乘   
        
        for node_num_i in range(hidden_node_num):#计算下一层节点序列的向量
            w_i=weight_w_input[node_num_i] #取出第node_num_i个权重向量，计算下一层第node_num_i个节点的值
            input_weight_mat=input_xi*w_i #计算输入层与权重向量的元素乘积得到input_weight_mat
            input_weight_num=np.sum(input_weight_mat) #计算输入层乘积和，input_weight_num为输入层与权重向量乘积之和
            input_weight_num_activate=tanh(input_weight_num) #输入层激活，激活函数为tanh()
            hidden_node_vector[node_num_i]=input_weight_num_activate #计算节点值并赋值到下一节点向量中
        hidden_layer_1_node_num=hidden_node_vector #将下一层节点值更新,激活函数为tanh()
        #输入层结束
        
        #隐层第一层
        #输入计算,input_weight_num_activate作为隐层第一层输入向量,hidden_layer_num1为第一层隐层输出向量
        hidden_layer_num1_activate=hidden_layer(hidden_layer_1_node_num,weight_w_1)
        #hidden_layer_num1_activate=tanh(hidden_layer_num1) #隐层第一层激活，激活函数为tanh()
        #隐层第一层结束
        
        #隐层第二层
        #输入计算,hidden_layer_num1_activate作为隐层第二层输入向量,hidden_layer_2_node_num为第二层隐层输出向量
        hidden_layer_2_node_num=hidden_layer_num1_activate
        hidden_layer_num2_activate=hidden_layer(hidden_layer_2_node_num,weight_w_output)
        #hidden_layer_num2_activate=tanh(hidden_layer_num2) #隐层第二层激活，激活函数为tanh()
        #隐层第二层结束
        
        #输出层
        
        reult_yk=hidden_layer_num2_activate #得到输出result_yk
        #前向计算结束
        
        #反向传播更新部分
        #计算每一层，每一个节点的误差值
        tk=x[sample_i,10] #取出此样本input_xi的实际判断值，位于数列最后一列，命名为tk，
        delta_k=reult_yk-tk #yk为计算出来的结果，tk为原本的结果，delta_k为总差值
        delta_k=delta_k[0] #delta_k是一个数字
        #delta_k=np.sqrt(delta_k**2)
        delta_layer_2=BP_error(weight_w_output,delta_k,hidden_layer_2_node_num) #weight_w_output是本层到下一层的权重，hidden_layer_num2_activate是底一层的输出值，delta_k总差值
        hidden_layer_1_node_num=np.array(hidden_layer_1_node_num) #转换成array格式数据以便于后续计算
        delta_layer_1=BP_error(weight_w_1,delta_k,hidden_layer_1_node_num) #第一层隐层误差向量，delta_layer_1
        #计算最接近输出层的反向传播数
        
        #更新节点权重向量
        eta_learning_rate=0.005 #自定义学习率eta，eta为超参数
        #previous_weight_vector为更新钱的权重向量
        weight_wij_input_temp=update_weight_vector(input_xi,delta_layer_1,eta_learning_rate,weight_w_input)#更新输入权重向量
        weight_wij_2_temp=update_weight_vector(hidden_layer_1_node_num,delta_layer_2,eta_learning_rate,weight_w_1)#更新隐层第一层权重向量
        delta_layer_3=np.array(tk)
        weight_wij_output_temp=weight_wij_output_temp-eta_learning_rate*delta_layer_3*hidden_layer_num2_activate
    #BP更新权重结束
        
    #测试集计算正确率
    
    X_bar_test=Augment_feature(test_set) #强化测试矩阵
    X_test_augmented=np.mat(X_bar_test) #强化矩阵矩阵化
    X_data_scaling=X_test_augmented
    X_test_scale=Scale_linearly(X_data_scaling) #线性缩放强化向量
    X_test_scale_input=np.array(X_test_scale)
    x_test=Reset_example_vector(X_test_scale_input) #线性缩放结束，x_test作为测试数据集
    
    #BP网络预测
    (test_num,test_dimension)= np.shape(x_test) #取出x_test矩阵的维度数与样本数
    result_predict=np.zeros(test_num)
    #H=9 #隐层节点数
    #weight_w_input
    #weight_w_1
    #weight_w_output
    #BP前向预测计算，x为输入值，隐层层数网络为9
    predict_delta=np.zeros(test_num) #初始化预测误差序列
    for test_i in range(test_num):
        [result_predict[test_i],predict_delta[test_i]]=forward_calculate(x_test[test_i],weight_w_input,weight_w_1,weight_w_output)
    result_predict_abs=np.sqrt(result_predict**2) #取结果序列的绝对值
    #预测计算结束
    for result_i in range(test_num):
        if result_predict_abs[result_i]>0.5:
            result_predict_abs[result_i]=1
        else:
            result_predict_abs[result_i]=0
    result_real=x_test[:,10] #取出测试集结果向量
    matched_vector=result_real-result_predict_abs #计算预测正确的结果序列，元素值为0代表预测正确
    
    #计算正确率部分
    result_set=Counter(matched_vector)
    correct_num_train=result_set[0.0]
    correct_rate=correct_num_train/test_num #计算预测正确率
    print('The accuracy of test set is',correct_rate,'')
    #计算正确率部分结束n_BP():
    return correct_rate
#Accuracy_temp=run_BP(subset_num)
date_num,vector_num=Df_gold_price.shape
Accuracy_prediction_seq=np.zeros(date_num) 
for i_date in range(8,200):
    Accuracy_prediction_temp=np.zeros(5)
    subset_num=i_date
    N =date_num
    #st = time.process_time()
    for i in range(5):
        Accuracy_temp=run_BP(subset_num)
        Accuracy_prediction_temp[i]=Accuracy_temp
        print('前面',i_date,'个数据的第',i,'次训练')
        #p = round((i + 1) * 100 / N)
        Accuracy_temp_main=np.mean(Accuracy_prediction_temp)
        Accuracy_prediction_seq[i_date]=Accuracy_temp_main

Accuracy_prediction_criterion=np.ones(Accuracy_prediction_seq[6:i_date].size)/2
x=range(Accuracy_prediction_seq[6:i_date].size)
plt.ylim(ymax = 1)
plt.title(r'$BP \ Accuracy$')
plt.ylabel('Accuracy (ACC)')
plt.xlabel('Length of data (Days)')
plt.scatter(x,Accuracy_prediction_seq[6:i_date],c = 'r',marker = '.',label='LSTM Prediction')#, color='green', label='LSTM Prediction')
plt.plot(Accuracy_prediction_criterion, color='green', label='BP Creterion')
