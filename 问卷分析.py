# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 20:20:21 2018

@author: Chris
"""
import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
from math import * 
plt.rcParams[ 'font.sans-serif'] = [ 'Microsoft YaHei']
plt.rcParams[ 'axes.unicode_minus'] = False

def stopwordsandsynonyms():
    '''哈工大停用词表导入'''
    stopwords = set()
    data =  open(r'.\src\allstopwords',encoding ='utf-8').readlines()
    for line in data:
        stopwords.update(set(line.lower().strip().split()))
    stopwords = list(stopwords)
    stopwords.append(' ')
    
    '''同近义词词表导入，后面发现部分近义词替换会产生歧义，故导入后未使用同近义词替换'''
    file = open('.\src\chinese_synonym.txt',encoding ='utf-8').readlines()
    synonyms=[]
    for line in file:
        synonyms.extend([line.strip().split()])
    return  stopwords,synonyms

'''获得数据1_五个人分别在三个批次删除的APP及其删除原因(delreason)、当初下载的原因(dwonreason)'''
def data1_get(): #用字典存储5*3个sheets
    global paths 
    paths = []
    for i in range(1,6):
        paths.append(r'.\src\问卷 ({}).xlsx'.format(i))
    data1={}
    sheets = ['Sheet1','Sheet2','Sheet3']
    for i in range(len(paths)): # 5 xlsxs
        for j in range(len(sheets)): # 3sheets
            df = pd.read_excel(paths[i],sheet_name=sheets[j],index_col=0)
            df.columns=['delreason','downreason']
            df=df.iloc[:-1]  #放弃最后一行数据（总结性原因）
            data1[(i+1)*10+j+1]=df # 把15个sheet逐个存储到字典中
    return data1
        
''' 纵向分析（调查五个人各自的情况） '''
def data1_analysis_people(data1): 
    # 设立一个新的DataFrame，存储分析的结果
    df = pd.DataFrame(np.arange(10).reshape(5,2))
    df.columns=['delreason','downreason']
    df.index=range(1,6)
    
    # 删除APP的原因分析
    for i in range(5): #5个人
        delkeywords = []
        delstr=''
        for j in range(3): #3个批次
            delstr+=''.join(data1[(i+1)*10+j+1]['delreason']).strip()
        ls=jieba.lcut(delstr) #对每个批次的删除原因拼接成字符串并用jieba分词
        delkeywords.extend([i for i in ls if i not in stopwords]) #关键词法对分词结果中，所在停用词表的词去掉
        dict={} #设立字典统计词频并排序
        for word in delkeywords:
            dict[word]=dict.get(word,0)+1
        ls = list(dict.items())
        ls.sort(key = lambda x:x[1],reverse=True) 
    
    # 当初下载APP的原因分析        
        downkeywords = []
        downstr=''
        for j in range(3): #3个批次
            downstr+=''.join(data1[(i+1)*10+j+1]['downreason']).strip()
        lt=jieba.lcut(downstr)
        downkeywords.extend([i for i in lt if i not in stopwords])
        dict={}
        for word in downkeywords:
            dict[word]=dict.get(word,0)+1
        lt = list(dict.items())
        lt.sort(key = lambda x:x[1],reverse=True)  
    
        df['delreason'][i+1]=ls
        df['downreason'][i+1]=lt
        
    df.to_excel(r'.\results\del_people.xlsx')
    return df

''' 横向分析（调查三个批次各自的情况） '''
'''句法同函数data1_analysis_people(data1)，故不作多余注释'''
def data1_analysis_turns(data1):
    df = pd.DataFrame(np.arange(6).reshape(3,2))
    df.columns=['delreason','downreason']
    df.index=range(1,4)
    for j in range(3): #3个批次
        delkeywords = []
        delstr=''
        for i in range(5): #5个人
            delstr+=''.join(data1[(i+1)*10+j+1]['delreason']).strip()
        ls=jieba.lcut(delstr)
        delkeywords.extend([i for i in ls if i not in stopwords])
        dict={}
        for word in delkeywords:
            dict[word]=dict.get(word,0)+1
        ls = list(dict.items())
        ls.sort(key = lambda x:x[1],reverse=True) 
        
        downkeywords = []
        downstr=''
        for i in range(5): #5个人
            downstr+=''.join(data1[(i+1)*10+j+1]['downreason']).strip()
        lt=jieba.lcut(downstr)
        downkeywords.extend([i for i in lt if i not in stopwords])
        dict={}
        for word in downkeywords:
            dict[word]=dict.get(word,0)+1
        lt = list(dict.items())
        lt.sort(key = lambda x:x[1],reverse=True)  
    
        df['delreason'][j+1]=ls
        df['downreason'][j+1]=lt
        
    df.to_excel(r'.\results\del_turns.xlsx')
    return df

'''获得数据2_“打死也不删除的APP”及其保留的原因（keepreason）和当初下载的原因（downreason）'''
def data2_get():
    sheet = 'Sheet4' 
    data2={} #同样用字典存储
    for i in range(len(paths)):
        df = pd.read_excel(paths[i],sheet_name=sheet,index_col=0)
        df.columns=['keepreason','downreason']
        data2[i]=df
    return data2

'''统计保留app的情况'''
def data2_analysis(data2):    
    # 对五份问卷选择保留的APP及其数量进行统计，并画图
    appdict = {}
    lt=[]
    for i in range(len(data2)): 
        for word in data2[i].index:
            lt.append(word)
            appdict[word]=appdict.get(word,0)+1 
    
    plt.figure().set_figwidth(12)
    X=range(len(appdict))
    Y=appdict.values()
    plt.bar(X,Y,facecolor='lightskyblue',edgecolor = 'k')
    plt.title('保留的APP数量') #图表标题
    plt.xlabel("APPs") #x轴名称
    plt.ylabel('保留的数量') #y轴名称
    plt.xticks(range(len(appdict)),list(appdict.keys()))
    plt.savefig(r'.\results\keepapp.jpg')
    plt.show()
    
    # 利用pd.DataFrame统计该函数分析出来的结果
    df2 = pd.DataFrame.from_dict(appdict,orient='index')
    df2.columns=['number']
    df2['keepreason'] = df2['downreason'] =df2['keepkeywords'] = df2['downkeywords']= None
    
    
    # 统计各app保留的原因
    appkeep={} #先用字典存储数据，再传到df2['keepreason']中
    for i in appdict.keys():
        appkeep[i]=[]
    for i in range(len(data2)): #11个apps
        for word in data2[i].index:
            appkeep[word].append(data2[i]['keepreason'][word])
            df2['keepreason'][word]=appkeep[word]
    
    # 统计各app当初下载的原因
    appdown={}  #先用字典存储数据，再传到df2['downreason']中
    for i in appdict.keys():
        appdown[i]=[]
    for i in range(len(data2)): #11个apps
        for word in data2[i].index:
            appdown[word].append(data2[i]['downreason'][word])
            df2['downreason'][word]=appdown[word]
            
    # 对两种原因提取关键词，方法类似：整合成字符串，jieba分词，淘汰停用词
    for word in appdict.keys():
        lt=[]
        keepkeywords=[]
        #addkeywords=[]
        dict = {}
        keepreason=','.join(str for str in appkeep[word])
        lt=jieba.lcut(keepreason)
        keepkeywords.extend([i for i in lt if i not in stopwords])
        
# =============================================================================
#       同近义词替换，发现效果并不好
#         for keyword in keepkeywords:
#             for line in synonyms:
#                  if keyword in line:
#                     print(keyword,line[0])
#                     try:
#                         keepkeywords.remove(keyword)
#                         addkeywords.append(line[0])
#                         break
#                     except:
#                         pass
# =============================================================================
                
        for keyword in keepkeywords:        
            dict[keyword]=dict.get(keyword,0)+1
        ls = list(dict.items())
        ls.sort(key = lambda x:x[1],reverse=True)
        df2['keepkeywords'][word] = ls
    
    for word in appdict.keys():
        downkeywords=[]
        lt=[]
        dict = {}
        downreason=','.join(str for str in appdown[word])
        lt=jieba.lcut(downreason)
        downkeywords.extend([i for i in lt if i not in stopwords])
        
        for keyword in downkeywords:        
            dict[keyword]=dict.get(keyword,0)+1
        ls = list(dict.items())
        ls.sort(key = lambda x:x[1],reverse=True)
        df2['downkeywords'][word] = ls
    
    df2.to_excel(r'.\results\keepapps.xlsx')
    return df2

'''获取数据3_五个人针对给出的16可能的删除原因赋予自己心目中的权值'''
def data3_get():
    data3 = pd.read_excel('.\src\删除权值.xlsx',index_col=0) 
    data3 = data3.drop(data3.index[12]) #删除无效行
    for i in range(len(data3.index)):
        for j in range(len(data3.columns)):
            data3.values[i][j] = data3.values[i][j] / sum(data3[data3.columns[j]]) #归一化

    return data3

def data3_analysis(data3):
    dataT=data3.T
    weight=cal_weight(dataT) #利用熵值法计算数据规约的结果
    weight.index=data3.index
    weight.columns=['weight']
    weight = weight.sort_values(by='weight',ascending=False) #排序
    weight.to_excel(r'.\results\delreason_weight.xlsx')
    return weight
    

'''定义熵值法函数'''
'''转载自CSDN“好吃的鱿鱼”，url：https://blog.csdn.net/qq_24975309/article/details/82026022'''
def cal_weight(x):  #传入的是dataframe，传出的是DataFrame
    # 求k
    rows = x.index.size  # 行
    cols = x.columns.size  # 列
    k = 1.0 / log(rows)
 
    lnf = [[None] * cols for i in range(rows)]
 
    # 矩阵计算--
    # 信息熵
    # p=array(p)
    x = np.array(x)
    lnf = [[None] * cols for i in range(rows)]
    lnf = np.array(lnf)
    for i in range(0, rows):
        for j in range(0, cols):
            if x[i][j] == 0:
                lnfij = 0.0
            else:
                p = x[i][j] / x.sum(axis=0)[j]
                lnfij = log(p) * p * (-k)
            lnf[i][j] = lnfij
    lnf = pd.DataFrame(lnf)
    E = lnf
 
    # 计算冗余度
    d = 1 - E.sum(axis=0)
    # 计算各指标的权重
    w = [[None] * 1 for i in range(cols)]
    for j in range(0, cols):
        wj = d[j] / sum(d)
        w[j] = wj
        # 计算各样本的综合得分,用最原始的数据
    
    w = pd.DataFrame(w)
    return w

    
    
if __name__ == "__main__":
    '''按顺序执行以上函数'''
    stopwords,synonyms=stopwordsandsynonyms()
    data1 = data1_get()
    data1_people = data1_analysis_people(data1)
    data1_turns = data1_analysis_turns(data1)
    data2=data2_get()
    df2=data2_analysis(data2)
    data3 = data3_get()
    weight = data3_analysis(data3)
    
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   