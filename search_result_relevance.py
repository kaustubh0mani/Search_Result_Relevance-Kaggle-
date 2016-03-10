
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import math

def func(text):
    regex = re.compile('[^a-z A-Z]')
    #First parameter is the replacement, second parameter is your input string
    return regex.sub('', text)

def cleanhtml(raw_html):
    return re.sub('(http://\S+|\S*[^\w\s]\S*)','',raw_html)

def stopword(text):
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    text = pattern.sub('', text)
    return text

def word_filter(words):
    filtered = []
    for i in words:
        if len(i)>2:
            filtered.append(i)
    return filtered


train = pd.read_csv('train.csv').fillna(" ")
temp = np.array(train.product_description)
temp4 = np.array(train['query'])
temp5 = np.array(train.product_title)
for i in range(len(temp)):
    temp[i]+= temp5[i]
    temp[i]+= temp4[i]

temp2 = []

for i in temp:
     temp2.append(cleanhtml(i))

for i in range(len(temp2)):
    temp2[i] = func(temp2[i])
    temp2[i] = temp2[i].lower()
    temp2[i] = stopword(temp2[i])

train.product_description = temp2
tf_list = []
check = []
word_name = []
temp3 = word_filter(temp2)
for i in temp3:
    words = []
    words =  i.split(" ")
    temp_uniq_words = list(set(words))
    uniq_words = []
    for s in temp_uniq_words:
        if len(s)>2:
            uniq_words.append(s)
            check.append(s)
    for j in uniq_words:
        DocSum = 0 #Number of times term t appears in a document
        for k in words:
                if(j == k):
                    DocSum+=1
        word_name.append(j)
        tf_list.append(float(DocSum)/len(word_filter(words)))

word_name = np.array(word_name)
idf_list = []
for i in word_name:
    DocSum = 0
    for j in temp3:
        words = []
        words =  j.split(" ")
        temp_uniq_words = list(set(words))
        uniq_words = []
        for s in temp_uniq_words:
            if len(s)>2:
                uniq_words.append(s)
        for k in uniq_words:
            if(i == k):
                DocSum+=1
    idf_list.append(math.log(float(len(temp3)/DocSum)))
