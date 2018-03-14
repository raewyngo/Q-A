#!/usr/bin/env python3
#-*-coding=utf-8-*-  
import jieba
def cut(content):
    content_cut = list(jieba.cut(content, cut_all=False))
    content_cut=[word.encode('utf-8') for word in content_cut]  
    punctuations = ['','\n','\t',',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '、', '，', '。', '：', '；', '？', '（', '）', '【', '】', '！', '#', '￥']   
    content_cut = [str(x) for x in content_cut if x not in punctuations]
    return ' '.join(content_cut)

train_question=[]
train_answer=[]
train_index=[]
train_label=[]
dev_question=[]
dev_answer=[]
dev_index=[]
dev_label=[]

q=""
cont=0
with open('raw_data/BoP2017-DBQA.txt','r') as f,open('train_question.txt','w') as f1,open('train_answer.txt','w') as f2,open('train_index.txt','w') as f3,open('train_label.txt','w') as f4:
    for line in f:
        line=line.strip().split('\t')
        label=line[0]
        question=' '.join(line[1:-1])
        answer=line[-1]
        if question!=q:
            q=question
            cont+=1
        f1.write(question+'\n')
        f2.write(answer+'\n')
        f3.write(str(cont)+'\n')
        f4.write(label[-1]+'\n')

with open('raw_data/BoP2017-DBQA.dev.txt','r') as f,open('dev_question.txt','w') as f1,open('dev_answer.txt','w') as f2,open('dev_index.txt','w') as f3,open('dev_label.txt','w') as f4:
    for line in f:
        line=line.strip().split('\t')
        label=line[0]
        question=' '.join(line[1:-1])
        answer=line[-1]
        if question!=q:
            q=question
            cont+=1
        f1.write(question+'\n')
        f2.write(answer+'\n')
        f3.write(str(cont)+'\n')
        f4.write(label[-1]+'\n')



train_question=[]
train_answer=[]
train_index=[]
train_label=[]
dev_question=[]
dev_answer=[]
dev_index=[]
dev_label=[]

q=""
cont=0
with open('raw_data/BoP2017-DBQA.txt','r') as f,open('train_question_cut.txt','w') as f1,open('train_answer_cut.txt','w') as f2,open('train_index.txt','w') as f3,open('train_label.txt','w') as f4:
    for line in f:
        line=line.strip().split('\t')
        label=line[0]
        question=' '.join(line[1:-1])
        answer=line[-1]
        if question!=q:
            q=question
            cont+=1
        f1.write(cut(question)+'\n')
        f2.write(cut(answer)+'\n')
        f3.write(str(cont)+'\n')
        f4.write(label[-1]+'\n')

with open('raw_data/BoP2017-DBQA.dev.txt','r') as f,open('dev_question_cut.txt','w') as f1,open('dev_answer_cut.txt','w') as f2,open('dev_index.txt','w') as f3,open('dev_label.txt','w') as f4:
    for line in f:
        line=line.strip().split('\t')
        label=line[0]
        question=' '.join(line[1:-1])
        answer=line[-1]
        if question!=q:
            q=question
            cont+=1
        f1.write(cut(question)+'\n')
        f2.write(cut(answer)+'\n')
        f3.write(str(cont)+'\n')
        f4.write(label[-1]+'\n')


