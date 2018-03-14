#!/usr/bin/env python3
#-*-coding=utf-8-*-  
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import re
import jieba
from sklearn.externals import joblib
import codecs
random.seed(2016)

def proba(lr,Vectorizer,question,answer):
	question_len = question.split()
	answer_len = answer.split()
	cont = 0 
	degree = 0
	for w in answer_len:
		if w in question_len:
			cont += 1
			degree = cont*1.0/len(answer_len)
	question=Vectorizer.transform([question]).toarray()
	answer=Vectorizer.transform([answer]).toarray()
	x=np.hstack((question*answer,question+answer))
	x=np.hstack((x,np.array([[degree,len(answer_len),len(question_len),len(answer_len)-len(question_len)]])))

	return lr.predict_proba(x)[0][1]
	


def read_data(file_question,file_answer,file_label=None,file_index=None):
	question=[]
	answer=[]
	label=[]
	index=[]
	with codecs.open(file_question,'r','utf-8') as f:
		for line in f:
				question.append(line.strip())
	with codecs.open(file_answer,'r','utf-8') as f:
		for line in f:
				answer.append(line.strip())
	if file_label is not None:
		with codecs.open(file_label,'r','utf-8') as f:
			for line in f:
					label.append(int(line.strip()))
	else:
		for line in question:
			label.append(None)
	if file_index is not None:
		with codecs.open(file_index,'r','utf-8') as f:
			for line in f:
					index.append(int(line.strip()))
	else:
		for line in question:
			index.append(None)

	data=[]
	for line in zip(question,answer,label,index):
		data.append(line)
	return data			

#TFIDF+Cosine模型没有训练，只有验证，5fold
#参考文献：http://www.ruanyifeng.com/blog/2013/03/tf-idf.html  http://www.ruanyifeng.com/blog/2013/03/cosine_similarity.html
def train(file_question,file_answer,file_label,file_index):
	#读数据
	data=read_data(file_question,file_answer,file_label,file_index)
	# question_index=[]
	# for line in data:
	# 	question_index.append(line[-1])
	# question_index=list(set(question_index))
	# avg_score=[]
	# #交叉验证，5fold
	# random.shuffle(question_index)
	# for i in range(5):
	# 	#获得第ifold的验证下标
	# 	index=[0]*1000000
	# 	for j in question_index[int(i*len(question_index)/5):int((i+1)*len(question_index)/5)]:
	# 		index[j]=1
	# 	#获得训练和验证数据
	# 	train_data=[]
	# 	dev_data=[]
	# 	train_fit_data=[]
	# 	for line in data:
	# 		if index[line[-1]]==0:
	# 			train_data.append(line)
	# 			train_fit_data.append(line[0])
	# 			train_fit_data.append(line[1])
	# 		else:
	# 			dev_data.append(line)
		

	# 	vectorizer=TfidfVectorizer(max_features=2000,norm=None)
	# 	vectorizer.fit_transform(train_fit_data)
	# 	lr = LogisticRegression(C=10000,verbose=1,max_iter=1000)
	# 	question=[]
	# 	answer=[]
	# 	label=[]
	# 	for line in train_data:
	# 		question.append(line[0])
	# 		answer.append(line[1])
	# 		label.append(line[2])
	# 	question=vectorizer.transform(question).toarray()
	# 	answer=vectorizer.transform(answer).toarray()
	# 	X=np.hstack((question*answer,question+answer))

	# 	Y=label
	# 	lr.fit(X,Y)

	# 	#随机模型没有训练，所以直接随机分数
	# 	with codecs.open('model/dev_'+str(i)+'_output.txt','w') as f1,codecs.open('model/dev_'+str(i)+'_index.txt','w') as f2,codecs.open('model/dev_'+str(i)+'_label.txt','w') as f3:
	# 		for line in dev_data:
	# 			f1.write(str(proba(lr,vectorizer,line[0],line[1]))+'\n')
	# 			f2.write(str(line[3])+'\n')
	# 			f3.write(str(line[2])+'\n')
	# 	#计算MRR
	# 	s=score('model/dev_'+str(i)+'_output.txt','model/dev_'+str(i)+'_index.txt','model/dev_'+str(i)+'_label.txt')
	# 	avg_score.append(s)
	# 	print("Fold "+str(i)+" done!")
	# 	print(s)
	# print("Dev MRR: "+str(avg_score))
	# print("Mean Dev MRR: "+str(sum(avg_score)/5))

	train_fit_data=[]
	for line in data:
		train_fit_data.append(line[0])
		train_fit_data.append(line[1])		
	vectorizer=TfidfVectorizer(max_features=1000,norm=None)
	vectorizer.fit_transform(train_fit_data)
	#lr = LogisticRegression(C=10000,verbose=1,max_iter=1000)
	# question=[]
	# answer=[]
	# label=[]
	# for line in data:
	# 	question.append(line[0])
	# 	answer.append(line[1])
	# 	label.append(line[2])
	# question=vectorizer.transform(question).toarray()  
	# answer=vectorizer.transform(answer).toarray()
	# X=np.hstack((question*answer,question+answer))

	# f=[]
	# for line in data:
	# 	question=line[0].split()
	# 	answer=line[1].split()
	# 	cont=0
	# 	degree=0
	# 	for w in answer:
	# 		if w in question:
	# 			cont+=1
	# 		degree=cont*1.0/len(answer)
	# 	f.append([degree,len(answer),len(question),len(answer)-len(question)])
	# X=np.hstack((X,np.array(f)))
	# Y=label
	#lr.fit(X,Y)
	#joblib.dump(lr,'/Users/Shared/graduation_project/paper/webshow/show/search/model/mymodel.pkl')
	lr = joblib.load('/Users/Shared/graduation_project/paper/webshow/show/search/model/mymodel.pkl')
	return lr,vectorizer



def inference(lr,vectorizer,file_question,file_answer,file_infer):
	data=read_data(file_question,file_answer)
	score_list = []
	score = ""
	count = 0
	#随机模型没有训练模型，所以直接随机分数
	with codecs.open(file_infer,'w','utf-8') as f:
		for line in data:
			score = proba(lr,vectorizer,line[0],line[1])
			f.write(str(score)+'\n')
			score_list.append(score) 
	return score_list

def score(file_infer,file_index,file_label):
	output=[]
	index=[]
	label=[]
	with codecs.open(file_infer,'r') as f1,codecs.open(file_index,'r') as f2,codecs.open(file_label,'r') as f3:
		for line in zip(f1,f2,f3):
			output.append(float(line[0]))
			index.append(int(line[1]))
			label.append(int(line[2]))

	index_score={}
	for i in index:
		index_score[i]=[]
	for line in zip(output,index):
		index_score[line[1]].append(line[0])
	rank=[]
	for line in zip(label,index,output):
		if line[0]==1:
			score=line[2]
			score_list=index_score[line[1]]
			lager_num=0
			equal_num=0
			for n in score_list:
				if n>score:
					lager_num+=1
				elif n==score:
					equal_num+=1
			rank.append(random.randint(lager_num+1,lager_num+equal_num))
	score=sum([float(1.0/r) for r in rank])/len(rank)
	return score


def cut(content):
    content_cut = list(jieba.cut(content, cut_all=False))
    content_cut=[word.encode('utf-8') for word in content_cut]  
    punctuations = ['','\n','\t',',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '、', '，', '。', '：', '；', '？', '（', '）', '【', '】', '！', '#', '￥']   
    content_cut = [str(x) for x in content_cut if x not in punctuations]
    return ' '.join(content_cut)


def data_get(question,article):
	print article
	lines = re.split('？|。|; |, |\n|\?|\.',article.encode('utf-8'))
	print lines
	while '' in lines:
		lines.remove('')
	question_cut = cut(question)
	questionlast = ''
	answerlast= ''
	f1 = open('/Users/Shared/graduation_project/paper/webshow/show/search/data/question.txt','w') 
	f2 = open('/Users/Shared/graduation_project/paper/webshow/show/search/data/answer.txt','w')
	for line in lines:
		f1.write(question_cut+'\n')
		questionlast += question_cut+'\n'
		line_cut = cut(line)
		f2.write(line_cut+'\n')
		answerlast += line_cut+'\n'
	return questionlast,answerlast,lines




def run(input_question,input_article):
	lr,vectorizer=train("/Users/Shared/graduation_project/paper/webshow/show/search/data/train_question_cut.txt","/Users/Shared/graduation_project/paper/webshow/show/search/data/train_answer_cut.txt","/Users/Shared/graduation_project/paper/webshow/show/search/data/train_label.txt","/Users/Shared/graduation_project/paper/webshow/show/search/data/train_index.txt")
	question = ""
	answer = ""
	answers = ""
	score_list = []
	print input_question
	print input_article
	f = open ('/Users/Shared/graduation_project/paper/webshow/show/search/data/answers_here.txt','w')
	question,answer,answers = data_get(input_question,input_article)
	print answers
	mapp = []
	count = 0

	score_list =  inference(lr,vectorizer,'/Users/Shared/graduation_project/paper/webshow/show/search/data/question.txt','/Users/Shared/graduation_project/paper/webshow/show/search/data/answer.txt',"/Users/Shared/graduation_project/paper/webshow/show/search/model/infer_output.txt")

	for i in answers:
		mapp.append((score_list[count],i))
		count += 1
	print mapp

	mapp = sorted(mapp)
	mapp.reverse()
	for i in mapp:
		print i[0],i[1]
		f.write(str(i[0])+"\t"+i[1]+'\n')
	#print answers,score_list
	#test_score=score("model/infer_output.txt","../data/dev_index.txt","../data/dev_label.txt")
	#print("Test MRR: "+str(test_score))




#run("在校全日制本科生多少人？","华南师范大学（South China Normal University），简称“华师”，坐落于广州市，是广东省人民政府和教育部共建高校，入选国家“双一流”世界一流学科建设高校、首批“211工程”、国家“111计划”[2]  、“卓越教师培养计划”、广东省高水平大学整体建设高校、广东省重点大学，是中国100所首批联入CERNET和INTERNET网的高等院校之一。华南师范大学始建于1933年，前身是当代著名教育家林砺儒先生创建的广东省立勷勤大学师范学院；1982年10月，易名为华南师范大学；2006年，学校通过“十五”“211工程”建设整体验收。2004年，原中共中央总书记、国家主席胡锦涛出席澳门回归五周年庆典期间，称该校是中国数家名牌师范大学之一。教育家罗浚、汪德亮、五四新诗开创者之一康白情、古代文学家李镜池、古汉语学家吴三立、历史学家王越、逻辑学家李匡武、心理学家阮镜清、教育学家叶佩华、朱勃，数学家叶述武，物理学家黄友谋、刘颂豪等先后在此执教。截至2017年6月，学校有广州石牌、广州大学城和佛山南海3个校区，占地面积3025亩，校舍面积155万平方米，图书374万册。下设25个二级学院，拥有84个本科专业；有专任教师1979人；有在校全日制本科生24894人，硕士研究生7553人，博士研究生842人，博士后在站98人，留学生1019人。2017年5月23日，华南师范大学与韩山师院、岭南师院、韶关学院等签署《教育硕士联合培养框架协议》。2017年9月，入围一流学科建设高校名单。")
