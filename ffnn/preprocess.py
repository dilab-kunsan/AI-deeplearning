#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
import fasttext
import gensim

data_dir = 'data/'
total_data = 2000
categories = {u'건강복지':0, u'통일북한':1, u'교육':2, u'경제':3, u'문화':4, u'과학':5, u'미분류':6, u'지방':7, u'국제':8, u'사회':9, u'산업':10, u'복지':11, u'정치':12, u'스포츠':13, u'환경자원':14}

def vectorize(text, vocab={}):
    opts = [i for i in text.split(' ') if len(i)>0]

    cidx = 0
    tmp = []
    while cidx<len(opts):
        c0 = opts[cidx]
        if cidx+1<len(opts):
            c1 = opts[cidx+1]
        else:
            c1 = False
        if cidx+2<len(opts):
            c2 = opts[cidx+2]
        else:
            c2 = False
        if c2:
            s = c0+'_'+c1+'_'+c2
            if s in vocab:
                tmp.append(vocab[s])
                cidx+=3
                continue
        else:
            pass
        if c1:
            s = c0+'_'+c1
            if s in vocab:
                tmp.append(vocab[s])
                cidx+=2
                continue
        else:
            pass
        if c0 in vocab:
            tmp.append(vocab[c0])
        elif c0.lower() in vocab:
            tmp.append(vocab[c0.lower()])
        else:
            tmp.append(vocab['</s>'])
            pass
        cidx+=1
    return tmp

def fasttext_w2v(w2v_data, data):
	w2v_fn = data_dir+'input.txt'
	with open(w2v_fn,'a') as f:
		sents = []
		for idx , d in enumerate(w2v_data):
			sent = []
			morphs = d.split(' ')
			for m in morphs:
				f.write(m+' ')
				sent.append(m)
			sents.append(sent)
			f.write('\n')
	model = fasttext.skipgram(w2v_fn,data_dir+'model',lr=0.1,dim=300,ws=3,min_count=1)
	print 'Word2Vec model finish!'
	w2v = gensim.models.KeyedVectors.load_word2vec_format(data_dir+'model.vec')
	vocab = {}
	for i in range(len(w2v.index2word)):
		vocab[w2v.index2word[i]] = i
	tmp = []
	maxlen = 0
	pad = 0
	for d in data:
		body = d[0]
		v = vectorize(body, vocab=vocab)
		if len(v) > maxlen:
			maxlen = len(v)
		tmp.append(v)
	print 'max length :', maxlen	
	e = np.zeros((len(data), maxlen)).astype('int32')
	e.fill(pad)

	for i in range(len(tmp)):
		v = tmp[i]
		e[i, :len(v)] = v
	
	td = open(data_dir+'train_data.txt','w')
	tl = open(data_dir+'train_label.txt','w')
	sd = open(data_dir+'test_data.txt','w')
	sl = open(data_dir+'test_label.txt','w')
	idx = len(data)/5
	
	for i, (vec, d) in enumerate(zip(e, data)):
		label  = d[1]
		vec = vec.tolist()
		if i>idx:
			for v in vec:
				td.write(str(v)+' ')
			td.write('\n')
			tl.write(str(label)+'\n')
		else:
			for v in vec:
				sd.write(str(v)+' ')
			sd.write('\n')
			sl.write(str(label)+'\n')
	
	td.close()
	tl.close()
	sd.close()
	sl.close()	

from konlpy.tag import Kkma, Twitter
def nlp(data):
	twi = Twitter()
	kkm = Kkma()
	data_nlp = []
	w2v_data = []
	err=0
	for idx, d in enumerate(data):
		if idx % 100 == 0:
			print idx
		try:
			body = d[0]
			art_label = d[1]
			body = str(unicode(body))
			sents = kkm.sentences(body)
			bod = []
			for sent in sents:
				sent = twi.morphs(sent)
				'''
				for s in sent:
					print '\t', s
				'''
				bd = ' '.join(sent)
				bod.append(bd)
			bo = '\n'.join(bod)
			data_nlp.append([bo, art_label])
			w2v_data.append(bo)
		except UnicodeError:
			err+=1
	print 'NLP finish!'
	print '\tshape : ', len(data_nlp)
	fasttext_w2v(w2v_data, data_nlp)
	return data_nlp

import re
import codecs
def preprocess(text):
	text = text.replace('\\','').replace('“','\"').replace('”','\"').replace('‘','\'').replace('’','\'').replace("[","").replace("]","").replace('｢','').replace('｣','').replace('『','').replace('』','').replace('･','')
	hangul = re.compile('[^ \,\.\!\?\"\'a-z0-9ㄱ-ㅣ가-힣]+') # 위와 동일
	result = hangul.sub('', text)
	
	return result

'''
0 : id
2 : body
7 : ???
'''
import csv
def process(i):
	f_name = data_dir+'news_20160%s_article.txt' % i
	print f_name

	pre_arts = []
	with open(f_name, "rb") as news:
		lines = news.readlines()
		for idx, line in enumerate(lines):
			if idx==total_data:
				break
			data = line.split('\t')
			art_body = data[2]
			art_label = categories[unicode(data[7]).strip()]
			art_body = preprocess(art_body)
			if art_body is None or len(art_body)<=10:
				continue
			pre_arts.append([art_body, art_label])
	print 'Preprocessing finish!'
	print '\tshape : ', len(pre_arts)
	nlp_arts = nlp(pre_arts)

if __name__ == '__main__':
	l = [1]
	for i in l:
		process(i)
