# coding: utf-8

"""
this module is used to augament training data
1、随机复制一个字
2、随机去掉一个字
3、随机交换两个字
4、用同义词替换一个词
"""

import os, re, sys
import jieba as jb
import numpy as np
import random
import pickle
import copy as cp

reload(sys)
sys.setdefaultencoding('utf8')

# 同义词典，可以自定义添加
homo_file = 'new_homo.txt' 


def loadHomoDict(homo_file):
	homoDict = {}
	with open(homo_file, 'r') as f:
		for line in f.readlines():
			line = line.strip().split()
			if len(line) != 2:
				continue
			word, homo_word = line
			word = word.decode('utf8')
			homo_word = homo_word.decode('utf8')
			if homo_word not in homoDict:
				homoDict[homo_word] = []
				homoDict[homo_word].append(word)
			else:
				homoDict[homo_word].append(word)
			if word not in homoDict:
				homoDict[word] = []
				homoDict[word].append(homo_word)
			else:
				homoDict[word].append(homo_word)
	return homoDict
homoDict = loadHomoDict(homo_file)

"""
随机复制一个字
"""
def randCopyWord(s):
	if len(s) >= 2:
		idx = random.randint(0, len(s)-1)
		s = list(s)
		s.insert(idx, s[idx])
		return ''.join(s)
	else:
		return s*2

"""
随机去掉一个字
"""
def randDelteWord(s):
	if len(s) >= 2:
		idx = random.randint(0, len(s)-1)
		#s = list(s)
		del s[idx]
		return ''.join(s)
	return ''.join(s)


"""
随机交换两个字／词
"""
def randSwapWord(s):
	s = list(s)
	if len(s) >= 2:
		idx = random.randint(0, len(s)-2)
		s[idx], s[idx+1] = s[idx+1], s[idx]
		return ''.join(s)

	return ''.join(s)

"""
同义词替换
"""
def replaceWithHomophone(segList):
	new_res = []
	for i, seg in enumerate(segList):
		if seg in homoDict:
			segTmp = segList
			homoList = homoDict[seg]
			idx = random.randint(0, len(homoList)-1)
			randHomo = homoList[idx]
			#print randHomo, segTmp[i]
			segTmp[i] = randHomo
			new_res.append(''.join(segTmp))
	return new_res


def entity_augament(entity):
	augament_type = random.randint(0,1)
	if augament_type:
		#随机交换两个字／词
		entity_seg = list(jb.cut(entity, cut_all=False))
		# print randSwapWord(entity_seg)
		return randSwapWord(entity_seg)
	else:
		#同义词替换
		entity_seg = list(jb.cut(entity, cut_all=False))
		if len(entity_seg) >= 2:
			new_res = replaceWithHomophone(entity_seg)
			if new_res:
				return new_res[random.randint(0,len(new_res)-1)]
		return entity

		'''
			#随机去掉一个字
			entity3 = line[random_index_list[2]].decode('utf8')
			entity3_seg = list(jb.cut(entity3, cut_all=False))
			new_entity3_del = randDelteWord(entity3_seg)
			outline = cp.copy(line)
			outline[random_index_list[2]] = new_entity3_del
			fn.write(','.join(outline) + '\n')
		'''

def newReadFile(filename, new_augamention_file,stop_line_num):
	fn = open(new_augamention_file, 'w')
	line_num = 0
	with open(filename, 'r') as f:
		for line in f.readlines():
			line = line.strip().split(',')
			if line_num >= stop_line_num :
				break
			#手术、疾病全增强
			outline1 = cp.copy(line)
			for index,item in enumerate(line):
				if (item != '') and (index % 3 == 0):
					outline1[index] = entity_augament(item)
			# print outline1
			fn.write(','.join(outline1) + '\n')
			line_num += 1
	fn.close()



'''
随机打乱输出数据横行的顺序
'''
def data_random(infile,outfile):
	data = list()
	with open(infile, 'r') as f:
		for line in f.readlines():
			data.append(line)

	random.shuffle(data)
	fn = open(outfile, 'w')
	for line in data:
		fn.write(line)
	fn.close()

'''
merge data
'''
def merge_data(in_file,out_file):
	out = open(out_file, 'a')
	with open(in_file, 'r') as f:
		for line in f.readlines():
			out.write(line)
	out.close()


if __name__ == '__main__':
	in_file = 'data/data_negative/neg_0.7.txt'
	stop_line_num = 18000
	newReadFile(in_file, 'neg_0.3_argud.txt',stop_line_num)
	merge_data('neg_0.3_argud.txt',in_file)
	#data_random(in_file,out_file)
