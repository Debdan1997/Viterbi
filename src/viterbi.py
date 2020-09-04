'''
This program tags given corpus of input sentences (as per a pre-specified format)
'''

import pickle 
import os
import json
import numpy as np
import sys
import math

try:
	if 'bigram' not in os.listdir('../rsc'):                                          #if bigram model is not stored already
		file_handle = open('../data/train.txt', 'r')
		word_tag_list = []
		global_list = []
		tag_counts = {}
		mapcounts = {}
		mapcounts['E']={}
		mapcounts['S']={}
		tag_counts['S']=0
		tag_counts['E']=0
		for sentence in file_handle:
			sentence = sentence.rstrip()
			word_tag_list = sentence.split('@')
			global_list.append(['.','S'])
			for word_tag in word_tag_list:
				if not word_tag:
					continue
				word, tag = word_tag.split('#')
				word = word.lower()
				if tag != "$$$":
					global_list.append([word, tag])                                   #compute the lexical counts
					if tag not in mapcounts.keys():
						mapcounts[tag] = {}
					if word not in mapcounts[tag].keys():
						mapcounts[tag][word]=1
					else:
						mapcounts[tag][word]+=1
					if tag not in tag_counts.keys():
						tag_counts[tag]=1
					else:
						tag_counts[tag]+=1
			global_list.append(['.','E'])
			tag_counts['S']+=1                                                        #compute the tag counts
			tag_counts['E']+=1

		bigram = {}                                                                   #create bigram table
		for tag1 in tag_counts.keys():
			bigram[tag1]={}
			for tag2 in tag_counts.keys():
				bigram[tag1][tag2]=0

		for i in range(1,len(global_list)):
			bigram[global_list[i-1][1]][global_list[i][1]]+=1


		for tag1 in bigram.keys():                                                    #convert bigram counts to bigram probabilities
			for tag2 in bigram[tag1].keys():
				value = bigram[tag1][tag2] / float(tag_counts[tag1])
				bigram[tag1][tag2] = value


		for tag in mapcounts.keys():                                                  #convert lexical counts to lexical probabilities
			for word in mapcounts[tag].keys():
				if mapcounts[tag][word]:
					value = mapcounts[tag][word] / float(tag_counts[tag])
					mapcounts[tag][word] = value


		pickle.dump(mapcounts, open('../rsc/mapcounts', 'wb'))                       #store the models
		pickle.dump(bigram, open('../rsc/bigram', 'wb'))
		pickle.dump(tag_counts, open('../rsc/tag_counts', 'wb'))
		#pickle.dump(global_list, open('../rsc/global_list', 'wb'))

	else:                                                                            #otherwise retrieve the models
		mapcounts = pickle.load(open('../rsc/mapcounts', 'rb'))
		bigram = pickle.load(open('../rsc/bigram', 'rb'))
		tag_counts = pickle.load(open('../rsc/tag_counts', 'rb'))
		#global_list = pickle.load(open('../rsc/global_list', 'rb'))
except:
	print("No model(s) found (missing rsc directory) or no /data/train.txt to train from")


def lexical(word, tag):                                                          #fetch lexical probabilities
	if word in mapcounts[tag].keys():
		return mapcounts[tag][word]
	else:
		return 0.00000001


tags = [key for key in tag_counts.keys() if key not in ['S','E']]


def viterbi(word_list, forward=None):                                            #viterbi implementation
	global tags
	try:
		T = len(word_list)
		seq_score = [ dict.fromkeys(tag_counts.keys(),0) for i in range(T+1) ]
		back_ptr = [ dict.fromkeys(tag_counts.keys(),"") for i in range(T+1) ]
		for key in seq_score[0].keys():
			if forward == None:
				seq_score[0][key] = lexical(word_list[0], key) * bigram['S'][key]
			else:
				seq_score[0][key] = forward[0][key] * bigram['S'][key]

		for t in range(1, T):
			for tag1 in tag_counts.keys():
				for tag2 in tag_counts.keys():
					if forward == None:
						term = seq_score[t-1][tag2] * bigram[tag2][tag1] * lexical(word_list[t], tag1)
					else:
						term = seq_score[t-1][tag2] * bigram[tag2][tag1] * forward[t][tag1]
					if seq_score[t][tag1] <= term:
						seq_score[t][tag1] = term
						back_ptr[t][tag1] = tag2

		for tag2 in tag_counts.keys():
			if forward == None:
				term = seq_score[T-1][tag2] * bigram[tag2]['E']
			if seq_score[T]['E'] <= term:
				seq_score[T]['E'] = term
				back_ptr[T]['E'] = tag2

		tag_seq = ["" for i in range(T+1)]
		tag_seq[T] = 'E'
		for i in range(T - 1, -1, -1):
			tag_seq[i] = back_ptr[i + 1][tag_seq[i + 1]]

		return tag_seq[:-1]

	except:
		return ""


def get_words_and_tags(sentence):                                                    #extracting words and tags given a sentence
	words = []
	tags = []
	try:
		if sentence != "\n":
			sentence = sentence.rstrip()
			word_tag_list = sentence.split('@')
			for word_tag in word_tag_list:
				word, tag = word_tag.split('#')
				word = word.lower()
				words.append(word)
				tags.append(tag)
		return words,tags
	except:
		return "",""


def get_predicted_tags(words):
	return viterbi(words)


def get_accuracy_scores(file_path):

	file_handle = open(file_path,'r')                                        #calculating and dumping precision measures   

	global tags
	matrix = dict.fromkeys(tag_counts)
	output = dict.fromkeys(tags)
	for keys in matrix.keys():
		matrix[keys]=[0,0,0]
	for keys in output.keys():
		output[keys]={"Precision":0,"Recall":0,"F-Score":0}

	given_tags = []
	predicted_tags = []
	for sentence in file_handle:
		words, actual_tags = get_words_and_tags(sentence)
		if actual_tags == "":
			continue
		if len(words) != 0:
			output_tags = get_predicted_tags(words)
			if output_tags == "":
				continue
			given_tags.append(actual_tags)
			predicted_tags.append(output_tags)

	try:
		for sentence_tags in zip(given_tags, predicted_tags):
			for ind in range(len(sentence_tags[0])):
				matrix[sentence_tags[0][ind]][0]+=1
				matrix[sentence_tags[1][ind]][1]+=1
				if sentence_tags[0][ind] == sentence_tags[1][ind]:
					matrix[sentence_tags[1][ind]][2]+=1
	except:
		print("Please check sanity of input file")

	del matrix['S']
	del matrix['E']
	for key in matrix.keys():
		if matrix[key][0] == 0:
			precision = float('nan')
		else:
			precision = matrix[key][2]/matrix[key][0]
		if matrix[key][1] == 0:
			recall = float('nan')
		else:
			recall = matrix[key][2]/matrix[key][1]			
		output[key]["Precision"]=round(precision,5)
		output[key]["Recall"]=round(recall,5)
		if not (math.isnan(recall) or math.isnan(precision)):
			output[key]["F-Score"]=round((2*(precision*recall)/(precision+recall)),5)
		else:
			output[key]["F-Score"]=float('nan')
	
	file = open('output.txt','w')
	text = "Precision, Recall and F-Score measures for various POS tags are as reported below. Please do note that the measures for the 'X' tag are relatively low. This can be attributed to the fact that there may be several new words belonging to that category which were not found in the training data.\nSome of the below entries might reflect as 'NaN'. This could be due to two reasons - the POS tag doesn't occur in the test corpus at all; or the model didn't tag any word under that category.\n\n"
	file.write(text)
	json.dump(output,file,separators=('\n', ' = '),indent=4)

	file_handle.close()


get_accuracy_scores(sys.argv[1])