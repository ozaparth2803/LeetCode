#!/usr/bin/python
import sys
import string
import re
#import zipimport
#importer = zipimport.zipimporter('nltk.mod')
#nltk = importer.load_module('nltk') 
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
stop_words=["you've", 'when', 'them', 'wasn', 've', 'she', 'her', 'from', 'further', 'or', 'there', 'am', 'isn', 'again', 'being', 'does', 'myself', 'most', 'hasn', "haven't", 'he', "didn't", 'against', 'wouldn', 'down', 'how', 'under', 'this', 'same', 'their', 'above', 'couldn', 'ma', 'me', 'each', 'the', 'we', 'just', 'these', "isn't", 'too', "aren't", 'itself', 'you', 'all', "shouldn't", 'his', 're', 'such', 'has', "wasn't", "mustn't", 'who', 'your', 'herself', 'between', 'our', 'off', 'it', 'few', 'than', 'won', 'ain', 'have', 'they', 'i', 'only', 'as', "you're", 'before', 'once', 'which', 'didn', 'y', 'are', 'himself', "don't", 'having', 'ourselves', 'own', 'and', 'in', 'to', 'don', 'was', 'haven', 'then', 's', 'themselves', 'until', 'were', 'yours', 'for', 'd', "shan't", 'so', "weren't", "she's", "needn't", "wouldn't", 'yourself', "you'd", 'be', 'doing', 'weren', "couldn't", "it's", 'ours', 'here', 'll', 'o', 'other', 'into', 'that', 'should', 'at', 'more', 'mustn', 'up', 'whom', 'him', 'on', 'because', 'of', "doesn't", 'if', 'very', 'is', 'through', 'after', 'any', 'with', 'no', 'those', 't', 'my', 'doesn', 'shouldn', 'can', 'm', 'hers', 'what', 'over', 'now', "you'll", 'while', 'shan', 'do', 'had', "hasn't", "that'll", "hadn't", 'a', 'where', 'its', 'during', 'hadn', 'aren', 'by', 'not', 'did', 'nor', 'both', 'below', 'yourselves', 'some', 'theirs', 'why', 'out', 'about', 'mightn', "should've", "won't", 'been', 'will', "mightn't", 'but', 'an', 'needn','this','says','like','dont']
value=1
for line in sys.stdin:
	line = line.strip()
	keys = line.split()
	for key in keys:
		key=re.sub('\W',' ',key)
		key=key.replace(" ","")
            #for p in string.punctuation:
                #key = key.replace(p, "")
                #value = 1

            	if (key)!="" and len(key) > 3 and key not in stop_words and 'https' not in key and key.isdigit() is False:
			for char in key:
				if char.isdigit() is True:
					key=key.replace(char,"")
                	print( "%s\t%d" % (key.lower(), value) )
			#else:	
				#print( "%s\t%d" % (key.lower(), value) )

