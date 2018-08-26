#!/usr/bin/python

import sys

oldkey=None
count=0
word_dict={}

for line in sys.stdin:
	#line=re.sub('<.*?>', '', line)
	line = line.strip()
    	key, value = line.split("\t", 1)
    	value = int(value)

    	if key not in word_dict:
        	word_dict[key] = value
   	else:
        	word_dict[key] += value


keys = [(k, word_dict[k]) for k in sorted(word_dict, key=word_dict.get, reverse=True)]
for k, v in keys:
    print("%s\t%d" % (k, word_dict[k]))

