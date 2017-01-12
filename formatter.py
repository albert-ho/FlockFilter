import re
import csv
import random

# please set the DIR_PATH to where the files are.

DIR_PATH = "Data/"

RAW_FILE = DIR_PATH + "MelaniaTrumpStorm.csv"
LABEL_FILE = DIR_PATH + "MelaniaTrumpLabels.csv"
TRAIN_FILE = DIR_PATH + "MelaniaTrumpTrain.csv"
TEST_FILE  = DIR_PATH + "MelaniaTrumpTest.csv"
SORT_FILE  = DIR_PATH + "MelaniaTrumpSort.csv"

'''
RAW_FILE   = DIR_PATH + "KaepernickStorm.csv"
LABEL_FILE = DIR_PATH + "KaepernickLabels.csv"
TRAIN_FILE = DIR_PATH + "KaepernickTrain.csv"
TEST_FILE  = DIR_PATH + "KaepernickTest.csv"
SORT_FILE  = DIR_PATH + "KaepernickSort.csv"
'''
########


def readCsv(fname, skipFirst=False, delimiter=";"):
    reader = csv.reader(open(fname,"rb"),delimiter=delimiter)
    rows = []
    count = 1
    for row in reader:
        if not skipFirst or count > 1:      
            rows.append(row)
        count += 1
    return rows


def write_csv(x,filename,x_format):
    wtr = open(filename,"w+")
    for i in range(len(x)):
		wtr.write(x_format(x[i]))
		wtr.write("\n")
    wtr.close()
    

def raw_tweet_format(tweet):
	[id_str, epoch, date_time, username] = tweet[:4]
	text = "".join(tweet[4:])
	return "2;{}".format(text)
	

def labeled_tweet_format(tweet):
	label = tweet[0]
	text = "".join(tweet[1:])
	return "{};{}".format(label, text)
    

def sample(tweets, n_samples):
	return random.sample(tweets, n_samples)


if __name__=="__main__":
	
	'''
    tweets = readCsv(RAW_FILE)
    tweet_sample = sample(tweets, 3000)
    write_csv(tweet_sample, LABEL_FILE, raw_tweet_format)
    '''
    
	labeled_tweets = readCsv(LABEL_FILE)
	labeled_tweets = [t for t in labeled_tweets if (int(t[0]) != 2)]
	n = 500
	tweets = sample(labeled_tweets, n)
	train_set = tweets[:(n/25)]
	test_set = tweets[(n/25):]
	write_csv(train_set, TRAIN_FILE, labeled_tweet_format)
	write_csv(test_set, TEST_FILE, labeled_tweet_format)
