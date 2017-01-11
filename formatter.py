import re
import csv
import random

# please set the DIR_PATH to where the files are.

DIR_PATH = "./"

SRC_FILE   = DIR_PATH + "KaepernickStorm.csv"
DST_FILE   = DIR_PATH + "KaepernickLabels.csv"

########


def readCsv(fname, skipFirst=True, delimiter=";"):
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
    

def tweet_format(tweet):
	[id_str, epoch, date_time, username] = tweet[:4]
	text = "".join(tweet[4:])
	return ";{};{}".format(id_str, text)
    

def sample(tweets, n_samples=1000):
	return random.sample(tweets, n_samples)


if __name__=="__main__":
    tweets = readCsv(SRC_FILE, False)
    tweet_sample = sample(tweets, 1000)
    write_csv(tweet_sample, DST_FILE, tweet_format)
    
