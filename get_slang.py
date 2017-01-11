import urllib,urllib2,json,re,datetime,sys,cookielib
from pyquery import PyQuery
import lxml

from bs4 import BeautifulSoup
import json

########

DIR_PATH = "Data/"
SLANG_FILE = DIR_PATH + "slang.json"

########


def dictify(dl, slang_dict):
	for dt in dl.find_all("dt"):
		if not isinstance(dt.contents[0], basestring):
			continue
		key = dt.contents[0].strip(" :")
		slang_dict[key] = dt.parent.findNext("dd").contents[0]
	return slang_dict


def getResponse(letter):
	cookieJar = cookielib.CookieJar()
	url = "https://noslang.com/dictionary/{}".format(letter)

	headers = [
		('Host', "twitter.com"),
		('User-Agent', "Mozilla/5.0 (Windows NT 6.1; Win64; x64)"),
		('Accept', "application/json, text/javascript, */*; q=0.01"),
		('Accept-Language', "de,en-US;q=0.7,en;q=0.3"),
		('X-Requested-With', "XMLHttpRequest"),
		('Referer', url),
		('Connection', "keep-alive")
	]

	opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cookieJar))
	opener.addheaders = headers

	try:
		response = opener.open(url)
		html_doc = response.read()
		
	except:
		print "NoSlang weird response. Try to see on browser: {}".format(url)
		sys.exit()
		return
	
	soup = BeautifulSoup(html_doc, 'html.parser')
	dl = soup.find('dl')
	return dl


def readJson(filename):
    with open(filename, 'r') as f:
        d = json.load(f, 'utf8')
    return d


def writeJson(d, filename):
    with open(filename, 'w+') as f:
        json.dump(d, f)


if __name__ == '__main__':
	dl = getResponse('t')
	d = readJson(SLANG_FILE)
	new_d = dictify(dl, d)
	writeJson(new_d, SLANG_FILE)
