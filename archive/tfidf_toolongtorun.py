# Python version: 3.7.6
# Package used: pyspark, specifically pyspark.sql

import os, re, sys, math
from pyspark import SparkConf, SparkContext

# helper function
def atoi(text):
	return int(text) if text.isdigit() else text

def natural_keys(text):
	return [ atoi(c) for c in re.split('(\d+)', text)]

def split_line(line):
    return re.findall('[A-Za-z\']+(?:\`[A-Za-z]+)?',line)

INPUT_DIR = "lab2 input"
STOPWORDS_FILE = "stopwords.txt"
QUERY_FILE = "query.txt"

# scan the directory and put all files in the list
files = os.listdir(INPUT_DIR)
#files = ["f11.txt", "f12.txt", "f13.txt"]
#files = ["sample1.txt", "sample2.txt"] # for lightweight purpose
files.sort(key=natural_keys)
files_count = len(files)

# initiate the spark
conf = SparkConf()
sc = SparkContext(conf=conf)



# get the stopwords and punctuations
# CHECK: Should we modify the stopwords? It does not look optimal
# "theyre" is there but "they're" isnt, for example
f = open(STOPWORDS_FILE, "r")
stopwords = f.read().splitlines()
punctuations = ["", "'", '"']

"""
Step 1: Compute TF of every word in a document
The code reads the file one by one. For each file, an RDD with key-value pair of (word, freq)
is created. Then, another RDD with key-value pair of ((doc, word), freq) is generated
doc: name of file
word: word
freq: the occurrence of the word in the doc
"""
termfs = []
for file in files:
	# Get the TF and append the RDD in a list
	lines = sc.textFile(INPUT_DIR + "/" + file)
	#words = lines.flatMap(lambda l: re.split(r'[^\w]+',l.lower()))  # doesnt consider apostrophe
	words = lines.flatMap(lambda l: split_line(l.lower()))  # consider apostrophe
	words = words.filter(lambda l: l not in stopwords and l not in punctuations)
	pairs = words.map(lambda l: (l, 1))
	counts = pairs.reduceByKey(lambda l1, l2: l1 + l2)
	counts_doc = counts.map(lambda l: ((file, l[0]), l[1]))
	counts_doc.persist()
	termfs.append(counts_doc)

# put the TFs in a single RDD
termf_rdd = sc.union(termfs)

# Step 2 & 3 : Compute normalized TF-IDF
"""
DF for every word (docf_rdd) is created by doing mapreduce on termf_rdd.
For every file, the termf_rdd is filtered to only retrieve data for that file.
After changing the key-value to be (word, freq), it is joined with docf_rdd to make rdd of
(word, )
"""
# Get the unique words and append the RDD in a list.
unique_words_rdd = termf_rdd.map(lambda l: (l[0][1], 1))
docf_rdd = unique_words_rdd.reduceByKey(lambda n1, n2: n1 + n2)
docf_rdds = []
for file in files:
	temp = docf_rdd.map(lambda l: ((file, l[0]), l[1]))
	temp.persist()
	docf_rdds.append(temp)
docf_rdd_dup = sc.union(docf_rdds)

tf_df = termf_rdd.join(docf_rdd_dup)
tfidf = tf_df.map(lambda l: (l[0], (1 + math.log(l[1][0]))*math.log(files_count/l[1][1])))

# Step 3
tfidf_sq = tfidf.map(lambda l: (l[0], l[1]*l[1]))
mag_dict = {}
for file in files:
	mag = math.sqrt(tfidf_sq.filter(lambda l: l[0][0] == file).values().sum())
	mag_dict[file] = mag
tfidf_norm = tfidf.map(lambda l: (l[0], l[1]/mag_dict[l[0][0]]))

# Step 4
f = open("query.txt", "r").read().lower()
#f = "my name is ghifari i work in micron i study in nus"
#f = "south park is truly the work of a genius thank you sir parker"
query = re.split(r'[^\w]+', f)
query_rdd = sc.parallelize(query).map(lambda l: (l, 1)).distinct()
# normalize the query_rdd
query_mag = math.sqrt(query_rdd.map(lambda l: (l[0], l[1]*l[1])).values().sum())
query_rdd = query_rdd.map(lambda w: (w[0], w[1]/query_mag))

# for element in query_rdd.collect():
# 	print(element)
# exit()

relevance_list = []
for file in files:
	query_doc = query_rdd.map(lambda l: ((file, l[0]), l[1]))
	query_rdd_tfidf = query_doc.join(tfidf_norm)
	elem_mul = query_rdd_tfidf.map(lambda w: (w[0], w[1][0]*w[1][1]))
	elem_mul.persist()
	relevance = elem_mul.values().sum()
	relevance_list.append(relevance)

result = sc.parallelize(list(zip(files, relevance_list)))
result = result.sortBy(lambda a: -a[1])
result.repartition(1).saveAsTextFile("outfile") #TODO: Handle 'file already exists' exception
sc.stop()

