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
#files = ["f11.txt", "f12.txt", "f13.txt"]  # test data
files.sort(key=natural_keys)
files_count = len(files)

# initiate the spark
conf = SparkConf()
sc = SparkContext(conf=conf)

# get the stopwords and punctuations
f = open(STOPWORDS_FILE, "r")
stopwords = f.read().splitlines()
punctuations = ["", "'", '"']

# RDDs for every document
termfs = []
unique_words_docs = []
tfidfs_list = []

"""
Step 1: Compute TF of every word in a document
The code reads the file one by one. For each file, an RDD with key-value pair of (word, freq)
is created. All RDDs are appended in a list termfs
word: word
freq: the occurrence of the word in the doc
"""
for file in files:
	# Step 1: Compute TF of every word in a document
	# Get the TF and append the RDD in a list
	lines = sc.textFile(INPUT_DIR + "/" + file)
	#words = lines.flatMap(lambda l: re.split(r'[^\w]+',l.lower()))  # doesnt consider apostrophe
	words = lines.flatMap(lambda l: split_line(l.lower()))  # consider apostrophe
	words = words.filter(lambda w: w not in stopwords and w not in punctuations)
	pairs = words.map(lambda w: (w, 1))
	counts = pairs.reduceByKey(lambda n1, n2: n1 + n2)
	counts.persist()
	termfs.append(counts)

	# Get the unique words and append the RDD in a list
	uniques = pairs.distinct()
	uniques.persist()
	unique_words_docs.append(uniques)

# put all the unique words in the docf
unique_words_rdd = sc.union(unique_words_docs)
docf = unique_words_rdd.reduceByKey(lambda n1, n2: n1 + n2)

"""
Step 2 and step 3: Compute the normalized TF-IDF of every word for every document
The RDDs are computed separately
The RDDs are appended in tfidfs_doc
"""
tfidfs_doc = []
i = 0  # counter to map to the txt file. It's ugly, I know..
os.system("rm -r rdd")
for termf in termfs:
	temp = termf.join(docf)
	tfidf = temp.map(lambda w: (w[0], (1 + math.log(w[1][0]))*math.log(files_count/w[1][1])))
	mag = math.sqrt(tfidf.map(lambda w: (w[0], w[1]*w[1])).values().sum())
	tfidf_norm = tfidf.map(lambda w: (w[0], w[1]/mag))
	tfidf_norm.persist()
	tfidfs_doc.append(tfidf_norm)
	# save the RDD
	tfidf_norm.saveAsSequenceFile("rdd/" + files[i][:-4]);
	i += 1

# save the RDD for the docf
docf.saveAsSequenceFile("rdd/docf")

"""
Step 4: Compute the relevance of each document w.r.t a query
"""
f = open("query.txt", "r").read().lower()
#f = "south park is truly the work of a genius thank you sir parker"  # test data
query = re.split(r'[^\w]+', f)
query_rdd = sc.parallelize(query)
query_rdd = query_rdd.filter(lambda w: w not in stopwords and w not in punctuations)
query_rdd = query_rdd.map(lambda w: (w, 1)).distinct()

# normalize the query_rdd
denominator = math.sqrt(query_rdd.map(lambda w: (w[0], w[1]*w[1])).values().sum())
query_rdd = query_rdd.map(lambda w: (w[0], w[1]/denominator))

relevance_list = []
for tfidf in tfidfs_doc:
	query_rdd_tfidf = query_rdd.join(tfidf)
	elem_mul = query_rdd_tfidf.map(lambda w: (w[0], w[1][0]*w[1][1]))
	elem_mul.persist()
	relevance = elem_mul.values().sum()
	relevance_list.append(relevance)

"""
Step 5: Sort and save the top 10 documents
"""
n = 10
result = sc.parallelize(list(zip(files, relevance_list)))
result = result.sortBy(lambda a: -a[1])
result = sc.parallelize(result.take(n))
os.system("rm -r outfile")
result.repartition(1).saveAsTextFile("outfile")
sc.stop()

