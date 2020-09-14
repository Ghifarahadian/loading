# tf-idf-implementation
This code implements TF-IDF algorithm to a set of text files. Then, a sentence in 'query.txt' is compared with the score of the inputs by using this algorithm to find the most similar text.

It uses pyspark RDD for the computation to allow a cluster of computers to compute the score in parallell.

The TF-IDF score is computed by following this formula:
(1 + log (TF)) * log (N/DF)

N: total documents
TF: number of word in document
DF: number of documents having the word
