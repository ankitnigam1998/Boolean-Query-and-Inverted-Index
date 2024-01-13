from tqdm import tqdm
# from preprocessor import TextProcessor
# from indexer import Indexer
# from linkedlist import LinkedList
from collections import OrderedDict
import inspect as inspector
from collections import OrderedDict as CustomOrderedDict
import math
import sys
import argparse
import json
import time
import random
import flask
from flask import Flask
from flask import request
import hashlib
import copy
import collections
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

class Node:

    def __init__(self, val=0):
        self.val = val
        self.nxt = None
        self.ski = None 
        self.tf = 0.0
        self.tf_idf = 0.0
    

class LinkedList:

    def __init__(self):
        self.st_node = None
        self.end_node = None
        self.length, self.n_skips, self.idf = 0, 0, 0.0
        self.skip_length = None

    def traverse_list(self):
        traversal = []
        if self.st_node is None:
            return
        else:
            temp = self.st_node
            while temp is not None:
                traversal.append(temp.val)
                temp = temp.nxt
            return traversal

    def traverse_skips(self):
        traversal = []
        if self.st_node is None:
            return
        else:
            temp = self.st_node
            while temp is not None:
                traversal.append(temp.val)
                temp = temp.ski
            return traversal


    def add_skip_connections(self):
        n_skips = math.floor(math.sqrt(self.length))
        if n_skips * n_skips == self.length:
            n_skips = n_skips - 1

        self.n_skips = n_skips
        self.skip_len = round(math.sqrt(self.length),0)
        if self.skip_len == 1:
            return
        counter = 0
        temp1 = self.st_node
        temp2 = self.st_node
        while temp2 is not None:
            counter += 1
            temp2 = temp2.nxt
            if counter%self.skip_len == 0:
                temp1.ski = temp2
                temp1 = temp2


    def insert_at_end(self, new_node):
        if self.st_node is None:
            new_node.nxt = self.st_node
            self.st_node = new_node
        elif self.st_node.val > new_node.val:
            new_node.nxt = self.st_node
            self.st_node = new_node
        else:
            temp = self.st_node
            while temp.nxt is not None and new_node.val > temp.nxt.val:
                temp = temp.nxt
            new_node.nxt = temp.nxt
            temp.nxt = new_node
    

    def estimate_tf_idf_score(self):
        temp = self.st_node
        while temp is not None:
            temp.tf_idf = temp.tf * self.idf
            temp = temp.nxt

class Indexer:
    def __init__(self):
        self.custom_index  = CustomOrderedDict({})

    def get_index(self):
        return self.custom_index 

    def generate_inverted_index(self, document_id, tokenized_document):
        unique_tokens = set(tokenized_document)
        unique_token_list = list(unique_tokens)
        for custom_token in unique_token_list:
            token_count = tokenized_document.count(custom_token)
            term_frequency = token_count / len(tokenized_document)
            self.add_to_index(custom_token, document_id, term_frequency)

    def add_to_index(self, term, document_id, term_frequency):
        if term not in self.custom_index :
            posting_list = LinkedList()
        else:
            posting_list = self.custom_index [term]
        new_node = Node(document_id)
        new_node.term_frequency = term_frequency
        posting_list.insert_at_end(new_node)
        posting_list.length += 1
        self.custom_index [term] = posting_list

    def sort_terms(self):
        sorted_index = CustomOrderedDict({})
        for term in sorted(self.custom_index .keys()):
            sorted_index[term] = self.custom_index [term]
        self.custom_index  = sorted_index

    def add_skip_connections(self):
        for term in self.custom_index .keys():
            plist = self.custom_index [term]
            plist.add_skip_connections()

    def calculate_tf_idf(self, total_documents):
        for term in self.custom_index .keys():
            plist = self.custom_index [term]
            inverse_document_frequency = (total_documents / plist.length)
            plist.idf = inverse_document_frequency
            plist.estimate_tf_idf_score()

class TextProcessor:

    def extract_id(self, text_data):

        data_array = text_data.split("\t")
        return int(data_array[0]), data_array[1]

    def text_tokenize(self, text):

        processed_text = text.lower()
        processed_text = re.sub(r'[ ](?=[ ])|[^A-Za-z0-9 ]+', ' ', processed_text)
        processed_text = re.sub(r'\s+', ' ', processed_text)

        word_list = processed_text.split()

        stop_words = set(stopwords.words('english'))

        filtered_words_list = [word for word in word_list if word not in stop_words]

        stemmer = PorterStemmer()

        stemmed_words_list = [stemmer.stem(word) for word in filtered_words_list]
        return stemmed_words_list

app = Flask(__name__)

class ProjectRunner:
    def __init__(self):
        self.p_proc = TextProcessor()
        self.indxr = Indexer()

    def _merge(self, list1, list2):
        merged_list = LinkedList()
        counter = 0
        p1 = list1.st_node
        p2 = list2.st_node
        while p1 is not None and p2 is not None:
            counter += 1
            if p1.val == p2.val:
                new_node = copy.deepcopy(p1)
                new_node.tf_idf = max(p1.tf_idf, p2.tf_idf)
                merged_list.insert_at_end(new_node)
                p1 = p1.nxt
                p2 = p2.nxt
            elif p1.val < p2.val:
                p1 = p1.nxt
            else:
                p2 = p2.nxt
        return merged_list, counter

    def _daat_and(self, terms):
        term_posting_length = {}
        for term in terms:
            plist = self.indxr.custom_index[term]
            term_posting_length[term] = plist.length
        sorted_terms = OrderedDict(sorted(term_posting_length.items(), key=lambda val: val[1]))
        terms = list(sorted_terms.keys())
        num_terms = len(terms)
        comparisons = 0
        term0 = terms[0]
        plist0 = self.indxr.custom_index[term0]
        for i in range(1, num_terms):
            term_i = terms[i]
            plist_i = self.indxr.custom_index[term_i]
            plist0, comp = self._merge(plist0, plist_i)
            comparisons += comp
        return plist0, comparisons

    def _merge_skip(self, list1, list2):
        p1 = list1.st_node
        p2 = list2.st_node
        merged_list = LinkedList()
        counter = 0
        while p1 is not None and p2 is not None:
            counter += 1
            if p1.ski is not None and p2.val > p1.ski.val:
                p1 = p1.ski
            elif p2.ski is not None and p1.val > p2.ski.val:
                p2 = p2.ski
            elif p1.val == p2.val:
                new_node = copy.deepcopy(p1)
                new_node.tf_idf = max(p1.tf_idf, p2.tf_idf)
                merged_list.insert_at_end(new_node)
                p1 = p1.nxt
                p2 = p2.nxt
            elif p1.val < p2.val:
                p1 = p1.nxt
            else:
                p2 = p2.nxt
        return merged_list, counter

    def _daat_and_skip(self, terms):
        term_posting_length = {}
        for term in terms:
            plist = self.indxr.custom_index[term]
            term_posting_length[term] = plist.length
        sorted_terms = OrderedDict(sorted(term_posting_length.items(), key=lambda val: val[1]))
        terms = list(sorted_terms.keys())
        num_terms = len(terms)
        comparisons = 0
        term0 = terms[0]
        plist0 = self.indxr.custom_index[term0]
        for i in range(1, num_terms):
            term_i = terms[i]
            plist_i = self.indxr.custom_index[term_i]
            plist0, comp = self._merge_skip(plist0, plist_i)
            comparisons += comp
        return plist0, comparisons

    def tfidf(self, post_list):
        temp = post_list.st_node
        res = {}
        while temp is not None:
            res[temp.val] = temp.tf_idf
            temp = temp.nxt
        pairs = dict(sorted(res.items(), key=lambda it: it[1], reverse=True))
        sorted_docs = list(pairs.keys())
        return sorted_docs

    def _get_postings(self, term):
        plist = self.indxr.custom_index[term]
        return plist.traverse_list()

    def _get_postings_skips(self, term):
        plist = self.indxr.custom_index[term]
        return plist.traverse_skips()

    def _output_formatter(self, op):
        if op is None or len(op) == 0:
            return [], 0
        op_no_score = [int(i) for i in op]
        results_count = len(op_no_score)
        return op_no_score, results_count

    def run_indexer(self, corpus_file):
        total_documents = 0
        with open(corpus_file, 'r', encoding="utf8") as fp:
            for line in tqdm(fp.readlines()):
                doc_id, document = self.p_proc.extract_id(line)
                tokenized_document = self.p_proc.text_tokenize(document)
                self.indxr.generate_inverted_index(doc_id, tokenized_document)
                total_documents += 1
        self.indxr.sort_terms()
        self.indxr.add_skip_connections()
        self.indxr.calculate_tf_idf(total_documents)

    def run_queries(self, query_list):
        output_dict = {'postingsList': {},
                       'postingsListSkip': {},
                       'daatAnd': {},
                       'daatAndSkip': {},
                       'daatAndTfIdf': {},
                       'daatAndSkipTfIdf': {},
                       }

        for query in tqdm(query_list):
            input_terms = self.p_proc.text_tokenize(query)

            for term in input_terms:
                postings, skip_postings = self._get_postings(term), self._get_postings_skips(term)

                output_dict['postingsList'][term] = postings
                output_dict['postingsListSkip'][term] = skip_postings

            and_op_no_skip, and_comparisons_no_skip = self._daat_and(input_terms)
            and_op_no_skip_sorted = self.tfidf(and_op_no_skip)
            and_comparisons_no_skip_sorted = and_comparisons_no_skip

            and_op_skip, and_comparisons_skip = self._daat_and_skip(input_terms)
            and_op_skip_sorted = self.tfidf(and_op_skip)
            and_comparisons_skip_sorted = and_comparisons_skip

            and_op_no_score_no_skip, and_results_count_no_skip = self._output_formatter(and_op_no_skip.traverse_list())
            and_op_no_score_skip, and_results_count_skip = self._output_formatter(and_op_skip.traverse_list())
            and_op_no_score_no_skip_sorted, and_results_count_no_skip_sorted = self._output_formatter(
                and_op_no_skip_sorted)
            and_op_no_score_skip_sorted, and_results_count_skip_sorted = self._output_formatter(and_op_skip_sorted)

            output_dict['daatAnd'][query.strip()] = {}
            output_dict['daatAnd'][query.strip()]['results'] = and_op_no_score_no_skip
            output_dict['daatAnd'][query.strip()]['num_docs'] = and_results_count_no_skip
            output_dict['daatAnd'][query.strip()]['num_comparisons'] = and_comparisons_no_skip

            output_dict['daatAndSkip'][query.strip()] = {}
            output_dict['daatAndSkip'][query.strip()]['results'] = and_op_no_score_skip
            output_dict['daatAndSkip'][query.strip()]['num_docs'] = and_results_count_skip
            output_dict['daatAndSkip'][query.strip()]['num_comparisons'] = and_comparisons_skip

            output_dict['daatAndTfIdf'][query.strip()] = {}
            output_dict['daatAndTfIdf'][query.strip()]['results'] = and_op_no_score_no_skip_sorted
            output_dict['daatAndTfIdf'][query.strip()]['num_docs'] = and_results_count_no_skip_sorted
            output_dict['daatAndTfIdf'][query.strip()]['num_comparisons'] = and_comparisons_no_skip_sorted

            output_dict['daatAndSkipTfIdf'][query.strip()] = {}
            output_dict['daatAndSkipTfIdf'][query.strip()]['results'] = and_op_no_score_skip_sorted
            output_dict['daatAndSkipTfIdf'][query.strip()]['num_docs'] = and_results_count_skip_sorted
            output_dict['daatAndSkipTfIdf'][query.strip()]['num_comparisons'] = and_comparisons_skip_sorted

        return output_dict


@app.route("/execute_query", methods=['POST'])
def execute_query():
    start_time = time.time()
    queries = request.json["queries"]

    output_dict = runner.run_queries(queries)

    with open(output_location, 'w') as fp:
        json.dump(output_dict, fp)

    response = {
        "Response": output_dict,
        "time_taken": str(time.time() - start_time),
        "username_hash": username_hash
    }
    return flask.jsonify(response)

if __name__ == "__main__":
    output_location = "./data/postingsListOutput.json"
    corpus_file = "./data/input_corpus.txt"
    username_hash = hashlib.md5("ankitnig".encode()).hexdigest()

    runner = ProjectRunner()

    runner.run_indexer(corpus_file)

    app.run(host="0.0.0.0", port = 9999)