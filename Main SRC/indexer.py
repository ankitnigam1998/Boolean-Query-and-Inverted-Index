from linkedlist  import LinkedList as CustomLinkedList
from linkedlist  import Node as CustomNode
from collections import OrderedDict as CustomOrderedDict

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