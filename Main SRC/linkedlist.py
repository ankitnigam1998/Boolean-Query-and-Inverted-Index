import math

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
