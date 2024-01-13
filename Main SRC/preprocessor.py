import collections
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

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
