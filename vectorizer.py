from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import numpy
import gensim


def vectorize_tfidf(data, column, remove_stopwords=True):
    vectorizer = CountVectorizer(analyzer='word', 
                                 stop_words=stopwords.words(['german', 'english']) if remove_stopwords else None)
    #, token_pattern=r"(?u)\b\w\w\w+\b")
    x_counts = vectorizer.fit_transform(data[column])
    # filter out documents with less than 3 entries
    data['Sufficient_Info'] = list(x_counts.getnnz(axis=1) >2)
    x_counts = x_counts[x_counts.getnnz(axis=1) > 2]

    print(f"Found {len(vectorizer.vocabulary_)} words in Count Vectorizer")

    transformer = TfidfTransformer(smooth_idf=False)
    x_tfidf = transformer.fit_transform(x_counts)

    xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)

    return x_counts, xtfidf_norm, vectorizer, data

class Datastories_embedding:

    def __init__(self, embedding_path=None):
        self.text_processor = TextPreProcessor(
            # terms that will be normalized
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                'time', 'url', 'date', 'number'],
            # terms that will be annotated
            annotate={"hashtag", "allcaps", "elongated", "repeated", 'emphasis', 'censored'},
            #annotate={},
            fix_html=True,  # fix HTML tokens
    
            # corpus from which the word statistics are going to be used 
            # for word segmentation 
            segmenter="twitter", 
    
            # corpus from which the word statistics are going to be used 
            # for spell correction
            corrector="twitter", 
    
            unpack_hashtags=True,  # perform word segmentation on hashtags
            unpack_contractions=True,  # Unpack contractions (can't -> can not)
            spell_correct_elong=False,  # spell correction for elongated words
    
            # select a tokenizer. You can use SocialTokenizer, or pass your own
            # the tokenizer, should take as input a string and return a list of tokens
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
            # list of dictionaries, for replacing tokens extracted from the text,
            # with other expressions. You can pass more than one dictionaries.
            dicts=[emoticons]
        )
        if embedding_path is not None:
            m = gensim.models.KeyedVectors.load(embedding_path)
            self.embeddings = m.vectors
            self.word_indices = m.key_to_index
            self.embeddings_dict = {word: self.embeddings[index] for word, index in self.word_indices.items()}
        else:
            self.dim = 100
            self.embeddings_dict = self.load_embeddings_dict()
            self.embeddings, self.word_indices = self.get_embeddings(self.embeddings_dict, self.dim)

    
    def load_embeddings_dict(self):
        embeddings_dict = {}
        dim = 100
        f = open('../datastories-semeval2017-task4/embeddings/datastories.twitter.100d.txt', "r", encoding="utf-8")
        for i, line in enumerate(f):
            values = line.split()
            word = values[0]
            coefs = numpy.asarray(values[1:], dtype='float32')
            # if not self.is_ascii(word):
            #     print(word)
        
            # if word.lower() in {'<unk>', "<unknown>"}:
            #     print(word)
            #     print("UNKNOWN")
            #     print()
    
            embeddings_dict[word] = coefs
        f.close()
        print('Found %s word vectors.' % len(embeddings_dict))
        return embeddings_dict


    def get_embeddings(self, vectors, dim):
        vocab_size = len(vectors)
        print('Loaded %s word vectors.' % vocab_size)
        wv_map = {}
        pos = 0
        # +1 for zero padding token and +1 for unk
        emb_matrix = numpy.ndarray((vocab_size + 2, dim), dtype='float32')
        for i, (word, vector) in enumerate(vectors.items()):
            pos = i + 1
            wv_map[word] = pos
            emb_matrix[pos] = vector
    
        # add unknown token
        pos += 1
        wv_map["<unk>"] = pos
        emb_matrix[pos] = numpy.random.uniform(low=-0.05, high=0.05, size=dim)
    
        return emb_matrix, wv_map


    def preprocess_sent(self, sentence):
        return self.text_processor.pre_process_doc(sentence)

    def index_vector(self, sentence, max_length):
        ekphrased = self.preprocess_sent(sentence)
        if len(ekphrased) > max_length:
            # Remove stopwords if vector length is too big
            ekphrased = [x for x in ekphrased if x not in stopwords.words('english', 'german')]
            ekphrased = ekphrased[:max_length]
        X = [self.word_indices[word] if word in self.word_indices else self.word_indices['<unk>'] for word in ekphrased]
        X = numpy.pad(numpy.array(X), (0, max_length -len(X)), 'constant', constant_values=(0,0))
        return X
