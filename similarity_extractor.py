import numpy as np
from scipy.spatial.distance import cosine


class WeightedSimilarityExtractor:
    def __init__(self, words_sim, idfs, avgsl, k, b, interval_borders):
        '''
        words_sim: Word-Word Similarity Matrix (Under one embedding)
        idfs: IDF of words (np.array)
        avgsl, k, b: the same meaning with Equation
        interval_borders: The border of interval
        '''
        self.words_sim = words_sim
        self.idfs = idfs
        self.avgsl = avgsl
        self.k = k
        self.b = b
        self.interval_borders = interval_borders
    
    def cal_sem(self, long_s, short_s):
        '''
        calculate sem(w, s, E)
        long_s: the long sentence
        short_s: the short sentence
        '''
        long_indexes = np.repeat(long_s, len(short_s))
        short_indexes = np.tile(short_s, len(long_s))

        sem_matrix = self.words_sim[long_indexes, short_indexes].reshape((len(long_s), len(short_s)))
        return np.max(sem_matrix, 1)
    
    def extract(self, sentence_pair):
        '''
        extract weighted_similarity features
        sentence_pair: given (mashup_sentence, service_sentence)
        '''

        long_s = sentence_pair[0] if len(sentence_pair[0]) > len(sentence_pair[1]) else sentence_pair[1]
        short_s = sentence_pair[1] if long_s is sentence_pair[0] else sentence_pair[0]
        sem_vector = self.cal_sem(long_s, short_s)
        idfs = self.idfs[long_s]
        k = self.k
        b = self.b
        
        #Assign words into different intervals according to their sem with the short sentence
        intervals = [[] for _ in range(len(self.interval_borders) + 1)]
        for sim_idx, sem in enumerate(sem_vector):
            for idx, border in enumerate(self.interval_borders):
                if sem <= border:
                    intervals[idx].append(sim_idx)
                    break
            else:
                intervals[-1].append(sim_idx)
        #Start Calculate e_{ws} in each interval
        interval_ws = []
        factor = k*(1-b+b*(len(short_s)/self.avgsl))
        for interval in intervals:
            if len(interval) == 0:
                interval_ws.append(0)
            else:
                interval_sem = sem_vector[interval]
                e_ws = ((k + 1) * interval_sem) / (interval_sem + factor)
                interval_idfs = self.idfs[long_s[interval]]
                e_ws = np.average(e_ws * interval_idfs)
                interval_ws.append(e_ws)
        return interval_ws

class TextLevelSimilarityExtractor:
    def __init__(self, E):
        '''
        extract text-level similarity feature
        E: given embedding
        '''
        self.E = E
    
    def extract(self, sentence_pair):
        s1, s2 = sentence_pair
        s1_embedding = np.average(self.E[s1], axis = 0)
        s2_embedding = np.average(self.E[s2], axis = 0)
        similarity = 1 - cosine(s1_embedding, s2_embedding)
        return [similarity]

class UnWeightedSimilarityExtractor:
    def __init__(self, words_sim, interval_borders):
        '''
        words_sim: Word-Word Similarity Matrix (Under one embedding)
        interval_borders: The border of interval
        '''
        self.words_sim = words_sim
        self.interval_borders = interval_borders   
    
    def extract(self, sentence_pair):
        s1, s2 = sentence_pair
        s1_row_idx = np.repeat(s1,s2.shape[0])
        s2_col_idx = np.tile(s2,s1.shape[0])

        biparitie_flattten = self.words_sim[s1_row_idx,s2_col_idx]

        intervals = [[] for _ in range(len(self.interval_borders) + 1)]
        for sim in biparitie_flattten:
            for idx, border in enumerate(self.interval_borders):
                if sim <= border:
                    intervals[idx].append(sim)
                    break
            else:
                intervals[-1].append(sim)

        for idx in range(len(intervals)):
            if len(intervals[idx]) == 0:
                intervals[idx] = 0
            else:
                intervals[idx] = sum(intervals[idx]) / len(intervals[idx])

        return intervals


def load_embedding(path):
    embedding = np.load(path)
    return embedding

def load_dictionary(path):
    id2word, word2id = {}, {}
    with open(path, encoding='utf8') as f:
        for line in f:
            if line and line.strip():
                line = line.strip()
                id_, word = line.split()
                id2word[int(id_)] = word
                word2id[word] = int(id_)
    return id2word, word2id

def load_sentence(path, word2id):
    with open(path, encoding='utf8') as f:
        content = f.readline().strip().split()
    content = np.array([word2id[word] for word in content])


    return content
    
    
def load_idfs(path, word2id):
    idfs = np.zeros(len(word2id))
    with open(path, encoding = 'utf8') as f:
        for line in f:
            if line and line.strip():
                line = line.strip()
                word, idf = line.split()
                idfs[word2id[word]] = idf
    return idfs
    
def main():
    EMBEDDING_PATH = 'data/embedding.npy'
    DICTIONARY_PATH = 'data/vocabulary.txt'
    MASHUP_PATH = 'data/mashup.txt'
    SERVICE_PATH = 'data/service.txt'
    IDF_PATH = 'data/idf.txt'
    AVGSL = 21.2
    WS_INTERVAL = [0.15, 0.4, 0.8]
    UN_INTERVAL = [0.45, 0.8]
    
    embedding = load_embedding(EMBEDDING_PATH)
    id2word, word2id = load_dictionary(DICTIONARY_PATH)
    mashup = load_sentence(MASHUP_PATH, word2id)
    service = load_sentence(SERVICE_PATH, word2id)
    idfs = load_idfs(IDF_PATH, word2id)

    #Calculate Words Sim Matrix
    all_words_idx = np.arange(0, len(id2word))
    words_sim = np.zeros((len(id2word), len(id2word)))

    for w1 in range(len(id2word)):
        for w2 in range(w1+1, len(id2word)):
            sim = 1 - cosine(embedding[w1], embedding[w2])
            words_sim[w1, w2] = sim 
            words_sim[w2, w1] = sim 

    # Extract three types of similarity features.
    f_ws = WeightedSimilarityExtractor(words_sim, idfs, AVGSL, 1.2, 0.75, WS_INTERVAL)
    f_un = UnWeightedSimilarityExtractor(words_sim, UN_INTERVAL)
    f_mts = TextLevelSimilarityExtractor(embedding)
    
    ws_feature = f_ws.extract((mashup, service))
    un_feature = f_un.extract((mashup, service))
    mts_feature = f_mts.extract((mashup, service))

    feature = np.concatenate((ws_feature, un_feature, mts_feature))

    return feature


main()