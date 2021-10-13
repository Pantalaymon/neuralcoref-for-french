
import re
from glob import glob
import numpy as np
from gensim.models import KeyedVectors, Word2Vec


def add_mean_vector(wv, new_vector_name):
    '''
       add a new vector that is the average of all the other vectors 
    '''
    array_all = np.array([wv[i] for i in range(len(wv))])
    mean_vector = np.mean(np.array(array_all),axis=0)
    wv[new_vector_name] = mean_vector
    return wv
    
def keyed_vectors_to_npy(wv, output_file_npy,output_file_txt):
    '''
        Converts the gensim keyed_vector to a npy file
    '''
    array = np.zeros((len(wv.index_to_key),wv.vector_size))
    with open(output_file_txt,"w",encoding="utf8") as output:
        for i,w in enumerate(wv.index_to_key):
            output.write(w+"\n")
            array[i] = wv[w]
        
    np.save(output_file_npy, array)



#directory with all the conll files

def make_list_sentences(conll_directory,suffix='.conll'):
    '''
        Make list of sentences from a conll file
        the list can be an input for gensim word2vec model
    '''
    sentences = []
    for file in glob(conll_directory+"/**/*"+suffix, recursive=True):
        print(file)
        sentence = []
        for line in open(file,encoding='utf8'):
            cols = line.split(" "*10)
            if len(cols) > 2:
                token = cols[3].lower()
                if re.match("\W+",token) or re.match("\d+",token): continue
                sentence.append(token)
            elif sentence:

                sentences.append(sentence)
                sentence = []
    return sentences
    
if __name__ == '__main__':
    #Generic Static pre-trained embeddings (trained on very large french corpus)

    MODEL_PATH = 'https://s3.us-east-2.amazonaws.com/embeddings.net/embeddings/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin'

    static_wv = KeyedVectors.load_word2vec_format(MODEL_PATH,binary=True, unicode_errors="ignore")
    static_wv = add_mean_vector(static_wv,"*UNK*")
    static_wv = add_mean_vector(static_wv,"<missing>")

    keyed_vectors_to_npy(static_wv, "static_word_embeddings.npy", "static_word_vocabulary.txt")

    print("Pretrained Model dimensions : ", (static_wv.vector_size,len(static.wv)))

    #Training new embeddings for the coreference corpus
                
    CONLL_DIRECTORY = './neuralcoref-master/neuralcoref/train/data'
    sentences_list = make_list_sentences(CONLL_DIRECTORY, '.v4_gold_conll')



    print("Training embeddings...")

    model = Word2Vec(
            sentences_list,
            min_count=1,
            workers=6,
            vector_size=200,
            epochs=100,
            sg=0)
    tuned_wv = add_mean_vector(model.wv,"*UNK*")
    tuned_wv = add_mean_vector(tuned_wv,"<missing>")
    print("Trained Model dimensions : ", (model.vector_size,len(model.wv)))

    keyed_vectors_to_npy(tuned_wv,"tuned_word_embeddings.npy" , "tuned_word_vocabulary.txt")

