# neuralcoref-for-french
Training coreference resolution model with neuralcoref on the DEMOCRAT french corpus


- make_french_embeddings.py : produce 200 dimensions vector with french vocabulary in npy format 
Those vectors are used in the training of the neuralcoref model

- democrat_metadata.csv : metadata of all the documents in the DEMOCRAT corpus. 
Those metadata are used by convert_conll.py to build the conll files

- convert_conll.py : browses all the files in the DEMOCRAT corpus and extracts the tokens to make sentences in the conll format
 Then merge the files and produce three large conll files : train , dev and test
