"""
This script implements some basic tests of the package that should be run before uploading
"""
#------------------------------------------------------------------------------
# Run interactively
#------------------------------------------------------------------------------
# import os
# # Manually set path of current file
# path_to_here = "/Users/muhlbach/Repositories/econnlp/src/"
# # Change path
# os.chdir(path_to_here)
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import pandas as pd
import bodyguard as bg
import scipy
import statsmodels

from econnlp.embedding.docs import DocumentEmbedder
from econnlp.comparing.compare import Comparer
#------------------------------------------------------------------------------
# COMPARE
#------------------------------------------------------------------------------
self = comparer = Comparer()

list_a = ["ALUMINIUM", "CHEMICALS", "POLYMER"]
list_b = ["Aluminium", "Chemical", "Polymers"]

comparer.find_n_matches(
    from_entry=list_a,
    to_entry=list_b,
    n_matches=5,
    )

comparer.find_best_match(
    from_entry=list_a,
    to_entry=list_b,
    return_type="df",
    strip=True,
    )



#------------------------------------------------------------------------------
# DEFAULT EXAMPLE
#------------------------------------------------------------------------------
# Instantiate
self = documentembedder = DocumentEmbedder(normalize=True,
                                           verbose=2,
                                           model_name_or_path="all-roberta-large-v1")

documentembedder.show_available_models()

embeddings = documentembedder.embed_documents(documents=["This is a document embedder",
                                                         "This is another document embedder",
                                                         "This is a third document embedder"],
                                              return_embeddings=True,
                                              return_type="df")

documentembedder.compute_similarity(a=["This is dull"],
                                    b=["This is fun"],
                                    metric="cosine")


from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-roberta-large-v1")

#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)
