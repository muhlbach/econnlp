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

from econnlp.embedding.docs import DocumentEmbedder
#------------------------------------------------------------------------------
# DEFAULT EXAMPLE
#------------------------------------------------------------------------------
# Instantiate
self = documentembedder = DocumentEmbedder(normalize=True,
                                           verbose=2)

documentembedder.show_available_models()


embeddings = documentembedder.embed_documents(documents=["This is a document embedder",
                                                         "This is another document embedder",
                                                         "This is a third document embedder"],
                                              return_type="df",
                                              model_name_or_path="all-roberta-large-v1")

