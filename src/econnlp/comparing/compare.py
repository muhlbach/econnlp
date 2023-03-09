#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import pandas as pd
import numpy as np
import bodyguard as bg

# from econnlp.embedding.docs import DocumentEmbedder
from ..embedding.docs import DocumentEmbedder

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
class Comparer(object):
    """
    Compare text
    """
    def __init__(self,
                 normalize=True,
                 verbose=True,
                 model_name_or_path="all-roberta-large-v1",
                 ):
        self.normalize = normalize
        self.verbose = verbose
        self.model_name_or_path = model_name_or_path
        self.documentembedder = DocumentEmbedder(normalize=self.normalize,
                                                 verbose=self.verbose,
                                                 model_name_or_path=self.model_name_or_path)
        
    # Settings
    FROM_NAME = "from_value"
    TO_NAME = "to_value"
    SIMIL = "similarity"
    RANK = "rank"
    RETURN_TYPE_OPT = ["dict", "df"]
        
    def find_n_matches(self, from_entry, to_entry, n_matches=5, distance_metric="cosine", strip=False):

        
        # Sanity check
        bg.sanity_check.check_type(x=from_entry,
                                   allowed=list,
                                   name="from_entry")

        bg.sanity_check.check_type(x=to_entry,
                                   allowed=list,
                                   name="to_entry")
        
        bg.sanity_check.check_type(x=n_matches,
                                   allowed=int,
                                   name="n_matches")

        
        
        
        # Collect documents
        documents = bg.lists.unique(l=from_entry+to_entry)
        
        self.documentembedder.embed_documents(documents=documents,
                                              return_embeddings=False,
                                              return_type="df")
        
        from_entry_embd = self.documentembedder.extract_embeddings(which_embeddings=from_entry,
                                                                   return_type="df")
        
        to_entry_embd = self.documentembedder.extract_embeddings(which_embeddings=to_entry,
                                                                   return_type="df")        

        similarity = bg.distance.compute_similarity(a=from_entry_embd,
                                                    b=to_entry_embd,
                                                    metric=distance_metric)
        
        similarity_long = similarity.melt(var_name=self.TO_NAME,
                                          value_name=self.SIMIL,
                                          ignore_index=False).reset_index().rename(columns={"index":self.FROM_NAME})
        
        similarity_long[self.RANK] = similarity_long.groupby(by=[self.FROM_NAME])[self.SIMIL].rank(method="average",
                                                                                                   ascending=False)
        
        similarity_long.sort_values(by=[self.FROM_NAME, self.RANK],
                                    inplace=True)

        mask_topn = similarity_long[self.RANK] <= n_matches
        
        similarity_top = similarity_long.loc[mask_topn]
        
        if strip:
            similarity_top = similarity_top.drop(columns=[col for col in similarity_top if col not in [self.FROM_NAME,self.TO_NAME]], inplace=False).reset_index(drop=True)
            
        return similarity_top
            
        
    def find_best_match(self, from_entry, to_entry, distance_metric="cosine", return_type="dict", strip=False):
        
        bg.sanity_check.check_str(x=return_type,
                                  allowed=self.RETURN_TYPE_OPT,
                                  name="return_type")
        
        if return_type=="dict":
            strip=True
        
        
        similarity_top = self.find_n_matches(from_entry=from_entry,
                                             to_entry=to_entry,
                                             n_matches=1,
                                             distance_metric=distance_metric,
                                             strip=strip)
        
        if return_type=="dict":
            similarity_top = similarity_top.set_index(keys=[self.FROM_NAME]).squeeze().to_dict()
            
        return similarity_top
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        