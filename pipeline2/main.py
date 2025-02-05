from encode import *
from doc_generate import *
from search import *
from text2sound import *
from textcheck import *


def main():
     encoder = EncoderUDL()
     query_list = ["What is the capital of France?", "What is the capital of USA?"]
     query_embeddings = encoder.encode(query_list)
     print(query_embeddings.shape)

     searcher = SearchUDL(cluster_dir = "miniset")
     distances, indices = searcher.search_queries(query_embeddings, top_k=5)
     print(f"distances.shape: {distances.shape}")
     print(f"indices.shape: {indices.shape}")

     doc_gen = DocGenerateUDL()
     query_text = "What is the capital of France?"
     doc_ids = indices[0]
     result = doc_gen.generate(query_text, doc_ids)
     print(result)

     audio = text2soundfunc(result)

     if textcheck(result):
          print("The generated text is valid.")
     else:
          print("The generated text is invalid.")

main()