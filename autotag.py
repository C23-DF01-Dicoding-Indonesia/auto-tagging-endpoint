import requests
import pinecone
import json
import os
from transformers import AutoTokenizer
import json

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENVIRONMENT') 
VECTOR_LENGTH = 384
INDEX_NAME = "auto-tag-ml"

def pinecone_init():

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )


def get_index(index_name: str):
    return pinecone.Index(index_name)



def get_embedding(
    text: str
) -> list[float]:
    """
    Embed a single text with embedding-endpoint
    # run the docker of embedding-endpoint first
    """
    tokenizer = AutoTokenizer.from_pretrained("carlesoctav/multi-qa-en-id-mMiniLMv2-L6-H384")
    batch = tokenizer(text)
    batch = dict(batch)
    batch = [batch]
    input_data = {"instances": batch}
    r = requests.post("http://localhost:8501/v1/models/bert:predict", data=json.dumps(input_data))
    result = json.loads(r.text)["predictions"][0]["last_hidden_state"][0]
    return result



def semantic_search(query: str, index_name: str, top_k: int) -> dict:
    """
    Semantic search of a query in a Pinecone index.
    """
    index = get_index(index_name)
    xq = get_embedding(query)
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    return xc


def get_tag_result(query: str, index_name: str , top_k: int, top_m:int) -> list:
    """
    Get search result of a query in a Pinecone index.
    """
    xc = semantic_search(query, index_name, top_k)
    result = dict()
    # use dictionary to 
    for i in xc["matches"]:
        for j in i["metadata"]["tags"]:
            if j in result:
                result[j] += i["score"]
            else:
                result[j] = i["score"]

    result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    #take top 5
    result = [i[0] for i in result][:top_m]
    print(result)
    return result


if __name__ == "__main__":
    pinecone_init()
    get_tag_result("kenapa machine learning susah sekali ya", INDEX_NAME,top_k = 15, top_m = 5)
