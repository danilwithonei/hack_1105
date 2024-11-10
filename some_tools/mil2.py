
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

#model = SentenceTransformer('cointegrated/rubert-tiny2')
# model = SentenceTransformer('cointegrated/LaBSE-en-ru')
model = SentenceTransformer('intfloat/multilingual-e5-large')

milvus_client = MilvusClient(uri="./milvus_demo.db")
collection_name = "my_rag_collection"

def emb_text(text):
    embeddings = model.encode(text)
    return embeddings

question = "Как за 10 лет изменилось количество телепрограмм, привлекающих более 4-х млн. зрителей в Великобритании?"
#question = "How has the number of TV programmes attracting over 4 million viewers in the UK changed over 10 years?"

search_res = milvus_client.search(
    collection_name=collection_name,
    data=[
        emb_text(question)
    ],  # Use the `emb_text` function to convert the question to an embedding vector
    limit=5,  # Return top 3 results
    search_params={"metric_type": "IP", "params": {}},  # Inner product distance
    output_fields=["text", "file_name", "page"],  # Return the text field
)


for res in search_res[0]:
    print(res["entity"]["file_name"], res["entity"]["page"],res["distance"],res["entity"]["text"])

