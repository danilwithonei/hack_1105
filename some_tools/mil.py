
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from tqdm import tqdm
import os

#model = SentenceTransformer('cointegrated/rubert-tiny2')
#model = SentenceTransformer('cointegrated/LaBSE-en-ru')
model = SentenceTransformer('intfloat/multilingual-e5-large')


PATH = 'chunks_2'
texts = []

for file in os.listdir(PATH):
    with open('{}/{}'.format(PATH, file), 'r') as f:
        text = f.read()

    file_name = '_'.join(file.split('_')[:-3])
    page = int(file.split('_')[-2])
    texts.append((file_name, page, text))
    print(file_name,page)


def emb_text(text):
    embeddings = model.encode(text)
    return embeddings

milvus_client = MilvusClient(uri="./milvus_demo.db")

collection_name = "my_rag_collection"

if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

test_embedding = emb_text("This is a test")
embedding_dim = len(test_embedding)

milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Strong consistency level
)

data = []

for i, (file_name, page, text) in enumerate(tqdm(texts, desc="Creating embeddings")):
    data.append({"id" : i, "vector" : emb_text(text), "text" : text, "file_name" : file_name, "page" : page})

milvus_client.insert(collection_name=collection_name, data=data)



