import os
from tqdm import tqdm
from marker.convert import convert_single_pdf
from marker.models import load_all_models
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import PyPDF2


model = None

model_lst = None
PATH = "upload_data/"
pages_path = "pages"
OUTPUT_PATH = "converted_files"
PATH_CHUNKS = "chanks"

for d in [PATH,pages_path,OUTPUT_PATH,PATH_CHUNKS]:
    os.makedirs(d,exist_ok=True)
    
texts = []


def split_pdf(input_pdf_path:str, output_folder):
    if not os.path.exists(input_pdf_path) or not input_pdf_path.endswith(".pdf"):
        print(f"Файл {input_pdf_path} не найден.")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(input_pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(reader.pages)
        
        for page_num in range(total_pages):
            writer = PyPDF2.PdfWriter()
            writer.add_page(reader.pages[page_num])

            output_pdf_path = os.path.join(output_folder, f'page_{page_num + 1}.pdf')
            with open(output_pdf_path, 'wb') as output_pdf_file:
                writer.write(output_pdf_file)


def extract(file_name, output_name):
    global model_lst
    if model_lst is None:
        model_lst = load_all_models()

    print("FILE: {}".format(file_name))

    try:
        result, images, out_meta = convert_single_pdf(
            file_name, model_lst, langs=["ru"], batch_multiplier=4
        )

        with open(output_name, "w") as f:
            f.write(result)
    except:
        print("CUDA out of memory")
        result, images, out_meta = convert_single_pdf(
            file_name, model_lst, langs=["ru"], batch_multiplier=1
        )

        with open(output_name, "w") as f:
            f.write(result)


def convert_data():
    for file_index in tqdm(os.listdir(pages_path)):
        for page in os.listdir(os.path.join(pages_path, file_index)):
            path_to_page = os.path.join(pages_path, file_index, page)
            path_to_md_dir = os.path.join(OUTPUT_PATH, file_index)
            os.makedirs(path_to_md_dir, exist_ok=True)
            path_to_md = os.path.join(path_to_md_dir, f"{page[:-4]}.md")
            if os.path.isfile(path_to_md):
                continue
            extract(path_to_page, path_to_md)


def emb_text(text):
    global model
    if model is None:
        model = SentenceTransformer("intfloat/multilingual-e5-large")
    embeddings = model.encode(text)
    return embeddings

def get_chunks():
    threshold = 4096
    for file_index in tqdm(os.listdir(OUTPUT_PATH)):
        for page in os.listdir(os.path.join(OUTPUT_PATH,file_index)):
            path_to_page = os.path.join(OUTPUT_PATH,file_index,page)
            if os.path.isfile(path_to_page):
                with open(path_to_page, 'r', encoding='utf-8') as file:
                    data = file.read()
                    text = data.split('\n\n')
                    for i, element in enumerate(text):
                        tmp = element
                        for j in range(i + 1, len(text)):
                            if len(tmp) > threshold:
                                break
                            tmp = '{}\n{}'.format(tmp, text[j])
                        
                        p = os.path.join(PATH_CHUNKS,f"{file_index}_{page[:-3]}_{i}.txt")
                        a = tmp.replace("\n","")
                        if len(a) > 200:
                            with open(p, 'w', encoding='utf-8') as file:
                                file.write(tmp)

def add_data():
    for file_name in tqdm(os.listdir(PATH)):
        input_pdf = os.path.join(PATH,file_name)
        _output_dir = os.path.join(pages_path,file_name[:-4])
        r = os.makedirs(_output_dir,exist_ok=True)
        split_pdf(input_pdf, _output_dir)
    convert_data()
    get_chunks()
    for file in os.listdir(PATH_CHUNKS):
        with open(os.path.join(PATH_CHUNKS,file), "r") as f:
            text = f.read()

        file_name = "_".join(file.split("_")[:-3])
        page = int(file.split("_")[-2])
        texts.append((file_name, page, text))

    milvus_client = MilvusClient(uri="./milvus_demo.db")
    collection_name = "my_rag_collection"

    data = []
    for i, (file_name, page, text) in enumerate(
        tqdm(texts, desc="Creating embeddings")
    ):
        data.append(
            {
                "id": i,
                "vector": emb_text(text),
                "text": text,
                "file_name": file_name,
                "page": page,
            }
        )

    milvus_client.insert(collection_name=collection_name, data=data)
