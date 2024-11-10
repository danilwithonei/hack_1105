import openpyxl
from tqdm import tqdm

import rag
from settings import *


def read_xlsx(file_path):
    # Load the workbook
    workbook = openpyxl.load_workbook(file_path)
    
    # Select the active sheet
    sheet = workbook.active
    
    data = []
    for row in sheet.iter_rows(values_only=True):
        if row[0] == 'question':
            continue
        
        data.append(tuple(row[:4]))
    
    return data


INPUT = 'data/question_example.xlsx'
OUTPUT = 'sumbission.csv'

data = read_xlsx(INPUT)
print(data)

rag = rag.RAG(ollama_host = HOST, milvus_uri = MILVUS_DB, debug = not True)

with open(OUTPUT, 'w') as f:
    f.write('question, answer, filename, slide_number\n')
    for ind, (question, answer, filename, slide_number) in enumerate(tqdm(data)):
        answer, _, filename, slide_number = rag.tree(question)
        f.write('{},{},{},{}\n'.format(question, answer, filename, slide_number ))
