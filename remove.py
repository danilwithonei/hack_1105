import os
import re
from tqdm import tqdm


# directory = 'test.md'
pattern = re.compile(r'^\s*!\[.*?\]\(.*?\)\s*$')

PATH = 'converted_files_2'

g = 0
# for file_index in tqdm(os.listdir(PATH)):
#     for page in os.listdir(os.path.join(PATH,file_index)):
#         path_to_page   = os.path.join(PATH,file_index,page)
#         if os.path.isfile(path_to_page):
#             with open(path_to_page, 'r', encoding='utf-8') as file:
#                 lines = file.readlines()
#             old = len(lines)
#             filtered_lines = [line for line in lines if not pattern.match(line)]
#             g += old-len(filtered_lines)
#             with open(path_to_page, 'w', encoding='utf-8') as file:
#                 file.writelines(filtered_lines)
# print(g)
# for filename in tqdm(os.listdir(directory)):
#     file_path = os.path.join(directory, filename)
    
#     if os.path.isfile(file_path):
#         with open(file_path, 'r', encoding='utf-8') as file:
#             lines = file.readlines()
        
#         filtered_lines = [line for line in lines if not pattern.match(line)]
        
#         with open(file_path, 'w', encoding='utf-8') as file:
#             file.writelines(filtered_lines)

print("Строки удалены из всех файлов.")

PATH = 'converted_files_2'
path_to_save = "chunks_2/"
threshold = 4096
g = 0

for file_index in tqdm(os.listdir(PATH)):
    for page in os.listdir(os.path.join(PATH,file_index)):
        path_to_page = os.path.join(PATH,file_index,page)
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
                    
                    p = os.path.join(path_to_save,f"{file_index}_{page[:-3]}_{i}.txt")
                    a = tmp.replace("\n","")
                    if len(a) > 200:
                        g+=1
                        with open(p, 'w', encoding='utf-8') as file:
                            file.write(tmp)
print(g)