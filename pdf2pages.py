import PyPDF2
import os
from tqdm import tqdm

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

# Пример использования
input_dir = '/home/bob/Downloads/train_dataset_mediawise_train/train_data_mediawise/Media_Digital'
output_dir = 'res'

for file_name in tqdm(os.listdir(input_dir)):
    input_pdf = os.path.join(input_dir,file_name)
    _output_dir = os.path.join(output_dir,file_name[:-4])
    r = os.makedirs(_output_dir,exist_ok=True)
    split_pdf(input_pdf, _output_dir)
