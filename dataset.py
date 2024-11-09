from marker.convert import convert_single_pdf  
from marker.models import load_all_models
import os
from tqdm import tqdm
import pymupdf4llm 
from copy import deepcopy

#os.environ["CUDA_VISIBLE_DEVICES"]="" 
model_lst = load_all_models()

def extract2(file_name, output_name):
    print('FILE: {}'.format(file_name))

    try:
        result, images, out_meta = convert_single_pdf(
            file_name,
            model_lst,
            langs=['ru'],
            batch_multiplier=4
        )
    
        with open(output_name, 'w') as f:
            f.write(result)
    except:
        print('CUDA out of memory')
        result, images, out_meta = convert_single_pdf(
            file_name,
            model_lst,
            langs=['ru'],
            batch_multiplier=1
        )
    
        with open(output_name, 'w') as f:
            f.write(result)


def extract(file_name, output_name):
    print('FILE: {}'.format(file_name))
    
    result:str = pymupdf4llm.to_markdown(file_name)
    result_copy = deepcopy(result)
    result = result.replace("\n","")
    if len(result) >5:
        with open(output_name, 'w') as f:
            f.write(result_copy)
        return True
    else:
        return False


if __name__ == '__main__':
    import os

    PATH = 'res'
    OUTPUT_PATH = 'converted_files_2'

    for file_index in tqdm(os.listdir(PATH)):
        for page in os.listdir(os.path.join(PATH,file_index)):
            path_to_page   = os.path.join(PATH,file_index,page)
            path_to_md_dir = os.path.join(OUTPUT_PATH,file_index)
            os.makedirs(path_to_md_dir,exist_ok=True)
            path_to_md = os.path.join(path_to_md_dir,f"{page[:-4]}.md")
            if os.path.isfile(path_to_md):
                continue
                # r = extract(path_to_page, path_to_md)
            extract2(path_to_page,path_to_md)