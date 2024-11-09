from marker.convert import convert_single_pdf  
from marker.models import load_all_models
import os
from tqdm import tqdm

#os.environ["CUDA_VISIBLE_DEVICES"]="" 
model_lst = load_all_models()

def extract2(file_name, output_name):
    print('FILE: {}'.format(file_name))

    try:
        result, images, out_meta = convert_single_pdf(
            file_name,
            model_lst,
            langs=['ru'],
            batch_multiplier=1
        )
    
        with open(output_name, 'w') as f:
            f.write(result)
    except:
        result = 'CUDA out of memory'



if __name__ == '__main__':
    import os

    PATH = '/home/bob/non_work/hack_1105/res/25/page_133.pdf'
    OUTPUT_PATH = 'test.md'
    extract2(PATH,OUTPUT_PATH)

                