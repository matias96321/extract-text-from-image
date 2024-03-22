import os
import copy
from paddleocr import PaddleOCR
import requests
import numpy as np
from PIL import Image
import layoutparser as lp
from layoutparser.models.detectron2 import catalog

import warnings
warnings.filterwarnings("ignore")

def load_model(
    config_path: str = 'lp://<dataset_name>/<model_name>/config',
):

    config_path_split = config_path.split('/')
    dataset_name = config_path_split[-3]
    model_name = config_path_split[-2]
    
    # get the URLs from the MODEL_CATALOG and the CONFIG_CATALOG 
    # (global variables .../layoutparser/models/detectron2/catalog.py)
    model_url = catalog.MODEL_CATALOG[dataset_name][model_name]
    config_url = catalog.CONFIG_CATALOG[dataset_name][model_name]

    # override folder destination:
    if 'model' not in os.listdir():
        os.mkdir('model')

    config_file_path, model_file_path = None, None

    for url in [model_url, config_url]:
        filename = url.split('/')[-1].split('?')[0]
        save_to_path = f"model/" + filename
        if 'config' in filename:
            config_file_path = copy.deepcopy(save_to_path)
        if 'model_final' in filename:
            model_file_path = copy.deepcopy(save_to_path)

        # skip if file exist in path
        if filename in os.listdir("model"):
            continue
        # Download file from URL
        r = requests.get(url, stream=True, headers={'user-agent': 'Wget/1.16 (linux-gnu)'})

        with open(save_to_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=4096):
                if chunk:
                    f.write(chunk)

    # load the label map
    label_map = catalog.LABEL_MAP_CATALOG[dataset_name]
    
    return lp.models.Detectron2LayoutModel(
        config_path=config_file_path,
        model_path=model_file_path,
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7],
        label_map=label_map
    )

ocr_model = PaddleOCR( ocr=True, lang="en", ocr_version="PP-OCRv4" )

image = Image.open("imate-teste-3.png")

img = np.asarray(image) 

model = load_model('lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config')

layout = model.detect(image) 

text_blocks = lp.Layout([b for b in layout if (b.type=='Text' or b.type=='Title' )])
figure_and_List_blocks = lp.Layout(b for b in layout if (b.type=='Figure'))


h, w, _ = img.shape

left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)

left_blocks = text_blocks.filter_by(left_interval, center=True)
left_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)
right_blocks = lp.Layout([b for b in text_blocks if b not in left_blocks])
right_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)

text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])

for block in text_blocks:
    
    segment_image = (block.pad(left=5, right=5, top=5, bottom=5).crop_image(img))

    ocr_results = list(map(lambda x: x[0], ocr_model(segment_image)[1]))
    
    text = '\n'.join(ocr_results)

    block.set(text=text, inplace=True)

with open('result.txt', 'w', newline='') as f:
    for txt in text_blocks.get_texts():
        f.write(txt + '\n---\n')
    f.close()
