import os
import glob
from PIL import Image

def cal_res(data_name, data_pth):
    imgs = [glob.glob(os.path.join(data_pth, 'train', '*.jpg')),
            glob.glob(os.path.join(data_pth, 'test', '*.jpg'))]
    wd = 0
    ht = 0
    avg_cnt = 0
    noSamples = 0
    avg_res = 0
    for pths in imgs:
        for img_pth in pths:
            img = Image.open(img_pth)
            wd += img.size[0]
            ht += img.size[1]
            with open(img_pth.replace('jpg', 'txt'), 'r') as f:
                avg_cnt += len(f.readlines())
        noSamples += len(pths)
    avg_cnt = (avg_cnt / noSamples)
    avg_res = (wd / noSamples, ht / noSamples)
    print(f'{data_name} avg_cnt: {avg_cnt} noSamples: {noSamples} avg_res: {avg_res}')

cal_res('2', os.path.join('dataset', '2'))
cal_res('3', os.path.join('dataset', '3'))
cal_res('sha', os.path.join('dataset', 'sha'))
cal_res('shb', os.path.join('dataset', 'shb'))
cal_res('qnrf', os.path.join('dataset', 'qnrf'))
cal_res('nwpu', os.path.join('dataset', 'nwpu'))