import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np
import glob
from torch.utils.data import DataLoader

from tqdm import tqdm
from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings

warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")
    parser.add_argument('--dataset_file', default='SHHA')
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')
    parser.add_argument('--video_name', default='', help='video')
    return parser

def record(pth, mode, ext):
    img_ = sorted([x for x in os.listdir(os.path.join(pth)) if x.endswith(ext)])
    with open(os.path.join('test', f'{mode}.list'), 'a') as t_f:
        for img in img_:
            t_f.write(f'{mode}/{img}\n')
def video_to_image(pth, name, ext):
    # Replace 'path/to/your/input/video.mp4' with the path to your input video file
    input_path = os.path.join(pth, name)

    # Create a folder to store the output frames
    output_folder = os.path.join(pth, f'{name.split(".")[0]}_original')
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get the total number of frames
    total_frames = 300 if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 300 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine the number of digits needed for formatting
    num_digits = len(str(total_frames))

    # Read and save all frames
    for frame_count in tqdm(range(total_frames)):
        ret, frame = cap.read()

        if not ret:
            break

        # Save the frame with leading zeros in the file name
        output_path = os.path.join(output_folder, f"frame_{frame_count + 1:0{num_digits}d}.{ext}")
        # resized_frame = cv2.resize(frame, ((1080, 720)))
        resized_frame = cv2.resize(frame, ((1920, 1080)))
        cv2.imwrite(output_path, resized_frame)

    # Release the video capture object
    cap.release()
    record(os.path.join('test', f'{name.split(".")[0]}_original'), f'{name.split(".")[0]}_original', ext)
    return sorted(glob.glob(os.path.join('test', f'{name.split(".")[0]}_original', f'*.{ext}')))

def image_to_video(pth, name, ext):
    # Replace 'path/to/your/input/frames' with the path to the folder containing your image frames
    input_folder = os.path.join(pth, f'{name.split(".")[0]}_output')
    os.makedirs(input_folder, exist_ok=True)

    # Replace 'output_video.mp4' with the desired name for the output video file
    output_video_path = os.path.join(pth, f'{name.split(".")[0]}_output.mp4')

    # Get the list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(ext)]

    # Sort the image files based on their names
    image_files.sort()

    # Read the first image to get dimensions
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, layers = first_image.shape

    # Create a VideoWriter object
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    # Write each frame to the video
    for image_file in tqdm(image_files):
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()


def main(args, ext, debug=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()

    if not os.path.exists(os.path.join('test', f'{args.video_name.split(".")[0]}_original')):
        images = video_to_image('test', args.video_name, ext)
    else:
        images = sorted(glob.glob(os.path.join('test', f'{args.video_name.split(".")[0]}_original', f'*.{ext}')))

    for image in images:
        # load the images
        img_raw = Image.open(image).convert('RGB')
        # create the pre-processing transform
        transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # pre-proccessing
        def fit_size(size):
            return (int(size / args.crop_size)) * args.crop_size

        max_width = 1920
        max_height = 1920

        if img_raw.size[0] > img_raw.size[1]:
            if img_raw.size[0] > max_width:
                ratio = max_width / img_raw.size[0]
                wd = max_width
                ht = fit_size(img_raw.size[1] * ratio)
            else:
                wd = fit_size(img_raw.size[0]) if img_raw.size[0] > args.crop_size else args.crop_size
                ht = fit_size(img_raw.size[1]) if img_raw.size[1] > args.crop_size else args.crop_size
        elif img_raw.size[1] > img_raw.size[0]:
            if img_raw.size[1] > max_height:
                ratio = max_height / img_raw.size[1]
                ht = max_height
                wd = fit_size(img_raw.size[0] * ratio)
            else:
                wd = fit_size(img_raw.size[0]) if img_raw.size[0] > args.crop_size else args.crop_size
                ht = fit_size(img_raw.size[1]) if img_raw.size[1] > args.crop_size else args.crop_size
        else:
            wd = fit_size(img_raw.size[0]) if img_raw.size[0] > args.crop_size else args.crop_size
            ht = fit_size(img_raw.size[1]) if img_raw.size[1] > args.crop_size else args.crop_size

        img_raw_resized = img_raw.resize((wd, ht))
        img = transform(img_raw_resized)

        samples = torch.Tensor(img).unsqueeze(0)
        samples = samples.to(device)
        # run inference
        with torch.no_grad():
            outputs = model(samples)
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
            outputs_points = outputs['pred_points'][0]

            threshold = 0.5
            # filter the predictions
            points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
            predict_cnt = int((outputs_scores > threshold).sum())

            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

            outputs_points = outputs['pred_points'][0]
            # draw the predictions
            size = 2
            img_to_draw = cv2.cvtColor(np.array(img_raw_resized), cv2.COLOR_RGB2BGR)
            scale_x = img_raw.size[0] / img_to_draw.shape[1]
            scale_y = img_raw.size[1] / img_to_draw.shape[0]
            img_to_draw = cv2.resize(img_to_draw, (img_raw.size[0], img_raw.size[1]))
            resized_points = [(int(x * scale_x), int(y * scale_y)) for x, y in points]
            for p in resized_points:
                img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
            os.makedirs(os.path.join('test', f'{args.video_name.split(".")[0]}_output'), exist_ok=True)
            # save the visualized image
            cv2.putText(img_to_draw, f'predicted: {predict_cnt}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imwrite(os.path.join(os.path.join('test', f'{args.video_name.split(".")[0]}_output'), f'{os.path.basename(image)}'), img_to_draw)
            print(f'{os.path.basename(image)} pred_cnt: {predict_cnt}')

    image_to_video(os.path.join('test'), args.video_name, ext)

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    ext = 'jpg'
    main(args, ext)
    print(time.time() - start_time)