import argparse

import glob
from torch.utils.data import DataLoader

from PIL import Image
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
import random
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")
    parser.add_argument('--data', default='all')
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    parser.add_argument('--threshold', default=0.5)
    parser.add_argument('--crop_size', default=128)
    parser.add_argument('--verbose', default=True)
    parser.add_argument('--save_path', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--eval_freq', default=1, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def main(args, debug=False):
    times = []
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
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

    maes = []
    mses = []
    image_paths = sorted(glob.glob(os.path.join('data', args.data, 'test', '*.jpg')))

    for image_path in image_paths:
        # get gt_count
        with open(image_path.replace('jpg', 'txt'), 'r') as t_f:
            gt_cnt = len(t_f.readlines())
        # load the images
        img_raw = Image.open(image_path).convert('RGB')
        # create the pre-processing transform
        transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = img_raw.copy()

        # pre-proccessing
        def fit_size(size):
            return (int(size / args.crop_size)) * args.crop_size

        max_width = 1920
        max_height = 1920

        if image.size[0] > image.size[1]:
            if image.size[0] > max_width:
                ratio = max_width / image.size[0]
                wd = max_width
                ht = fit_size(image.size[1] * ratio)
            else:
                wd = fit_size(image.size[0]) if image.size[0] > args.crop_size else args.crop_size
                ht = fit_size(image.size[1]) if image.size[1] > args.crop_size else args.crop_size
        elif image.size[1] > image.size[0]:
            if image.size[1] > max_height:
                ratio = max_height / image.size[1]
                ht = max_height
                wd = fit_size(image.size[0] * ratio)
            else:
                wd = fit_size(image.size[0]) if image.size[0] > args.crop_size else args.crop_size
                ht = fit_size(image.size[1]) if image.size[1] > args.crop_size else args.crop_size
        else:
            wd = fit_size(image.size[0]) if image.size[0] > args.crop_size else args.crop_size
            ht = fit_size(image.size[1]) if image.size[1] > args.crop_size else args.crop_size

        image = image.resize((wd, ht))
        img = transform(image)
        samples = img.unsqueeze(0).to(device)
        # run inference
        with torch.no_grad():
            t1 = time.time()
            outputs = model(samples)
            t2 = time.time()
            times.append(t2-t1)
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
            outputs_points = outputs['pred_points'][0]

            # filter the predictions
            points = outputs_points[outputs_scores > args.threshold].detach().cpu().numpy().tolist()
            predict_cnt = int((outputs_scores > args.threshold).sum())
            # outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
            # outputs_points = outputs['pred_points'][0]

            if args.save_path:
                # draw the predictions
                img_to_draw = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                img_to_draw = cv2.resize(img_to_draw, (wd, ht))
                for p in points:
                    img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

                # save the visualized image
                img_to_draw = cv2.resize(img_to_draw, (img_raw.size[0], img_raw.size[1]))
                cv2.putText(img_to_draw, "Count:" + str(predict_cnt), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
                os.makedirs(args.save_path, exist_ok=True)
                cv2.imwrite(os.path.join(args.save_path, f'{os.path.basename(image_path)}'), img_to_draw)

            if args.verbose:
                print(f'{os.path.basename(image_path)} gt: {gt_cnt} pred: {predict_cnt}')

            # accumulate MAE, MSE
            mae = abs(predict_cnt - gt_cnt)
            mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
            maes.append(float(mae))
            mses.append(float(mse))

    # calc MAE, MSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))
    print(f'mse: {round(mse, 2)} mae: {round(mae, 2)}')
    print(np.mean(np.array(times)))

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    print(time.time() - start_time)