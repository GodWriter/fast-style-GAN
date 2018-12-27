import os
import tensorflow as tf
import argparse

from model import style_GAN_
from utils import check_folder

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of fast-style-GAN"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save training logs')
    parser.add_argument('--model_dir', type=str, default='model',
                        help='Directory to save the model')
    parser.add_argument('--dataset_name', type=str, default='mnist',choices=['mnist', 'coco'],
                        help='Name of the dataset')
    parser.add_argument('--vgg_path', type=str, default='../vgg-model/imagenet-vgg-verydeep-19.mat',
                        help='Directory to load the pretrained vgg model')
    parser.add_argument('--style_strength', type=float, default='1.',
                        help='the strength of the style loss')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    assert args.epoch >= 1

    # --batch_size
    assert args.batch_size >= 1

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model = style_GAN_(sess,
                           epoch=args.epoch,
                           batch_size=args.batch_size,
                           checkpoint_dir=args.checkpoint_dir,
                           result_dir=args.result_dir,
                           log_dir=args.log_dir,
                           model_dir=args.model_dir,
                           dataset_name=args.dataset_name,
                           vgg_path=args.vgg_path,
                           style_strength=args.style_strength)

        # buld_graph
        model.build_model()

        # launch the graph in a session
        model.train()
        print("[*] Training finished!")

        # visualize learned generator
        model.visualize_results(args.epoch - 1)
        print("[*] Testing finished!")

if __name__ == '__main__':
    main()