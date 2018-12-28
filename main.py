import vgg19
import argparse
import tensorflow as tf

from model import style_GAN_
from utils import check_folder

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of fast-style-GAN"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=32, help='The size of batch')
    parser.add_argument('--folder_path', type=str, default='data', help='Path of the dataset')
    parser.add_argument('--style_image', type=str, default='style.jpg', help='Path of the style image')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save training logs')
    parser.add_argument('--model_dir', type=str, default='model',
                        help='Directory to save the model')
    parser.add_argument('--content_layers', nargs='+', type=str, default=['conv4_2'], help='VGG19 layers used for content loss')
    parser.add_argument('--style_layers', nargs='+', type=str, default=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
                        help='VGG19 layers used for style loss')
    parser.add_argument('--content_layer_weights', nargs='+', type=float, default=[1.0], help='Content loss for each content is multiplied by corresponding weight')
    parser.add_argument('--style_layer_weights', nargs='+', type=float, default=[.2,.2,.2,.2,.2],
                        help='Style loss for each content is multiplied by corresponding weight')
    parser.add_argument('--dataset_name', type=str, default='shipData',choices=['shipdata', 'coco'],
                        help='Name of the dataset')
    parser.add_argument('--vgg_path', type=str, default='../vgg-model/imagenet-vgg-verydeep-19.mat',
                        help='Directory to load the pretrained vgg model')
    parser.add_argument('--loss_ratio', type=float, default='1e-3',
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

    CONTENT_LAYERS = {}
    for layer, weight in zip(args.content_layers, args.content_layer_weights):
        CONTENT_LAYERS[layer] = weight

    STYLE_LAYERS = {}
    for layer, weight in zip(args.style_layers, args.style_layer_weights):
        STYLE_LAYERS[layer] = weight

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        vgg = vgg19.VGG19(args.vgg_path)

        model = style_GAN_(sess,
                           epoch=args.epoch,
                           batch_size=args.batch_size,
                           folder_path=args.folder_path,
                           style_image_path=args.style_image,
                           checkpoint_dir=args.checkpoint_dir,
                           result_dir=args.result_dir,
                           log_dir=args.log_dir,
                           model_dir=args.model_dir,
                           dataset_name=args.dataset_name,
                           net=vgg,
                           loss_ratio=args.loss_ratio,
                           content_layer_ids = CONTENT_LAYERS,
                           style_layers_ids = STYLE_LAYERS)

        # build_graph
        model.build_model()

        # launch the graph in a session
        model.train()
        print("[*] Training finished!")

        # visualize learned generator
        model.visualize_results(args.epoch - 1)
        print("[*] Testing finished!")

if __name__ == '__main__':
    main()