import argparse
import os
import time

import numpy as np
import tensorflow as tf

import style_transfer_tester
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MAX_SIZE = 512


def add_one_dim(image):
    shape = (1,) + image.shape
    return np.reshape(image, shape)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content', default='yuanshi22.jpg', type=str)
    parser.add_argument('--style', default='fg1.jpg', type=str)
    parser.add_argument('--output', type=str, default='result.jpg')
    parser.add_argument('--max_size', type=int, default=512)

    return check_args(parser.parse_args())


def check_args(args):
    try:
        assert os.path.exists(args.content)
    except:
        print('There is no content')
        return None
    return args


def main():
    try:
        start_time = time.time()
        # parse arguments
        args = parse_args()
        if args is None:
            return 0
        # load content image and style image
        content_image = utils.load_image(args.content, max_size=args.max_size)
        init_image = content_image
        style_image = utils.load_image(args.style, shape=(content_image.shape[1], content_image.shape[0]))
        # open session
        sess = tf.Session()
        # build the graph
        transformer = style_transfer_tester.StyleTransferTester(session=sess,
                                                                content_image=add_one_dim(content_image),
                                                                style_image=add_one_dim(style_image),
                                                                init_image=add_one_dim(init_image),
                                                                fg_index=args.style.split('/')[-1].split('.')[0]
                                                                )
        # execute the graph
        output_image = transformer.test()
        # close session
        sess.close()
        # remove batch dimension
        shape = output_image.shape
        output_image = np.reshape(output_image, shape[1:])
        # plot result
        utils.plot_images(content_image=content_image, style_image=style_image, mixed_image=output_image)
        # save result
        utils.save_image(output_image, args.output)
        end_time = time.time()
        return format(end_time - start_time, '.2f')
    except:
        return 0


if __name__ == '__main__':
    run_status = main()
