import numpy as np
import tensorflow as tf

mean_pixel = np.array([123.68, 116.779, 103.939])


def preprocess(image):
    return image - mean_pixel


def undo_preprocess(image):
    return image + mean_pixel


class StyleTransferTester:
    def __init__(self, init_image, content_image,
                 style_image, session, fg_index):
        self.fgIndex = fg_index
        self.sess = session

        # preprocess input images
        self.p0 = np.float32(preprocess(content_image))
        self.a0 = np.float32(preprocess(style_image))
        self.x0 = np.float32(preprocess(init_image))

        # build graph for style transfer
        self._build_graph()

    def _build_graph(self):
        """ prepare data """
        # this is what must be trained
        self.x = tf.Variable(self.x0, trainable=True, dtype=tf.float32)

        # graph input
        self.p = tf.placeholder(tf.float32, shape=self.p0.shape, name='content')
        self.a = tf.placeholder(tf.float32, shape=self.a0.shape, name='style')

    def test(self):
        ckpt_root_path = './checkpoint/' + self.fgIndex
        ckpt_path = tf.train.latest_checkpoint(ckpt_root_path)
        # initialize parameters
        self.sess.run(tf.global_variables_initializer())
        # load pre-trained model
        saver = tf.train.Saver()
        # saver = tf.train.import_meta_graph(ckpt_path + ".meta")
        saver.restore(self.sess, ckpt_path)
        # get transformed image
        final_image = self.sess.run(self.x, feed_dict={self.a: self.a0, self.p: self.p0})
        # clip image
        final_image = np.clip(undo_preprocess(final_image), 0.0, 255.0)
        return final_image
