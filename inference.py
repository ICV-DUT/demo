#加载训练好的模型进行推断
"""
查看input_data读取并处理过的图像

由于之前将图片转为float32了，但是imshow()是显示uint8类型的数据
而灰度值在uint8类型下是0~255，转为float32后会超出这个范围，所以色彩有点奇怪
"""

from input_data import *
import matplotlib.pyplot as plt

BATCH_SIZE = 2  # 一次查看两张照片
CAPACITY = 256
IMG_W = 208
IMG_H = 208

train_dir = 'D:/python/deep-learning/CatVsDog/Project/data/train/'
image_list, label_list = get_files(train_dir)
image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() and i < 1:
            img, label = sess.run([image_batch, label_batch])

            for j in np.arange(BATCH_SIZE):
                print("label: %d" % label[j])
                plt.imshow(img[j, :, :, :])
                plt.show()
            i += 1
    except tf.errors.OutOfRangeError:
        print("done!")
    finally:
        coord.request_stop()
    coord.join(threads)
