#定义model类
import tensorflow as tf


def cnn_inference(images, batch_size, n_classes):
    """
    输入
    images      输入的图像
    batch_size  每个批次的大小
    n_classes   n分类
    返回
    softmax_linear 还差一个softmax
    """
    # 第一层的卷积层conv1，卷积核为3X3，有16个
    with tf.variable_scope('conv1') as scope:
        # 建立weights和biases的共享变量
        # conv1, shape = [kernel size, kernel size, channels, kernel numbers]
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))  # stddev标准差
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        # 卷积层 strides = [1, x_movement, y_movement, 1], padding填充周围有valid和same可选择
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)         # 加入偏差
        conv1 = tf.nn.relu(pre_activation, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间

    # 第一层的池化层pool1和规范化norm1(特征缩放）
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm1')
        # ksize是池化窗口的大小=[1,height,width,1]，一般height=width=池化窗口的步长
        # 池化窗口的步长一般是比卷积核多移动一位
        # tf.nn.lrn是Local Response Normalization，（局部响应归一化）

    # 第二层的卷积层cov2，这里的命名空间和第一层不一样，所以可以和第一层取同名
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 16, 16],  # 这里只有第三位数字16需要等于上一层的tensor维度
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    # 第二层的池化层pool2和规范化norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                               padding='SAME',name='pooling2')
        # 这里选择了先规范化再池化

    # 第三层为全连接层local3
    with tf.variable_scope('local3') as scope:
        # flatten-把卷积过的多维tensor拉平成二维张量（矩阵）
        reshape = tf.reshape(pool2, shape=[batch_size, -1])  # batch_size表明了有多少个样本

        dim = reshape.get_shape()[1].value  # 知道-1(代表任意)这里具体是多少个
        weights = tf.get_variable('weights',
                                  shape=[dim, 256],  # 连接256个神经元
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)  # 矩阵相乘加上bias

    # 第四层为全连接层local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[256, 512], # 再连接512个神经元
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[512],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # 第五层为输出层softmax_linear
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('weights',
                                  shape=[512, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
        # 这里只是命名为softmax_linear，真正的softmax函数放在下面的losses函数里面和交叉熵结合在一起了，这样可以提高运算速度。
        # softmax_linear的行数=local4的行数，列数=weights的列数=bias的行数=需要分类的个数
        # 经过softmax函数用于分类过程中，它将多个神经元的输出，映射到（0,1）区间内，可以看成概率来理解

    return softmax_linear


def losses(logits, labels):
    """
    输入
    logits: 经过cnn_inference处理过的tensor
    labels: 对应的标签
    返回
    loss： 损失函数（交叉熵）
    """
    with tf.variable_scope('loss') as scope:
        # 下面把交叉熵和softmax合到一起写是为了通过spares提高计算速度
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='loss_per_eg')
        loss = tf.reduce_mean(cross_entropy, name='loss')  # 求所有样本的平均loss
    return loss
