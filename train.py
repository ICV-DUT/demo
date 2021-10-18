def training(loss, learning_rate):
    """
    输入
    loss: 损失函数（交叉熵）
    learning_rate： 学习率
    返回
    train_op: 训练的最优值
    """
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # global_step不是共享变量，初始值为0，设定trainable=False 可以防止该变量被数据流图的 GraphKeys.TRAINABLE_VARIABLES 收集,
        # 这样我们就不会在训练的时候尝试更新它的值。
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op


def evaluation(logits, labels):
    """
     输入
    logits: 经过cnn_inference处理过的tensor
    labels:
    返回
    accuracy：正确率
    """
    with tf.variable_scope('accuracy') as scope:
        prediction = tf.nn.softmax(logits)  # 这个logits有n_classes列
        # prediction每行的最大元素（1）的索引和label的值相同则为1 否则为0。
        correct = tf.nn.in_top_k(prediction, labels, 1)
        # correct = tf.nn.in_top_k(logits, labels, 1)   也可以不需要prediction过渡，因为最大值的索引没变，这里这样写是为了更好理解
        correct = tf.cast(correct, tf.float16)  # 记得要转换格式
        accuracy = tf.reduce_mean(correct)
    return accuracy
