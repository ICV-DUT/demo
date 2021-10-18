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
