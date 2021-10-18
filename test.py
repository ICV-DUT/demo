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
