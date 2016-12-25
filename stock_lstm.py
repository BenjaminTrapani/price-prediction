import tensorflow as tf

class StockLSTM(object):

    def __init__(self, is_training, simulationParams):
        self.batch_size = batch_size = simulationParams.batchSize
        self.num_steps = num_steps = simulationParams.technicalsPerPrice
        self.is_training = is_training

        size = simulationParams.hiddenSize

        self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        if is_training and simulationParams.keepProb < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=simulationParams.keepProb)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * simulationParams.numLayers)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        iw = tf.get_variable("input_w", [1, size])
        ib = tf.get_variable("input_b", [size])
        inputs = [tf.nn.xw_plus_b(i_, iw, ib) for i_ in tf.split(1, num_steps, self._input_data)]
        if is_training and simulationParams.keepProb < 1:
            inputs = [tf.nn.dropout(input_, simulationParams.keepProb) for input_ in inputs]

        outputs, states = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
        rnn_output = tf.reshape(tf.concat(1, outputs), [batch_size, size * num_steps])

        softmax_w = tf.get_variable("softmax_w", [size * num_steps, num_steps])
        softmax_b = tf.get_variable("softmax_b", [num_steps])
        logits = tf.matmul(rnn_output, softmax_w) + softmax_b
        self._cost = cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self._targets))

        self._output = logits
        self._final_state = states

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), simulationParams.maxGradNorm)
        # optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def output(self):
        return self._output

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op
