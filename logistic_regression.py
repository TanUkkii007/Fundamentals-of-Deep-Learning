import input_data
import tensorflow as tf
import time

mnist = input_data.read_data_sets("data/", one_hot=True)

learning_rate = 0.01
training_epochs = 60
batch_size = 100
display_step = 1

def inference(x):
    init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", [28*28, 10], initializer=init)
    b = tf.get_variable("b", [10])

    output = tf.nn.softmax(tf.matmul(x, W) + b)

    w_hist = tf.summary.histogram("weights", W)
    b_hist = tf.summary.histogram("biasses", b)
    y_hist = tf.summary.histogram("output", output)

    return output

def loss(output, y):
    dot_product = y * tf.log(output)

    xentropy = -tf.reduce_sum(dot_product, reduction_indices=1)

    loss = tf.reduce_mean(xentropy)

    return loss

def training(cost, global_step):
    tf.summary.scalar("cost", cost)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)

    return train_op

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("validation error", (1.0 - accuracy))

    return accuracy

if __name__ == '__main__':
    with tf.Graph().as_default():

        x = tf.placeholder("float", [None, 28*28])
        y = tf.placeholder("float", [None, 10])

        output = inference(x)

        cost = loss(output, y)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        train_op = training(cost, global_step)

        eval_op = evaluate(output, y)

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()

        sess = tf.Session()

        summary_writer = tf.summary.FileWriter("logistic_logs/", graph=sess.graph)

        init_op = tf.global_variables_initializer()

        sess.run(init_op)

        for epoch in range(training_epochs):
            avg_cost = 0.

            total_batch = int(mnist.train.num_examples / batch_size)

            for i in range(total_batch):
                minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)

                sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y})

                avg_cost += sess.run(cost, feed_dict={x: minibatch_x, y: minibatch_y}) / total_batch

            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost))

                accuracy = sess.run(eval_op, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})

                print("Validation Error:", (1 - accuracy))

                summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y})
                summary_writer.add_summary(summary_str, sess.run(global_step))

                saver.save(sess, "logistic_logs/model-checkpoint", global_step=global_step)

        print("Optimization Finished!")

        accuracy = sess.run(eval_op, feed_dict={x: mnist.test.images, y: mnist.test.labels})

        print("Test Accuracy:", accuracy)
