import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

display_step = 10


def compute_metrices(con_matrix):
    ((tn, fp), (fn, tp)) = con_matrix
    precision = 0
    if (tp + fp) > 0:
        precision = float(tp) / (tp + fp)
    recall = 0
    if (tp + fn) > 0:
        recall = float(tp) / (tp + fn)
    f1 = 0
    if (precision + recall) > 0:
        f1 = 2*precision*recall / (precision + recall)

    return precision, recall, f1


def logistic_regression(features, labels, test_features, test_labels, lexicon, files_prefix='', \
                                    learning_rate=0.0001, learning_rate_decay=0.5, learning_rate_decay_yellows=5,\
                                    max_yellow_cards=10, batch_size=100, training_epochs=10000, n_hidden=16, train=True):

    input_size = test_features.shape[1]
    n_data = test_features.shape[0]
    n_valid = int(n_data*0.1)
    n_train = n_data - n_valid
    if train:
        valid_features = features[:n_valid,:]
        valid_labels = labels[:n_valid,:]

        train_features = features[n_valid:,:]
        train_labels = labels[n_valid:, :]

    n_class = 2
    x = tf.placeholder(tf.float32, [None, input_size])
    y = tf.placeholder(tf.float32, [None, n_class])
    if n_hidden is None:
        logits = tf.layers.dense(inputs=x, units=n_class, activation=tf.nn.relu)
    else :
        h_1 = tf.layers.dense(inputs=x, units=n_hidden, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=h_1, units=n_class, activation=tf.nn.relu)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    pred = tf.nn.softmax(logits)

    tf.summary.scalar('cost', cost)

    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(learning_rate_initial, global_step, 10000, 0.7, staircase=True)
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

    learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
    optimizer = tf.train.AdamOptimizer(learning_rate_placeholder).minimize(cost)

    # Test model
    predictions = tf.argmax(pred, 1)
    correct_prediction = tf.equal(predictions, tf.argmax(y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    save_path = "models/{}model.h5".format(files_prefix)
    # Start training
    with tf.Session() as sess:
        filewriter = tf.summary.FileWriter("log")
        filewriter.add_graph(sess.graph)
        merge = tf.summary.merge_all()

        saver = tf.train.Saver()
        if train:
            # Run the initializer
            sess.run(init)
            epoch_steps = n_train // batch_size
            total_steps = training_epochs * epoch_steps
            best_valid_cost = np.Inf
            y_c = 0
            epoch_costs = []
            # Training cycle
            for step in range(total_steps):

                begin_index = (step * batch_size) % n_train
                end_index = min(begin_index + batch_size, n_data)
                batch_xs = train_features[begin_index:end_index]
                batch_ys = train_labels[begin_index:end_index]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, b = sess.run([optimizer, cost, merge], feed_dict={x: batch_xs, y: batch_ys, learning_rate_placeholder: learning_rate})
                epoch_costs.append(c)
                filewriter.add_summary(b, step)

                # Display logs per epoch step
                if (step + 1) % display_step == 0:
                    pass
                    #print("Epoch:{:04d}/{:d}, yc={:d}, cost={:.9f}".format(step + 1, total_steps, y_c, c))

                if (step + 1) % epoch_steps == 0:
                    c, acc = sess.run([cost, accuracy], feed_dict={x: valid_features, y: valid_labels})

                    if c < best_valid_cost:
                        best_valid_cost = c
                        y_c = 0
                        save_path = saver.save(sess, save_path)
                        print("News best model saved in file: {:s}".format(save_path))
                    else:
                        y_c += 1
                        if y_c % learning_rate_decay_yellows == 0:
                            learning_rate *= learning_rate_decay
                    print("validation: accuracy={:.4f}, cost={:.9f}, yc={}, lr={:.9}".format(acc, c, y_c, learning_rate))
                    if y_c == max_yellow_cards:
                        break

            print("Optimization Finished!")

        saver.restore(sess, save_path)

        acc, predicted_labels = sess.run([accuracy, predictions], feed_dict={x: test_features, y: test_labels})
        con_matrix = confusion_matrix(y_true=np.argmax(test_labels, axis=1), y_pred=predicted_labels)

        precision, recall, f1 = compute_metrices(con_matrix)

        print("Accuracy: {:.4f}, Precision={:.4f}, Recall={:.4f}, F1={:.4f}".format(acc, precision, recall, f1))
        print(con_matrix)

        return predicted_labels
