
import tensorflow as tf
import numpy as np


def var_summary(var_list, printer=None):
    printer = tf.constant(0) if printer is None else printer
    printer = tf.Print(input_=printer, data=[tf.reduce_mean(v) for v in var_list])
    return printer

def categorical_sample_logits(X):
    U = tf.random_uniform(tf.shape(X))
    return tf.argmax(X - tf.log(-tf.log(U)), axis=1)



class MLP(object):
    def __init__(self, name, hidden_sizes, in_dim=32, out_dim=5, trainable=False, lr=0.001, out_type='regress'):
        inp = tf.placeholder(dtype=tf.float32,shape=[None,in_dim])
        x = inp
        with tf.variable_scope(name):
            for i, si in enumerate(hidden_sizes):
                x = tf.layers.dense(inputs=x, activation=tf.nn.relu, name='h{}'.format(i), units=si,trainable=trainable)
            if out_type== 'regress':
                pred = tf.layers.dense(inputs=x, activation=None, units=out_dim, name='output', trainable=trainable)
            else:
                logits = tf.layers.dense(inputs=x, activation=None, units=out_dim, name='logits', trainable=trainable)
                pred = tf.argmax(tf.nn.softmax(logits=logits),axis=1)
                pred = tf.one_hot(pred, depth=out_dim)
                
            def predict(sess, X):
                return sess.run(pred, feed_dict={inp:X})
            self.predict = predict
            my_vars = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
            if trainable:
                y = tf.placeholder(dtype=tf.float32, shape=[None, out_dim])
                if out_type== 'regress':
                    loss = tf.reduce_mean(tf.square(y-pred))
                else:
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits))
                optimizer = tf.train.AdamOptimizer(learning_rate=lr)
                grads_clipped = [tf.clip_by_value(g, -1., 1.) for g,_ in optimizer.compute_gradients(loss, my_vars)]
                opt_op = optimizer.apply_gradients(zip(grads_clipped, my_vars))        
                def run_opt(sess, X, labels):
                    loss_val, _ = sess.run([loss, opt_op], feed_dict={inp:X, y:labels})
                    return loss_val
                self.run_opt = run_opt     
        printables = var_summary(my_vars)
        def printer(sess, X):
            sess.run(printables, feed_dict={inp:X})
            return
        self.printer = printer


ITERS = 1000000
BATCH = 128
IN_DIM = 256
OUT_TYPE = 'classification'

def main():
    tf.reset_default_graph()
    net = MLP('target', hidden_sizes=[128, 128, 64], in_dim=IN_DIM,out_type=OUT_TYPE)
    net2 = MLP('learner', hidden_sizes=[256,128,64,32], trainable=True, in_dim=IN_DIM,out_type=OUT_TYPE)
    def validate(net, net2, z, sess, out_type):
        target = net.predict(X=z, sess=sess)
        pred = net2.predict(X=z, sess=sess)
        if out_type == 'regress':
            target = target.reshape(-1)
            pred = pred.reshape(-1)
            return 1- (np.var(target-pred)/ (np.var(target)+1e-8))
        elif out_type == 'classification':
            target = np.argmax(target, axis=1)
            pred = np.argmax(pred, axis=1)
            return np.mean(np.array(target == pred,dtype=np.float32))
        else:
            raise NotImplementedError

    val_acc = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(ITERS):
            z = np.random.normal(size=[BATCH, IN_DIM])
            if i % 50 == 0:
                var_act = validate(net, net2, z, sess, out_type=OUT_TYPE)
                val_acc.append(var_act)
                #net.printer(sess=sess, X=z)
                print("Iteration {}. Var accouned for {}.".format(i, var_act))
            else:
                loss_val = net2.run_opt(X=z, labels=net.predict(X=z, sess=sess), sess=sess)
                if i % 20 ==0:
                    print("Iteration {}. Loss value {}".format(i, loss_val))

    np.save(file='log',allow_pickle=True,arr=val_acc)

if __name__ == '__main__':
    main()

