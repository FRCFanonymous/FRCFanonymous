import math
import os
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from time import time
import argparse
import LoadData_FCCF as DATA


os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='frappe',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=32,
                        help='Number of hidden factors.')
    parser.add_argument('--relation_layers', nargs='?', default='[128, 4]',
                        help="Size of each relation layer.")
    parser.add_argument('--deep_layers', nargs='?', default='[64]',
                        help="Size of each neuralFM layer.")
    parser.add_argument('--reg_scale', type=float, default=0.0,
                        help='Regularizer scale value')
    parser.add_argument('--keep_prob', nargs='?', default='[1, 1, 1]', 
                    help='dropout ratio')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per epoch')
    parser.add_argument('--RFM', type=int, default=1,
                    help='Whether to perform Relational network(0 or 1)')
    parser.add_argument('--NFM', type=int, default=0,
                    help='Whether to perform deep neural FM (0 or 1)')
    parser.add_argument('--AFM', type=int, default=1,
                    help='Whether to perform deep attentional FM (0 or 1)')


    return parser.parse_args()

class FCCF(BaseEstimator, TransformerMixin):
    def __init__(self, RFM, NFM, AFM, features_M, features_dim, hidden_factor, 
                 relation_layers, deep_layers, epoch, batch_size, learning_rate, reg_scale, keep,
                 verbose, random_seed=2016):
        # bind params to class
        self.RFM = RFM
        self.NFM = NFM
        self.AFM = AFM
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.relation_layers = relation_layers
        self.deep_layers = deep_layers
        self.features_M = features_M
        self.features_dim = features_dim
        self.reg_scale = reg_scale
        self.keep = keep
        self.epoch = epoch
        self.random_seed = random_seed
        self.verbose = verbose
        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []
 
        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default(): 
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.train_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.dropout_keep = tf.placeholder(tf.float32, shape=[len(self.keep)])
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._initialize_weights()
            nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_scale)
            # Model.

            self.FCCF = self.permutate(nonzero_embeddings, regularizer)
            self.cFM = self.FCCF
            self.FCCF = tf.reduce_sum(self.FCCF, 1)
            self.FCCF = tf.nn.dropout(self.FCCF, self.dropout_keep[1]) # dropout at the FM layer

            if not self.NFM:
                self.FCCF = tf.reduce_sum(self.FCCF, 1, keepdims=True)  # None * 1
            else:
                self.FCCF = self.deep_FM(self.FCCF, regularizer)

            # _________out _________
            self.feature_analysis = self.FCCF
            self.Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features) , 1)  # None * 1
            #self.Feature_bias = tf.layers.dense(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features) , 1)  # None * 1
            Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
            self.out = tf.add_n([self.feature_analysis, self.Feature_bias, Bias])  # None * 1
            #self.out = tf.add_n([self.Bilinear, Bias])  # None * 1
            self.out_con = tf.concat([self.feature_analysis, self.Feature_bias, Bias], 1)


            #_____get each component value__________
            self.components = self.cFM
            self.components = tf.reduce_sum(self.components, 2)
            self.com_sum = tf.reduce_sum(self.components, 1, keepdims=True)
            self.constitute = tf.concat([self.components, self.com_sum, tf.add(self.Feature_bias, Bias), self.out], 1)

            # Compute the loss.
            try:
                self.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                self.reg_losses = tf.add_n(self.reg_losses)
            except:
                self.reg_losses = 0
            
            if self.reg_scale> 0:
                self.loss = tf.nn.l2_loss(
                    tf.subtract(self.train_labels, self.out)) + self.reg_losses  # regulizer
            else:
                print("no relularizer")
                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out)) 

            # Optimizer.
            self.optimizer = tf.train.AdagradOptimizer(
                learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)

            # init
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape() 
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters


    def permutate(self, embeddings, regularizer):
        v_perm_list = []
        attention_perm_list = [] # only will be used when attention is used

        for i in range(self.features_dim):
            starter = i if self.RFM else i 
            for j in range(starter, self.features_dim):
                v_i = embeddings[:,i,:]
                v_j = embeddings[:,j,:]
                #dot product of two vectors
                if self.RFM:
                    if i != j:
                        v_output = self.relation_network(v_i, v_j, regularizer) 
                        #v_output = tf.multiply(v_i, v_j)
                    else:
                        v_output = self.single_network(v_i)
                else:
                    v_output = tf.multiply(v_i, v_j)
                if self.AFM:
                    att_output = tf.reduce_sum(tf.multiply(v_i, v_j), 1, keepdims=True) #dot product
                    #att_output = tf.layers.dense(tf.multiply(v_i, v_j), 1)  #MLP
                    #att_output = tf.reduce_sum(v_output, 1, keepdims=True)
                    attention_perm_list.append(att_output)
                    
                v_perm_list.append(tf.expand_dims(v_output, 1))
        v_concat = tf.concat(v_perm_list, axis=1)  #(None, num_permutation, self.outlayer) 

        if self.AFM:
            v_att = tf.expand_dims(tf.concat(attention_perm_list, 1), 2)
            attention_weights = tf.nn.softmax(v_att, axis=1)        
            return tf.multiply(attention_weights, v_concat)
        else:
            return v_concat


    def weight_init(self, input_dim, output_dim):
        glorot = np.sqrt(2.0 / (input_dim + output_dim))
        weights_kernel = tf.initializers.random_normal(mean=0.0, stddev=glorot) 
        weights_bias = tf.initializers.random_normal(mean=0.0, stddev=glorot)  

        return weights_kernel, weights_bias
      
    
    def relation_network(self, v1, v2, reg, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('relation_net', reuse=reuse) as scope:
            layers = tf.multiply(v1, v2)
            #layers = tf.concat([v1, v2], 1)

            for i in range(len(self.relation_layers)):
                if i == 0:
                    input_dim = self.hidden_factor
                else:
                    input_dim = self.relation_layers[i-1]
                output_dim = self.relation_layers[i]

                weights_kernel, weights_bias = self.weight_init(input_dim, output_dim)
                if i != len(self.relation_layers) - 1:
                    layers = tf.layers.dense(layers, self.relation_layers[i], activation=tf.nn.relu, kernel_initializer=weights_kernel, bias_initializer=weights_bias)
                else:
                    layers = tf.layers.dense(layers, self.relation_layers[i], kernel_initializer=weights_kernel, bias_initializer=weights_bias)
                layers = tf.layers.dropout(layers, rate=self.dropout_keep[0])


        return layers 
    
    def single_network(self, v_i):
        with tf.variable_scope('single_net', reuse=tf.AUTO_REUSE) as scope:
            layers = v_i
            #layers = tf.concat([v1, v2], 1)

            for i in range(len(self.relation_layers)):
                if i == 0:
                    input_dim = self.hidden_factor
                else:
                    input_dim = self.relation_layers[i-1]
                output_dim = self.relation_layers[i]

                weights_kernel, weights_bias = self.weight_init(input_dim, output_dim)
                if i != len(self.relation_layers) - 1:
                    layers = tf.layers.dense(layers, self.relation_layers[i], activation=tf.nn.relu, kernel_initializer=weights_kernel, bias_initializer=weights_bias)
                else:
                    layers = tf.layers.dense(layers, self.relation_layers[i], kernel_initializer=weights_kernel, bias_initializer=weights_bias)
                layers = tf.layers.dropout(layers, rate=self.dropout_keep[0])
        return layers

       
    def deep_FM(self, xinput, reg):
        layers = xinput
        xinput_dim = layers.get_shape().as_list()[1]
        # ________ Deep Layers __________
        for i in range(len(self.deep_layers)): 
            if i == 0:
                input_dim = xinput_dim 
            else:
                input_dim = self.deep_layers[i-1]
            output_dim = self.deep_layers[i]

            weights_kernel, weights_bias = self.weight_init(input_dim, output_dim)
            layers = tf.layers.dense(layers, self.deep_layers[i], activation=tf.nn.relu, kernel_initializer=weights_kernel, bias_initializer=weights_bias)
            layers = tf.layers.dropout(layers, rate=self.dropout_keep[2])
        
        if len(self.deep_layers) > 0:
            weights_kernel, weights_bias = self.weight_init(self.deep_layers[-1], 1)
            layers = tf.layers.dense(layers, 1, kernel_initializer=weights_kernel, bias_initializer=weights_bias)
        else:
            layers = tf.layers.dense(layers, 1)

        return layers
            
    def _initialize_weights(self):
        all_weights = dict()
    
        all_weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.features_M, self.hidden_factor], 0.0, 0.01),
            name='feature_embeddings')  # features_M * K
        all_weights['feature_bias'] = tf.Variable(
            tf.random_uniform([self.features_M, 1], 0.0, 0.0), name='feature_bias')  # features_M * 1
        all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1
        # relation layers
        return all_weights

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.train_features: data['X'], self.train_labels: data['Y'], self.dropout_keep: self.keep, self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        #p_fm, s_fm = self.sess.run((self.perm_result, self.sqr_result), feed_dict=feed_dict)
        #print(p_fm[0], s_fm[0]) 
        return loss

    def get_random_block_from_data(self, data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X , Y = [], []
        # forward get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X) < batch_size and i >= 0:
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i - 1
            else:
                break
        return {'X': X, 'Y': Y}

    def shuffle_in_unison_scary(self, a, b): # shuffle two lists simutaneously
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def train(self, Train_data, Validation_data, Test_data):  # fit a dataset
        # Check Init performance
        if self.verbose > 0:
            t2 = time()
            #init_train = self.evaluate(Train_data)
            init_train = 0
            init_valid = self.evaluate(Validation_data)
            init_test = self.evaluate(Test_data)
            print("Init: \t train=%.4f, validation=%.4f, test=%.4f [%.1f s]" %(init_train, init_valid, init_test, time()-t2))

        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Train_data['X'], Train_data['Y'])
            total_batch = int(len(Train_data['Y']) / self.batch_size)
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                # Fit training
                self.partial_fit(batch_xs)
            t2 = time()

            # output validation
            #train_result = self.evaluate(Train_data)
            train_result = 0
            valid_result = self.evaluate(Validation_data)
            test_result = self.evaluate(Test_data)
            test_const, test_con = self.generate_constitute(Test_data)

            self.train_rmse.append(train_result)
            self.valid_rmse.append(valid_result)
            self.test_rmse.append(test_result)
            if self.verbose > 0 and epoch%self.verbose == 0:
                print("Epoch %d [%.1f s]\t\ttrain=%.4f, validation=%.4f, test=%.4f [%.1f s]"
                      %(epoch+1, t2-t1, train_result, valid_result, test_result, time()-t2))
            if self.eva_termination(self.valid_rmse):
                break

    def eva_termination(self, valid):
        if len(valid) > 5:
            if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                return True
        return False

    def generate_constitute(self, data):
        feed_dict = {
            self.train_features: data['X'], 
            self.train_labels: [[y] for y in data['Y']], 
            self.dropout_keep: [1.0 for i in range(len(self.keep))], 
            self.train_phase: False}
        constitute, out_con = self.sess.run((self.constitute, self.out_con), feed_dict=feed_dict)

        return constitute, out_con
 

    def evaluate(self, data):  # evaluate the results for an input set
        num_example = len(data['Y'])
        feed_dict = {
            self.train_features: data['X'], 
            self.train_labels: [[y] for y in data['Y']], 
            self.dropout_keep: [1.0 for i in range(len(self.keep))], 
            self.train_phase: False}
        predictions = self.sess.run((self.out), feed_dict=feed_dict)
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(data['Y'], (num_example,))
        predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
        predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  # bound the higher values
        RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
        return RMSE
    
if __name__ == '__main__':
    # Data loading
    args = parse_args()
    data = DATA.LoadData(args.path, args.dataset)
    if args.verbose > 0:
        if args.RFM:
            print("FM: dataset=%s, factors=%d, #epoch=%d, batch=%d, lr=%.4f, relation_layers=%s, keep=%s, RFM=%s, NFM=%s, AFM=%s"
                %(args.dataset, args.hidden_factor, args.epoch, args.batch_size,
                args.lr, args.relation_layers, args.keep_prob, args.RFM, args.NFM, args.AFM))
        elif args.NFM:
            print("FM: dataset=%s, factors=%d, #epoch=%d, batch=%d, lr=%.4f, deep_layers=%s, keep=%s, RFM=%s, NFM=%s, AFM=%s"
                %(args.dataset, args.hidden_factor, args.epoch, args.batch_size,
                args.lr, args.deep_layers, args.keep_prob, args.RFM, args.NFM, args.AFM))

    # Training
    t1 = time()
    model = FCCF(
        args.RFM, 
        args.NFM,
        args.AFM,
        data.features_M, 
        data.feature_dim, 
        args.hidden_factor, 
        eval(args.relation_layers),
        eval(args.deep_layers),
        args.epoch, 
        args.batch_size, 
        args.lr, 
        args.reg_scale, 
        eval(args.keep_prob), 
        args.verbose)

    model.train(data.Train_data, data.Validation_data, data.Test_data)
    
    # Find the best validation result across iterations
    best_valid_score = 0
    best_valid_score = min(model.valid_rmse)
    best_epoch = model.valid_rmse.index(best_valid_score)
    print ("Best validation epoch = %d\t train = %.4f, valid = %.4f, test = %.4f [%.1f s]" 
           %(best_epoch+1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], model.test_rmse[best_epoch], time()-t1))
