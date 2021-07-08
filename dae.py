#!/usr/bin/env python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np
import random as my
from functools import partial
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

def load_matrix(file, random_state=42):
    ratings = pd.read_csv(file, sep = ' ', names=['userId', 'movieId', 'rating'], dtype=np.float32)  
    
    indices = range(len(ratings))
    train_val_indices, test_indices = train_test_split(indices, test_size=0.9, random_state=random_state)
    #train_indices, val_indices = train_test_split(train_val_indices, test_size=0.1, random_state=random_state)
    
    movie_idxs = {}
    user_idxs = {}
    def get_user_idx(user_id):
        if not user_id in user_idxs:
            user_idxs[user_id] = len(user_idxs)
        return user_idxs[user_id]
    
    def get_movie_idx(movie_id):
        if not movie_id in movie_idxs:
            movie_idxs[movie_id] = len(movie_idxs)
        return movie_idxs[movie_id]    

    num_users = ratings.userId.nunique()
    num_movies = ratings.movieId.nunique()
    data = {
        'train': {
            'mask': np.zeros((num_users, num_movies), dtype=np.float32),
            'users': set(),
            'movies': set(),
            'ratings': np.zeros((num_users, num_movies), dtype=np.float32),
        },
        
        'test': {
            'mask': np.zeros((num_users, num_movies), dtype=np.float32),
            'users': set(),
            'movies': set(),
            'ratings': np.zeros((num_users, num_movies), dtype=np.float32),
        },
    }

    for indices, k in [(train_val_indices, 'train'), (test_indices, 'test')]:
        for row in ratings.iloc[indices].itertuples():
            user_idx = get_user_idx(row.userId)
            movie_idx = get_movie_idx(row.movieId)
            data[k]['ratings'][user_idx, movie_idx] = row.rating
            data[k]['mask'][user_idx, movie_idx] = 1
            data[k]['users'].add(user_idx)
            data[k]['movies'].add(movie_idx)
            if (k == 'test'):
                data['train']['ratings'][user_idx, movie_idx] = np.random.randint(0, high = 5)
    return data
dp_dict = np.exp(my.uniform(1.85, 1.95))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    
    return tf.Variable(initial)


def dae(inputs_rating, n_layer, shape_list_rating, ratio=0.5, activation_function = None):
    '''
    rating info: user_id, item_id, ratings

    :param inputs_rating:
    :param n_layer:
    :param shape_list:
    :param ratio:
    :param active_function:
    :return:
    '''

    # Encoder
    name_encoder_rating = []
    weights = []
    for i in range(n_layer):
        name_encoder_rating.append('rating_encoder'+str(i+1))
        weights.append('weights'+str(i+1))
        if i == 0:
            locals()[weights[i]] = weight_variable([shape_list_rating[i], shape_list_rating[i+1]])
            bias = tf.Variable(tf.constant(0.0, shape=[shape_list_rating[i+1]], dtype=tf.float32))
            locals()[name_encoder_rating[i]] = activation_function(tf.matmul(inputs_rating, locals()[weights[i]])+ bias)
        else:
            locals()[weights[i]] = weight_variable([shape_list_rating[i], shape_list_rating[i+1]])
            bias = tf.Variable(tf.constant(0.0, shape=[shape_list_rating[i+1]], dtype=tf.float32))
            locals()[name_encoder_rating[i]] = activation_function(tf.matmul(locals()[name_encoder_rating[i-1]], locals()[weights[i]])+ bias)

    hidden_layer = locals()[name_encoder_rating[n_layer-1]]
    last_weight =  locals()[weights[n_layer-1]]
    # Decoder
    name_decoder_rating = []
    for i in range(n_layer):
        name_decoder_rating.append('rating_decoder'+str(i+1))
        if i == 0:
            bias = tf.Variable(tf.constant(0.0, shape=[shape_list_rating[n_layer+1]], dtype=tf.float32))
            locals()[name_decoder_rating[i]] = activation_function(tf.matmul(hidden_layer, weight_variable([shape_list_rating[n_layer], shape_list_rating[n_layer+1]]) + bias))
        else:
            bias = tf.Variable(tf.constant(0.0, shape=[shape_list_rating[n_layer+i+1]], dtype=tf.float32))
            locals()[name_decoder_rating[i]] = activation_function(tf.matmul(locals()[name_decoder_rating[i-1]], weight_variable([shape_list_rating[n_layer+i], shape_list_rating[n_layer+i+1]]) + bias))
    
    transformed_rating = locals()[name_decoder_rating[n_layer-1]]

    return hidden_layer, transformed_rating

def get_batch(dataset, index, n_batches, batch_size):
    if index == n_batches - 1:
        # print('this')
        r = dataset[batch_size*index:]
    else:
        r = dataset[batch_size*index: batch_size*(index+1)]
    input_mask = (r != 0).astype(np.float32)
    # print("rating", r.shape)
    return r, input_mask

'''
def validation_loss(x_train, x_test):
        
         Computing the loss during the validation time.
    		
    	  @param x_train: training data samples
    	  @param x_test: test data samples
    		
    	  @return networks predictions
    	  @return root mean squared error loss between the predicted and actual ratings
    	  
        
        outputs = sess.inference(x_train) # use training sample to make prediction
        mask=tf.where(tf.equal(x_test,0.0), tf.zeros_like(x_test), x_test) # identify the zero values in the test ste
        num_test_labels=tf.cast(tf.count_nonzero(mask),dtype=tf.float32) # count the number of non zero values
        bool_mask=tf.cast(mask,dtype=tf.bool) 
        outputs=tf.where(bool_mask, outputs, tf.zeros_like(outputs))
    
        MSE_loss=self._compute_loss(outputs, x_test, num_test_labels)
        RMSE_loss=tf.sqrt(MSE_loss)
        
        ab_ops=tf.div(tf.reduce_sum(tf.abs(tf.subtract(x_test,outputs))),num_test_labels)
            
        return outputs, x_test, RMSE_loss, ab_ops
'''

if __name__ == "__main__":
    tf.reset_default_graph()

    inputs_rating = load_matrix('ratings.txt')
    # compatible matrix
    num_movies = inputs_rating['train']['ratings'].shape[1]
    
    print(num_movies)
    #n_items = 2071 
    n_users = 1508
    rating_vector = tf.placeholder('float', [None, num_movies], name = 'rating_vector')
    
    input_mask = tf.placeholder(dtype=tf.float32, shape=[None, num_movies], name='input_mask')
    
    output_mask = tf.placeholder(dtype=tf.float32, shape=[None, num_movies], name='output_mask')
    

    _, transformed_rating = dae(rating_vector, 2, [num_movies, 256, 128, 256, num_movies],  
                                                    ratio=0.5, activation_function=tf.nn.relu)

    network = transformed_rating
    r_pred = transformed_rating

    loss_op = tf.math.divide(tf.reduce_sum(tf.square(tf.multiply((rating_vector - transformed_rating), output_mask))), num_movies)
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    loss_op = loss_op + 0.01 * l2_loss
    # outputs=self.inference(x_train) # use training sample to make prediction
    #     mask=tf.where(tf.equal(x_test,0.0), tf.zeros_like(x_test), x_test) # identify the zero values in the test ste
    #     num_test_labels=tf.cast(tf.count_nonzero(mask),dtype=tf.float32) # count the number of non zero values
    #     bool_mask=tf.cast(mask,dtype=tf.bool) 
    #     outputs=tf.where(bool_mask, outputs, tf.zeros_like(outputs))
    
    #     MSE_loss=self._compute_loss(outputs, x_test, num_test_labels)
    #     RMSE_loss=tf.sqrt(MSE_loss)
        
    #     ab_ops=tf.div(tf.reduce_sum(tf.abs(tf.subtract(x_test,outputs))),num_test_labels)

    n_epochs = 500
    batch_size = 100
    shuffle_batch = False

    # train
    n_train_batches = (len(inputs_rating['train']['ratings']) // batch_size) + 1
    
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss_op)
    with tf.Session() as sess:
        all_variables = tf.trainable_variables()
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            prediction = np.zeros((batch_size, num_movies))
            total_cost = 0
            minibatch_indices = range(n_train_batches)
            if shuffle_batch:
                np.random.shuffle(minibatch_indices)
            for minibatch_index in minibatch_indices:
                rating_batch, r_input_mask = get_batch(inputs_rating['train']['ratings'], minibatch_index, n_train_batches, batch_size)
                
                # print(rating_batch.shape)
                feed_dict = { rating_vector: rating_batch, input_mask: r_input_mask, output_mask: r_input_mask }

                #feed_dict.update(network.all_drop)
                op_cost, pred_batch, _ = sess.run([loss_op, r_pred, train_op], feed_dict = feed_dict)
                total_cost += op_cost

                prediction = np.concatenate((prediction, pred_batch), axis=0)
                #print (total_cost)
            #avg_cost = total_cost / n_users
            #model_improved = True if old_avg_cost == None or old_avg_cost > avg_cost else False
            #total_cost /= batch_size
            print ('Iteration: %s                    Cost: %s' % (epoch, total_cost))
            #ratings = inputs_rating['ratings']
            dataset = inputs_rating['test']
            this_input_mask = dataset['mask']
            #this_output_mask = (ratings != 0).astype(np.float32)
            
            prediction = np.delete(prediction, slice(0, batch_size), axis=0)
            prediction = np.multiply(prediction, this_input_mask)

            unseen_users = dataset['users']
            unseen_movies = dataset['movies']
            # for user in unseen_users:
            #     for movie in unseen_movies:
            #         if this_input_mask[user, movie] == 1:
            #             print(prediction[user, movie])
            mae = np.sum(np.absolute(prediction - dataset['ratings'])) / np.count_nonzero(this_input_mask)
            rmse = np.sqrt(np.sum((prediction - dataset['ratings'])**2) / np.count_nonzero(this_input_mask))
            print('MAE: %f    RMSE: %f' % (mae, rmse))
