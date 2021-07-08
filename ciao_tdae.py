#!/usr/bin/env python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np
from functools import partial
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

def load_matrix(file, random_state=42):
    ratings = pd.read_csv(file, sep = ',', names=['userId', 'movieId', 'catagory', 'ratingID', 'rating'], dtype=np.float32)  
    ratings.drop(['catagory', 'ratingID'], axis=1)
    indices = range(len(ratings))
    train_indices_1, test_indices_1 = train_test_split(indices, test_size=0.2, random_state=random_state)
    train_indices_2, test_indices_2 = train_test_split(train_indices_1, test_size=0.25, random_state=random_state)
    train_indices_3, test_indices_3 = train_test_split(train_indices_2, test_size=0.3333333, random_state=random_state)
    test_indices_4, test_indices_5 = train_test_split(train_indices_3, test_size=0.5, random_state=random_state)
    
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
    print(num_users)
    data = {
        'train_1': {
            'mask': np.zeros((num_users, num_movies), dtype=np.float32),
            'users': set(),
            'movies': set(),
            'ratings': np.zeros((num_users, num_movies), dtype=np.float32),
        },
        'test_1': {
            'mask': np.zeros((num_users, num_movies), dtype=np.float32),
            'users': set(),
            'movies': set(),
            'ratings': np.zeros((num_users, num_movies), dtype=np.float32),
        },
        'train_2': {
            'mask': np.zeros((num_users, num_movies), dtype=np.float32),
            'users': set(),
            'movies': set(),
            'ratings': np.zeros((num_users, num_movies), dtype=np.float32),
        },
        'test_2': {
            'mask': np.zeros((num_users, num_movies), dtype=np.float32),
            'users': set(),
            'movies': set(),
            'ratings': np.zeros((num_users, num_movies), dtype=np.float32),
        },
        'train_3': {
            'mask': np.zeros((num_users, num_movies), dtype=np.float32),
            'users': set(),
            'movies': set(),
            'ratings': np.zeros((num_users, num_movies), dtype=np.float32),
        },
        'test_3': {
            'mask': np.zeros((num_users, num_movies), dtype=np.float32),
            'users': set(),
            'movies': set(),
            'ratings': np.zeros((num_users, num_movies), dtype=np.float32),
        },
        'train_4': {
            'mask': np.zeros((num_users, num_movies), dtype=np.float32),
            'users': set(),
            'movies': set(),
            'ratings': np.zeros((num_users, num_movies), dtype=np.float32),
        },
        'test_4': {
            'mask': np.zeros((num_users, num_movies), dtype=np.float32),
            'users': set(),
            'movies': set(),
            'ratings': np.zeros((num_users, num_movies), dtype=np.float32),
        },
        'train_5': {
            'mask': np.zeros((num_users, num_movies), dtype=np.float32),
            'users': set(),
            'movies': set(),
            'ratings': np.zeros((num_users, num_movies), dtype=np.float32),
        },
        'test_5': {
            'mask': np.zeros((num_users, num_movies), dtype=np.float32),
            'users': set(),
            'movies': set(),
            'ratings': np.zeros((num_users, num_movies), dtype=np.float32),
        },
    }
    for indices, k in [(test_indices_2, 'train_1'), (test_indices_3, 'train_1'),(test_indices_4, 'train_1'),(test_indices_5, 'train_1'),(test_indices_1, 'test_1')]:
        for row in ratings.iloc[indices].itertuples():
            user_idx = get_user_idx(row.userId)
            movie_idx = get_movie_idx(row.movieId)
            data[k]['ratings'][user_idx, movie_idx] = row.rating
            data[k]['mask'][user_idx, movie_idx] = 1
            data[k]['users'].add(user_idx)
            data[k]['movies'].add(movie_idx)
            if (k == 'test_1'):
                data['train_1']['ratings'][user_idx, movie_idx] = np.random.randint(0, high = 5)
    for indices, k in [(test_indices_1, 'train_2'), (test_indices_3, 'train_2'),(test_indices_4, 'train_2'),(test_indices_5, 'train_2'),(test_indices_2, 'test_2')]:
        for row in ratings.iloc[indices].itertuples():
            user_idx = get_user_idx(row.userId)
            movie_idx = get_movie_idx(row.movieId)
            data[k]['ratings'][user_idx, movie_idx] = row.rating
            data[k]['mask'][user_idx, movie_idx] = 1
            data[k]['users'].add(user_idx)
            data[k]['movies'].add(movie_idx)
            if (k == 'test_2'):
                data['train_2']['ratings'][user_idx, movie_idx] = np.random.randint(0, high = 5)
    for indices, k in [(test_indices_1, 'train_3'), (test_indices_2, 'train_3'),(test_indices_4, 'train_3'),(test_indices_5, 'train_3'),(test_indices_3, 'test_3')]:
        for row in ratings.iloc[indices].itertuples():
            user_idx = get_user_idx(row.userId)
            movie_idx = get_movie_idx(row.movieId)
            data[k]['ratings'][user_idx, movie_idx] = row.rating
            data[k]['mask'][user_idx, movie_idx] = 1
            data[k]['users'].add(user_idx)
            data[k]['movies'].add(movie_idx)
            if (k == 'test_3'):
                data['train_3']['ratings'][user_idx, movie_idx] = np.random.randint(0, high = 5)
    for indices, k in [(test_indices_1, 'train_4'), (test_indices_2, 'train_4'),(test_indices_3, 'train_4'),(test_indices_5, 'train_4'),(test_indices_4, 'test_4')]:
        for row in ratings.iloc[indices].itertuples():
            user_idx = get_user_idx(row.userId)
            movie_idx = get_movie_idx(row.movieId)
            data[k]['ratings'][user_idx, movie_idx] = row.rating
            data[k]['mask'][user_idx, movie_idx] = 1
            data[k]['users'].add(user_idx)
            data[k]['movies'].add(movie_idx)
            if (k == 'test_4'):
                data['train_4']['ratings'][user_idx, movie_idx] = np.random.randint(0, high = 5)
    for indices, k in [(test_indices_1, 'train_5'), (test_indices_2, 'train_5'),(test_indices_3, 'train_5'),(test_indices_4, 'train_5'),(test_indices_5, 'test_5')]:
        for row in ratings.iloc[indices].itertuples():
            user_idx = get_user_idx(row.userId)
            movie_idx = get_movie_idx(row.movieId)
            data[k]['ratings'][user_idx, movie_idx] = row.rating
            data[k]['mask'][user_idx, movie_idx] = 1
            data[k]['users'].add(user_idx)
            data[k]['movies'].add(movie_idx)
            if (k == 'test_5'):
                data['train_5']['ratings'][user_idx, movie_idx] = np.random.randint(0, high = 5)

    return data

def load_trust(file):
    df = pd.read_csv(file, sep = ',', names=['truster','trustee','rating'], dtype=np.float32)
    user_idxs = {}
    num_users = df.truster.nunique()
    def get_user_idx(user_id):
        if not user_id in user_idxs:
            user_idxs[user_id] = len(user_idxs)
        return user_idxs[user_id]
    data = {
            'users': set(),
            'ratings': np.zeros((num_users, num_users), dtype=np.float32),
        },
    for row in df.iloc[indices].itertuples():
        user_idx = get_user_idx(row.userId)
        data['users'].add(user_idx)
        data['ratings'][user_idx, user_idx] = row.rating
        user_idx = get_user_idx(row.userId)

    return data['ratings'], np.count_nonzero(df)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def fc_layers(inputs, in_size, out_size, activation_function, drop_p=1.0):
    weights = weight_variable([in_size, out_size])
    bias = weight_variable([out_size])
    combine = tf.matmul(inputs, weights)+ bias
    if activation_function is None:
        drop = tf.nn.dropout(combine, drop_p)
        out = drop
    else:
        activate = activation_function(combine)
        drop = tf.nn.dropout(activate, drop_p)
        out = drop
    return out

def tdae(inputs_rating, inputs_trust, n_layer, shape_list_rating, shape_list_trust, ratio=0.5, active_function=tf.nn.sigmoid):
    '''
    rating info: user_id, item_id, ratings
    trust info: user_id, user_id, trust_score

    :param inputs_rating:
    :param inputs_trust:
    :param n_layer:
    :param shape_list:
    :param ratio:
    :param active_function:
    :return:
    '''

    # Encoder
    name_encoder_rating, name_encoder_trust = [],[]
    for i in range(n_layer):
        name_encoder_rating.append('rating_encoder'+str(i+1))
        name_encoder_trust.append('trust_encoder'+str(i+1))
        if i == 0:
            locals()[name_encoder_rating[i]] = fc_layers(inputs_rating, shape_list_rating[i], shape_list_rating[i+1], activation_function=active_function)
            locals()[name_encoder_trust[i]] = fc_layers(inputs_trust, shape_list_trust[i], shape_list_trust[i+1], activation_function=active_function)
        else:
            previous_layer_rating, previous_layer_trust = locals()[name_encoder_rating[i-1]], locals()[name_encoder_trust[i-1]]
            locals()[name_encoder_rating[i]] = fc_layers(previous_layer_rating,shape_list_rating[i],shape_list_rating[i+1], activation_function=active_function)
            locals()[name_encoder_trust[i]] = fc_layers(previous_layer_trust,shape_list_trust[i],shape_list_trust[i+1], activation_function=active_function)

    # Combination
    hidden_rating = locals()[name_encoder_rating[n_layer-1]]
    hidden_trust = locals()[name_encoder_trust[n_layer-1]]

    hidden_layer = tf.add(ratio*hidden_rating, (1-ratio)*hidden_trust)

    # Decoder
    name_decoder_rating, name_decoder_trust = [],[]
    for i in range(n_layer):
        name_decoder_rating.append('rating_decoder'+str(i+1))
        name_decoder_trust.append('trust_decoder'+str(i+1))
        if i == 0:
            locals()[name_decoder_rating[i]] = fc_layers(hidden_layer, shape_list_rating[n_layer+i], shape_list_rating[n_layer+i+1], activation_function=active_function)
            locals()[name_decoder_trust[i]] = fc_layers(hidden_layer, shape_list_trust[n_layer+i], shape_list_trust[n_layer+i+1], activation_function=active_function)
        else:
            previous_layer_rating, previous_layer_trust = locals()[name_decoder_rating[i-1]], locals()[name_decoder_trust[i-1]]
            locals()[name_decoder_rating[i]] = fc_layers(previous_layer_rating,shape_list_rating[n_layer+i],shape_list_rating[n_layer+i+1], activation_function=active_function)
            locals()[name_decoder_trust[i]] = fc_layers(previous_layer_trust,shape_list_trust[n_layer+i],shape_list_trust[n_layer+i+1], activation_function=active_function)

    transformed_rating = locals()[name_decoder_rating[n_layer-1]]
    transformed_trust = locals()[name_decoder_trust[n_layer-1]]

    return hidden_layer, transformed_rating, transformed_trust

def get_batch(dataset, index, n_batches, batch_size):
    if index == n_batches - 1:
        # print('this')
        r = dataset[batch_size*index:]
    else:
        r = dataset[batch_size*index: batch_size*(index+1)]
    input_mask = (r != 0).astype(np.float32)
    # print("rating", r.shape)
    return r, input_mask

def get_trust_batch(dataset, index, n_batches, batch_size):
    if index == n_batches - 1:
        # print('this')
        r = dataset[batch_size*index:]
    else:
        r = dataset[batch_size*index: batch_size*(index+1)]
    # print("trsut", r.shape)
    if (r.shape[0] > batch_size):
        r = dataset[batch_size*index: batch_size*(index+1)-1]
    return r

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

    inputs_rating = load_matrix('ciao_ratings.txt')
    inputs_trust, num_trust = load_trust('ciao_trusts.txt')
    # compatible matrix
    num_movies = inputs_rating['train_1']['ratings'].shape[1]
    #num_users = num_trust
    
    print(num_movies)

    print(inputs_trust.shape)
    #n_items = 2071 
    n_users = 4300
    rating_vector = tf.placeholder('float', [None, num_movies], name = 'rating_vector')
    trust_vector = tf.placeholder('float', [None, n_users], name = 'trust_vector')
    
    input_mask = tf.placeholder(dtype=tf.float32, shape=[None, num_movies], name='input_mask')
    
    output_mask = tf.placeholder(dtype=tf.float32, shape=[None, num_movies], name='output_mask')
    

    _, transformed_rating, transformed_trust = tdae(rating_vector, trust_vector, 3, [num_movies, 256, 128, 64, 128, 256, num_movies], 
                                                    [n_users, 256, 128, 64, 128, 256, n_users], 
                                                    ratio=0.7, active_function=tf.nn.relu)
    
    network = transformed_rating
    r_pred = transformed_rating

    loss_op = tf.math.divide(tf.reduce_sum(tf.square(tf.multiply((rating_vector - transformed_rating), output_mask))), num_movies)
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    loss_op = loss_op + 0.01 * l2_loss

    n_epochs = 300
    batch_size = 271
    shuffle_batch = False

    # train
    n_train_batches = len(inputs_rating['train_1']['ratings']) // batch_size
    minibatch_indices = range(n_train_batches)

    cost = loss_op
    train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
    with tf.Session() as sess:
        all_variables = tf.trainable_variables()
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            mae = 0
            rmse = 0
            
            if shuffle_batch:
                np.random.shuffle(minibatch_indices)
            
            for train_k, test_k in [('train_1', 'test_1'), ('train_2', 'test_2'),('train_3', 'test_3'),('train_4', 'test_4'),('train_5', 'test_5')]:
                total_cost = 0
                prediction = np.zeros((batch_size, num_movies))
                for minibatch_index in minibatch_indices:
                    
                    rating_batch, r_input_mask = get_batch(inputs_rating[train_k]['ratings'], minibatch_index, n_train_batches, batch_size)
                    trust_batch = get_trust_batch(inputs_trust, minibatch_index, n_train_batches, batch_size)
                    
                    # print(rating_batch.shape)
                    # print(trust_batch.shape)
                    feed_dict = { rating_vector: rating_batch, trust_vector: trust_batch, input_mask: r_input_mask, output_mask: r_input_mask }

                    #feed_dict.update(network.all_drop)
                    op_cost, pred_batch, _ = sess.run([cost, r_pred, train_op], feed_dict = feed_dict)
                    
                    total_cost += op_cost
                    prediction = np.concatenate((prediction, pred_batch), axis=0)

                print ('Iteration: %s                    Cost: %s' % (epoch, total_cost))
                dataset = inputs_rating[test_k]
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

                k_mae = np.sum(np.absolute(prediction - dataset['ratings'])) / np.count_nonzero(this_input_mask)
                k_rmse = np.sqrt(np.sum((prediction - dataset['ratings'])**2) / np.count_nonzero(this_input_mask))
                mae += k_mae
                rmse += k_rmse
            
            print('MAE: %f    RMSE: %f' % (mae/5, rmse/5))
