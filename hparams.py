# _*_ coding:utf-8 _*_

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# directory
flags.DEFINE_string('log_dir', './log/', 'log path.')
flags.DEFINE_string('data_dir', './data/', 'data path.')
flags.DEFINE_string('model_dir', './model/', 'model save path.')

# model version
flags.DEFINE_integer('model_version', 0, 'model version.') 
flags.DEFINE_string('model_name', 'GCN', 'model name.') 
flags.DEFINE_string('model_date', '0000-00-00', 'model date.') 

# data params
flags.DEFINE_integer('feature_dim', 1433, 'the original feature dimensions')
flags.DEFINE_integer('label_kinds', 7, 'the label count.')
flags.DEFINE_float('train_ratio', 0.6, 'the ratio of training data.')
flags.DEFINE_float('test_ratio', 0.2, 'the ratio of testing data.')
flags.DEFINE_float('val_ratio', 0.2, 'the ration of validation data.')
flags.DEFINE_boolean('reweight_adj', False, "whether or not reweight adjacency matrix")

# model params
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 16, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('hidden4', 8, 'Number of units in hidden layer 4.')
flags.DEFINE_integer('hidden5', 4, 'Number of units in hidden layer 5.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')

# settings for attention
flags.DEFINE_boolean('attention', True, 'whether or not add attention mechanism')

# settings for residual
flags.DEFINE_boolean("residual", True, 'whether or not use residual architecture')
flags.DEFINE_boolean("multi_weight", False, 'whether or not weight each graph seperately')
flags.DEFINE_integer('adj_power', 1, 'the power of adjacency matrix')

# hyper nodes
flags.DEFINE_integer('minimum_subgraph_size', 5, 'filter too small sub-graph')
flags.DEFINE_integer('nearest_neighbor_K', 5, 'hyper nodes K-nearest neighbors')
flags.DEFINE_float('beta', 0.5, "the weight between subgraph embedding and own embedding")


def create():
    return FLAGS

