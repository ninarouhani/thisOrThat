import tensorflow as tf
import numpy as np
from data import gen_data
rng = np.random
import pdb

# FAKE DATA
sigmoid = lambda x: 1. / (1. + np.exp(-x))

N_PROD = 101
D_PROD = 13

N_PERSON = 17
D_PERSON_STYLE = D_PROD

data = gen_data(N_PERSON, N_PROD, D_PROD)

#### TENSORFLOW MODEL
def ph(shape, dtype=tf.float32):
    return tf.placeholder(dtype, shape)

with tf.name_scope('product') as scope:
    prod_embeddings = tf.Variable(data['prod_vecs'].astype(np.float32))  # learned codes for each product
    prod_embedding2 = tf.Variable(tf.random_normal([N_PROD, 333]))  # learned codes for each product

# placeholders are network inputs
with tf.name_scope('user') as scope:
    i_user = ph((None), tf.int32)  # user index
    i_user_swiped = ph((None), tf.int32)  # indices of products that user swiped
    user_match = ph((None), tf.bool)  # if they say yes
    # user_weights = ph((D_PROD, 1))  #
    user_weight_embeddings = tf.Variable(tf.random_normal([N_PERSON, D_PERSON_STYLE]))

# look up products
user_prods = tf.nn.embedding_lookup(prod_embeddings, i_user_swiped)  # get the products the person responded to
user_weights = tf.nn.embedding_lookup(user_weight_embeddings, i_user)  # get the products the person responded to
hat_match_logit = tf.matmul(user_prods, tf.transpose(user_weights))  # guess about if they like or not

# get total regression loss
match_loss_indiv = tf.nn.sigmoid_cross_entropy_with_logits(
        tf.squeeze(hat_match_logit, [1]),
        tf.cast(user_match, tf.float32))  # how close each model was to person's guess
match_loss = tf.reduce_sum(match_loss_indiv)  # pool all people


# let's make a demo run
LRATE = 1e-3

def user_updater(user_opt):
    gvs = user_opt.compute_gradients(match_loss)
    user_gvs = [(g, v) for g, v in gvs if 'user' in v.name]
    product_gvs = [(g, v) for g, v in gvs if 'product' in v.name]
    user_update = user_opt.apply_gradients(user_gvs)
    return user_update, product_gvs


def product_updater(product_opt, product_gvs):
    zip_pgvs = zip(*product_gvs)  # zip_pgvs[i][j][k] - i is ith var, j is jth user, k={0, 1} is {grad, var}
    grads, _vars = [], []
    for user_gvs in zip_pgvs:  # for each product variable...
        if user_gvs[0][0] is not None:   # if product variable has a gradient...
            _vars.append(user_gvs[0][1])  # store variable
            grads0 = [tf.expand_dims(gg, 0) for gg, _ in user_gvs]  # pool gradients from each subject
            grads.append(tf.reduce_sum(tf.concat(0, grads0), [0]))  # get average/summed gradient
    prod_gvs = [(g, v) for g, v in zip(*[grads, _vars])]
    product_update = product_opt.apply_gradients(prod_gvs)
    return product_update


user_updates, product_gvs = [], []
for ii in xrange(N_PERSON):
    uu, pgv = user_updater(tf.train.AdamOptimizer(LRATE))
    user_updates.append(uu)
    # product_gvs += pgv
    product_gvs.append(pgv)

# user_updates, product_gvs = [user_updater(tf.train.AdamOptimizer(LRATE)) for _ in xrange(N_PERSON)]
product_opt = tf.train.AdamOptimizer(LRATE)
product_updater = product_updater(product_opt, product_gvs)


# with tf.Session() as sess:

