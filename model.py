import tensorflow as tf
import numpy as np
from data import gen_data
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
rng = np.random
import pdb

# FAKE DATA
sigmoid = lambda x: 1. / (1. + np.exp(-x))

N_PROD = 30
D_PROD = 10

N_PERSON = 17
D_PERSON_STYLE = D_PROD

ACTUAL_D_PROD = 10
data = gen_data(N_PERSON, N_PROD, ACTUAL_D_PROD, True)

# train_data = val_data = data
train_data, val_data = {}, {}
for kk in ['user_match', 'user_i_prod_swiped']:
    train_data[kk], val_data[kk] = [None] * N_PERSON, [None] * N_PERSON
    for pi in xrange(N_PERSON):
        train_data[kk][pi] = data[kk][pi][N_PROD/2:]
        val_data[kk][pi] = data[kk][pi][:N_PROD/2]

n_ratings_total = np.sum([dd.size for dd in train_data['user_match']])

#### TENSORFLOW MODEL
def ph(shape, dtype=tf.float32):
    return tf.placeholder(dtype, shape)

with tf.name_scope('product') as scope:
    prod_embeddings = tf.Variable(tf.random_normal([N_PROD, D_PROD]))  # learned codes for each product
    # prod_embeddings = tf.Variable(data['prod_vecs'].astype(np.float32), trainable=False)  # learned codes for each product

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

hat_match_logit = tf.matmul(user_prods, tf.reshape(user_weights, [D_PERSON_STYLE, 1]))  # guess about if they like or not
hat_match_logit = tf.squeeze(hat_match_logit, [1])

# get total regression loss
# how close each model was to person's guess
match_loss_indiv = tf.nn.sigmoid_cross_entropy_with_logits(
        hat_match_logit, 
        tf.squeeze(tf.cast(user_match, tf.float32), [1]))  

match_loss = tf.reduce_sum(match_loss_indiv)  # pool all people
train_match_loss = match_loss
# train_match_loss = train_match_loss +\
#             tf.reduce_mean(tf.square(user_prods)) * 1e+1 +\
#             tf.reduce_mean(tf.square(user_weights)) * 1e+1

hat_match = tf.greater(hat_match_logit, 0.)
n_correct = tf.reduce_sum(
                tf.cast(
                    tf.equal(hat_match, tf.squeeze(user_match)),
                    tf.float32))


# OPTIMIZER FUNCTIONS
def make_user_updater(user_opt, loss0):
    gvs = user_opt.compute_gradients(loss0)
    user_gvs = [(tf.clip_by_value(g, -1., 1.), v) for g, v in gvs if 'user' in v.name]
    product_vg_dict = {v: g for g, v in gvs if 'product' in v.name}
    user_update = user_opt.apply_gradients(user_gvs)
    return user_update, product_vg_dict


def make_gradient_placeholders(product_vg_dict):
    """takes in product_gvs dict made with make_user_updater"""
    placeholder_gradients = {}
    # for _, var in product_gvs:
    for var in product_vg_dict:
        if product_vg_dict[var] is not None:
            ph_shape = [None] + [dd.value for dd in var.get_shape().dims]
            ph_grad = tf.placeholder(tf.float32, ph_shape)
            placeholder_gradients[var] = ph_grad
    return placeholder_gradients


def make_pooled_gvs(placeholder_gradients):
    """takes in placeholders for indiv gradients made with make_gradient_placeholders"""
    pooled_gvs = []
    # for grads, var in placeholder_gradients_var:
    for var, grads in placeholder_gradients.iteritems():
        grad = tf.reduce_sum(grads, [0])
        grad = tf.clip_by_value(grad, -1., 1.)
        pooled_gvs.append((grad, var))
    return pooled_gvs

make_product_updater = lambda opt, product_gvs: opt.apply_gradients(product_gvs)

# TODO: we need a sparse solution
def isv_to_dense(isv):
    """convert IndexedSliceValue tensorflow object to dense numpy array"""
    out = np.zeros(isv.dense_shape)
    out[isv.indices] = isv.values
    return out


# LET'S RUN
# let's make a demo run
LRATE = 1e-3

# build individual style trainers
user_updaters, user_product_gvs = [], []
for ii in xrange(N_PERSON):
    # TODO: should they all share an optimizer?
    # user_opt = tf.train.AdamOptimizer(LRATE)
    user_opt = tf.train.GradientDescentOptimizer(LRATE)
    ud, pgv = make_user_updater(user_opt, train_match_loss)
    user_updaters.append(ud)
    user_product_gvs.append(pgv)


# build pooled product updater
placeholder_gradients = make_gradient_placeholders(user_product_gvs[0])
pooled_product_gvs = make_pooled_gvs(placeholder_gradients)
# product_updater = make_product_updater(tf.train.AdamOptimizer(LRATE), pooled_product_gvs)
product_updater = make_product_updater(tf.train.GradientDescentOptimizer(LRATE), pooled_product_gvs)


def prep_user_fd(data0, i_users0, bs=None):
    i_user_swiped0 = data0['user_i_prod_swiped'][i_users0]
    user_matches0 = data0['user_match'][i_users0]
    if bs is not None:
        ibs = rng.choice(len(i_user_swiped0), bs, replace=False)
        i_user_swiped0 = i_user_swiped0[ibs]
        user_matches0 = user_matches0[ibs]

    return {i_user: i_users0,
            i_user_swiped: i_user_swiped0,
            user_match: user_matches0}


def prep_product_fd(pgvs, placeholder_gradients_dict):
    fd = {}
    for var in pgvs[0]:
        stacked_grads = np.stack([isv_to_dense(pgv[var]) for pgv in pgvs])
        fd[placeholder_gradients_dict[var]] = stacked_grads
    return fd


im = None
# fig, ax = plt.subplots(4, 2)
ff, aa = plt.subplots()


N_UPDATE = int(1e4)
BS = None
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i_step in xrange(N_UPDATE):
        # TEST
        if i_step % 100 == 0:
            print 'step %d' % (i_step)
            for data0, tv in zip(*[[train_data, val_data], ['train', 'val']]):
                l0 = 0.
                n0 = 0.
                n_yes = 0.
                for i_user0 in np.arange(N_PERSON):
                    fd = prep_user_fd(data0, i_user0)
                    l0 += sess.run(match_loss, feed_dict=fd)
                    n0 += sess.run(n_correct, feed_dict=fd)
                    hat_yes = sess.run(hat_match, feed_dict=fd)
                    n_yes += hat_yes.astype(float).sum()
                print tv + ' loss: ' + str(l0 / n_ratings_total)
                print tv + ' acc: ' + str(n0 / n_ratings_total)

            # PLOT
            hm0, pm0 = [], []
            for i_user0 in xrange(N_PERSON):
                fd = prep_user_fd(data, i_user0)
                pm00 = sigmoid(sess.run(hat_match_logit, feed_dict=fd))
                hm00 = sess.run(hat_match, feed_dict=fd)
                pm0.append(pm00)
                hm0.append(hm00)
            pm0 = np.stack(pm0)
            hm0 = np.stack(hm0)
            pe0 = sess.run(prod_embeddings, feed_dict=fd)
            uwe0 = sess.run(user_weight_embeddings, feed_dict=fd)

            plt.ion()
            # if im is None:
            real_p_match = np.squeeze(data['user_p_match'], [2]).T
            diff_sort = np.sort((real_p_match - pm0.T).ravel())

            aa.clear()
            aa.hist(np.sort(real_p_match.ravel()), bins=100, color='b', alpha=0.3)
            aa.hist(np.sort(pm0.ravel()), bins=100, color='g', alpha=0.3)
            aa.hist(np.abs(diff_sort), bins=100, color='r', alpha=0.3)

            plt.draw()

            # [a0.clear() for a0 in fig.axes]
            # im = [[None]*2]*4
            # im[0][0] = ax[0][0].matshow(data['prod_vecs'], aspect='auto', cmap=plt.get_cmap('magma'), vmin=-3., vmax=3.)
            # im[0][1] = ax[0][1].matshow(pe0, aspect='auto', cmap=plt.get_cmap('magma'), vmin=-3., vmax=3.)

            # im[1][0] = ax[1][0].matshow(np.concatenate(data['user_prefs']), aspect='auto', cmap=plt.get_cmap('magma'), vmin=-3., vmax=3.)
            # im[1][1] = ax[1][1].matshow(uwe0, aspect='auto', cmap=plt.get_cmap('magma'), vmin=-3., vmax=3.)

            # im[2][0] = ax[2][0].matshow(real_p_match, aspect='auto', cmap=plt.get_cmap('magma'), vmin=0., vmax=1.)
            # im[2][1] = ax[2][1].matshow(pm0.T, aspect='auto', cmap=plt.get_cmap('magma'), vmin=0., vmax=1.)
            # im[3][0] = ax[3][0].hist(diff_sort, bins=100)
            # # im[3][0] = ax[3][0].matshow(np.abs(real_p_match - pm0.T), aspect='auto', cmap=plt.get_cmap('magma'), vmin=0., vmax=1.)
            # im[3][1] = ax[3][1].matshow(hm0, aspect='auto', cmap=plt.get_cmap('magma'), vmin=0., vmax=1.)
            plt.draw()
            plt.show()
            plt.pause(0.01)
            plt.ioff()

                # print tv + 'p_yes: ' + str(n_yes / n_ratings_total)
        pgvs = []
        # update each user given current product codes
        for i_user0 in rng.permutation(N_PERSON):
            fd = prep_user_fd(train_data, i_user0, BS)
            _, pgv = sess.run([user_updaters[i_user0], user_product_gvs[i_user0]], feed_dict=fd)
            pgvs.append(pgv)

            # fd = prep_user_fd(val_data, i_user0, BS)
            # _, pgv = sess.run([user_updaters[i_user0], user_product_gvs[i_user0]], feed_dict=fd)
            # pgvs.append(pgv)
        # update product codes with average gradients from users
        fd = prep_product_fd(pgvs, placeholder_gradients)
        sess.run(product_updater, feed_dict=fd)
