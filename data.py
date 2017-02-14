import numpy as np
rng = np.random
import pdb

sigmoid = lambda x: 1. / (1. + np.exp(-x))

# FAKE DATA
def gen_data(N_PERSON, N_PROD, D_PROD):

    # N_PROD = 101
    # D_PROD = 13

    # N_PERSON = 17


    prod_vecs = rng.randn(N_PROD, D_PROD)

    user_prefs = [rng.randn(1, D_PROD) for _ in xrange(N_PERSON)]
    user_weights = [rng.randn(D_PROD, 1) for _ in xrange(N_PERSON)]

    # every person looks at a different number of things
    n_ratings = rng.randint(N_PROD, size=N_PERSON) + 1
    i_prod_swiped = [rng.randint(N_PROD, size=nr) for nr in n_ratings]


    # let's see if they swipe right or left!
    i_users = np.arange(N_PERSON)
    matches = [None] * N_PERSON
    for i_user, swiped0, prefs0 in zip(*[i_users, i_prod_swiped, user_prefs]):
        # grab prods
        prods0 = prod_vecs[swiped0]
        # prob of match
        logit_p_match = np.dot(prods0, prefs0.T)
        p_match = sigmoid(logit_p_match)
        # draw match!
        match = (rng.rand(*p_match.shape) > p_match).astype(int)
        # add to our matches list
        matches[i_user] = match


    out = {}
    out['prod_vecs'] = prod_vecs
    out['user_prefs'] = user_prefs
    out['user_weights'] = user_weights
    out['user_n_ratings'] = n_ratings
    out['user_i_prod_swiped'] = i_prod_swiped
    out['user_match'] = matches

    return out
