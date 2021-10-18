import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from lightfm import LightFM
from lib import *

class BaselineSolver:
    def __init__(self, users, orgs):
        self.users = users.copy()
        self.orgs = orgs.copy()
        self.spb_orgs = orgs[orgs['city'] == 'spb']['org_id']
        self.msk_orgs = orgs[orgs['city'] == 'msk']['org_id']
    
    def fit(self, reviews, topk=N):
        tourist_reviews = reviews[reviews['rating'] >= 4]
        tourist_reviews = tourist_reviews[tourist_reviews['user_city'] != tourist_reviews['org_city']]

        msk_rev = tourist_reviews[tourist_reviews['org_city'] == 'msk']['org_id']
        self.msk_rev = msk_rev.value_counts().index[:topk].to_list()
        spb_rev = tourist_reviews[tourist_reviews['org_city'] == 'spb']['org_id']
        self.spb_rev = spb_rev.value_counts().index[:topk].to_list()
        

    def predict(self, X_test, path=None):
        test_users_with_locations = X_test.merge(self.users, on='user_id')
        choose = lambda x: self.spb_rev if x['city'] == 'msk' else self.msk_rev
        target = test_users_with_locations.apply(choose, axis=1)
        predictions = X_test.copy()
        predictions['target'] = target
        if path != None:
            predictions_str = predictions.copy()
            predictions_str['target'] = predictions_str['target'].apply(lambda x: ' '.join(map(str, x)))
            predictions_str.to_csv(path, index=None)

        return predictions

class BaselineMod1Solver:
    def __init__(self, users, orgs):
        self.users = users
        self.orgs = orgs
        self.spb_orgs = orgs[orgs['city'] == 'spb']['org_id']
        self.msk_orgs = orgs[orgs['city'] == 'msk']['org_id']
    
    def fit(self, reviews):
        tourist_reviews = reviews[reviews['rating'] >= 4]
        tourist_reviews = tourist_reviews[tourist_reviews['user_city'] != tourist_reviews['org_city']]

        def fil(r):
            ts_min = r['ts'].min()
            ts_max = r['ts'].max()
            return r[r['ts'] > (ts_min + (ts_max - ts_min) * (9/10))]['org_id'].value_counts().index[:N].to_list()
        
        def mer(a, b):
            ans = []
            i, j = 0, 0
            while len(ans) < N:
                if i == j:
                    if not (a[i] in ans):
                        ans.append(a[i])
                    i += 1
                else:
                    if not (b[j] in ans):
                        ans.append(b[j])
                    j += 1
            return ans


        msk_rev1 = tourist_reviews[tourist_reviews['org_city'] == 'msk']['org_id']
        msk_rev2 = tourist_reviews[tourist_reviews['org_city'] == 'msk']
        self.msk_rev = mer(msk_rev1.value_counts().index[:N].to_list(), fil(msk_rev2))
   
        spb_rev1 = tourist_reviews[tourist_reviews['org_city'] == 'spb']['org_id']
        spb_rev2 = tourist_reviews[tourist_reviews['org_city'] == 'spb']
        self.spb_rev = mer(spb_rev1.value_counts().index[:N].to_list(), fil(spb_rev2))
        

    def predict(self, X_test, path=None):
        test_users_with_locations = X_test.merge(self.users, on='user_id')
        choose = lambda x: self.spb_rev if x['city'] == 'msk' else self.msk_rev
        target = test_users_with_locations.apply(choose, axis=1)
        predictions = X_test.copy()
        predictions['target'] = target
        if path != None:
            predictions_str = predictions.copy()
            predictions_str['target'] = predictions_str['target'].apply(lambda x: ' '.join(map(str, x)))
            predictions_str.to_csv(path, index=None)

        return predictions

class BaseLightFMSolver:
    def __init__(self, users, orgs):
        self.users = users.copy()
        self.orgs = orgs.copy()
        self.cities = ['msk', 'spb']
        self.orgs_c = dict()
    
    def fit(self, _reviews, min_pos_rating=1):
        reviews = _reviews.copy()
        self.orgs, reviews = filter_min_pos_rating(reviews, self.orgs, min_pos_rating)
        reviews = filter_reviews(reviews, users=self.users, orgs=self.orgs)
        self.org_ctoi, self.org_itoc = create_mappings(self.orgs['org_id'])
        self.user_ctoi, self.user_itoc = create_mappings(self.users['user_id'])
        for city in self.cities:
            self.orgs_c[city] = self.orgs[self.orgs['city'] == city]['org_id'].map(self.org_ctoi).values
        
        cd_reviews = reviews[['user_id', 'org_id', 'rating']].groupby(['user_id', 'org_id']).mean().reset_index()
        cd_reviews['rating'] = cd_reviews['rating'].apply(lambda x: 1 if x >= 4 else -1)
        I = cd_reviews['user_id'].map(self.user_ctoi)
        J = cd_reviews['org_id'].map(self.org_ctoi)
        X = cd_reviews['rating']
        self.train = sparse.coo_matrix((X, (I, J)), shape=(max(self.user_itoc) + 1, max(self.org_itoc) + 1))
        self.model = LightFM(loss='warp', learning_rate=0.01, no_components=10, item_alpha=1e-4, user_alpha=1e-4)

    def fit_partial(self, epochs):
        self.model.fit_partial(self.train, epochs=epochs, verbose=True, num_threads=4)

    def predict(self, X_test, path=None, topk=N):
        test_users_with_locations = X_test.merge(self.users, on='user_id', how='left')
        def f(row):
            user = self.user_ctoi[row['user_id']]
            org_city = get_other_city(row['city'])
            orgs_in_city = self.orgs_c[org_city]
            res = self.model.predict(np.full(len(orgs_in_city), user), orgs_in_city)
            ind = np.argpartition(res, -topk)[-topk:]
            ind = np.vectorize(self.org_itoc.get)(orgs_in_city[np.flip(ind[np.argsort(res[ind])])])
            return ind
        target = test_users_with_locations.progress_apply(f, axis=1)
        predictions = test_users_with_locations.copy()
        predictions['target'] = target
        if path != None:
            save_predictions_to_file(predictions, path)

        return predictions

class SplitLightFMSolver:
    def __init__(self, users, orgs):
        self.users = users
        self.lfm_msk = BaseLightFMSolver(users[users['city'] == 'msk'], orgs[orgs['city'] == 'spb'])
        self.lfm_spb = BaseLightFMSolver(users[users['city'] == 'spb'], orgs[orgs['city'] == 'msk'])
    
    def fit(self, reviews, min_pos_rating=1):
        self.lfm_msk.fit(reviews, min_pos_rating=min_pos_rating)
        self.lfm_spb.fit(reviews, min_pos_rating=min_pos_rating)

    def fit_partial(self, epochs):
        self.lfm_msk.fit_partial(epochs)
        self.lfm_spb.fit_partial(epochs)

    def predict(self, X_test, path=None, topk=N):
        Xtm = X_test.merge(self.users, on='user_id')
        test_msk = Xtm[Xtm['city'] == 'msk'][['user_id']]
        test_spb = Xtm[Xtm['city'] == 'spb'][['user_id']]
        predictions = pd.concat([self.lfm_msk.predict(test_msk, topk=topk), self.lfm_spb.predict(test_spb, topk=topk)])[['user_id', 'target']]
        if path != None:
            save_predictions_to_file(predictions, path)
        return predictions