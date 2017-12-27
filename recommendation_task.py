import numpy as np
from scipy.sparse import coo_matrix
from sklearn.decomposition import NMF


class Recommendation:
    def __init__(self, fold=5, n_component=10, alpha=0.3, l1_ratio=0.3):
        self.fold = fold
        self.n_component = n_component
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def parse(self, f):
        """parsing"""
        print 'parse'
        self.dataList = []
        for line in open(f):
            person = line.split(',')[1:4]
            self.dataList.append(person)  # personID, itemID, qty
        self.dataList.pop(0)
        self.dataList = np.array([[int(i) for i in data] for data in self.dataList])  # list with each element in the form of (userID, itemID, qty)

    def split_data(self):
        """split data into n fold"""
        print 'split data'
        np.random.shuffle(self.dataList)
        l = len(self.dataList)/self.fold
        self.dataList = [self.dataList[i*l: (i+1)*l] for i in range(self.fold-1)] + [self.dataList[(self.fold-1)*l:]]  # each element in the list is splitted data list

    def rec(self, trData, teData):
        """mf or item-based or user-based"""
        print 'rec'
        trainingData = [j for i in trData for j in i]
        userID_to_rowIdx = {}
        itemID_to_colIdx = {}

        # mapping user id to row index and item id to column index
        row_id = col_id = 0
        for uid, iid, rating in trainingData:
            if uid not in userID_to_rowIdx:
                userID_to_rowIdx[uid] = row_id
                row_id += 1
            if iid not in itemID_to_colIdx:
                itemID_to_colIdx[iid] = col_id
                col_id += 1
        trData = [(userID_to_rowIdx[uid], itemID_to_colIdx[iid]) for uid, iid, rating in trainingData]  # list of (user idx, item idx)
        teData = [(userID_to_rowIdx[uid], itemID_to_colIdx[iid]) for uid, iid, rating in teData]  # list of (user idx, item idx)
        idx = zip(*trData)

        # NMF model
        M = coo_matrix(([1]*len(idx[0]), (idx[0], idx[1])), shape=(row_id, col_id))  # binary label: buy(1) or not(0)
        model = NMF(n_components=10, init='random', random_state=0, alpha=self.alpha, l1_ratio=self.l1_ratio)
        X = model.fit_transform(M)
        Y = model.components_

        teData_user_to_item = {}  # map user idx to the item idx list in teData
        trData_user_to_item = {}  # map user idx to the item idx list in trData
        for user_to_item, data in ((teData_user_to_item, teData), (trData_user_to_item, trData)):
            for uid, iid in data:
                if uid in user_to_item:
                    user_to_item[uid].add(iid)
                else:
                    user_to_item[uid] = {iid}

        ranking_list = []
        auc_list = []
        for uid in teData_user_to_item:
            if uid in trData_user_to_item:
                pred_ratinglist = X[uid, :].dot(Y)  # predict the value
                pred_ratinglist = [(item_idx, rating) for item_idx, rating in enumerate(pred_ratinglist) if item_idx not in trData_user_to_item[uid]]  # associate column index with predicted value
                totEle = len(pred_ratinglist)
                pred_ratinglist = sorted(pred_ratinglist, key=lambda x: x[1], reverse=True)  # sort the value
                ranking = [ranking for ranking, (item_idx, rating) in enumerate(pred_ratinglist) if item_idx in teData_user_to_item[uid]]  # record the ranking value for each item in the test data
                auc_list.append(Recommendation.compute_auc(self, ranking, totEle))
                ranking_list.append(ranking)

        return ranking_list, auc_list
    def rec_and_eval(self):
        """recommendation with mf"""
        auc_fold = []
        for testID in range(self.fold):
            print('{0} fold is tested'.format(testID))
            ranking_list, auc_list =Recommendation.rec(self, [j for i, j in enumerate(self.dataList) if i != testID], self.dataList[testID])
            auc_fold.append(auc_list)
        for fold, i in enumerate(auc_fold):
            print('Mean value of auc in fold {0} is {1}'.format(fold, sum(i)/float(len(i))))

    def compute_auc(self, ranklist, totElem):
        """given ranking value of each item compute auc for a user"""
        postive = len(ranklist)
        negative = totElem - postive
        tp = 0
        roc = []
        for rank in ranklist:
            tp += 1
            fp = rank + 1 - tp
            tpr = float(tp)/(postive)
            fpr = float(fp)/(negative)
            roc.append((fpr, tpr))
        roc.append((1, 1))
        auc = 0
        i = 0
        while i < len(roc) - 1:
            auc += roc[i][1] * (roc[i+1][0] - roc[i][0])
            i += 1

        return auc

if __name__ == '__main__':
    obj = Recommendation()
    obj.parse('rs.csv')
    obj.split_data()
    obj.rec_and_eval()