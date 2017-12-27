import jieba.analyse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm


class SentimentClassification:
    def __init__(self, fold=5):
        self.fold = fold

    def parse(self, f):
        """parse"""
        print('parse')
        self.dataList = []
        for line in open(f):
            dataline = line.split()
            if dataline[0].isdigit():
                self.dataList.append((int(dataline[0]), ' '.join(dataline[1:])))

    def split_data(self):
        """split data into n fold"""
        print('split data')
        np.random.shuffle(self.dataList)
        l = len(self.dataList)/self.fold
        self.dataList = [self.dataList[i*l: (i+1)*l] for i in range(self.fold-1)] + [self.dataList[(self.fold-1)*l:]]  # each element in the list is splitted data list

    def model1(self):
        """SVM or other"""
        model = svm.SVC()
        model.fit()

    def feature_extraction(self, trData, teData):
        trData = [j for i in trData for j in i]
        trlabelList, trData = zip(*trData)
        telabelList, teData = zip(*teData)
        trainingData = [' '.join(jieba.analyse.extract_tags(sentence)) for sentence in trData]
        vec = TfidfVectorizer()
        tr_words_feature = vec.fit_transform(trainingData)
        te_words_feature = vec.transform(teData)
        model = svm.LinearSVC()
        model.fit(tr_words_feature, trlabelList)
        acc = model.score(te_words_feature, telabelList)
        print acc

    def eval(self):
        for testID in range(self.fold):
            SentimentClassification.feature_extraction(self, [j for i, j in enumerate(self.dataList) if i != testID], self.dataList[testID])

if __name__=='__main__':
    obj = SentimentClassification()
    obj.parse('Ch_trainfile_Sentiment_3000.txt')
    obj.split_data()
    obj.eval()
