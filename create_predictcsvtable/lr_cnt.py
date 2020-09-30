import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

def run_training(fold):
    df = pd.read_csv("../input/train_folds.csv")
    df.review=df.review.apply(str)
    df_train=df[df.kfold!=fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    cnt=CountVectorizer()
    cnt.fit(df_train.review.values)

    xtrain=cnt.transform(df_train.review.values)
    xvalid= cnt.transform(df_valid.review.values)

    ytrain=df_train.sentiment.values
    yvalid=df_valid.sentiment.values

    clf =LogisticRegression()
    clf.fit(xtrain,ytrain)
    pred=clf.predict_proba(xvalid)[:,1]

    auc = metrics.roc_auc_score(yvalid, pred)
    print(f'fold={fold}, auc={auc}')

    df_valid.loc[:, 'lr_cnt_pred'] = pred

    return df_valid[['id', 'sentiment', 'kfold', 'lr_cnt_pred']]

if __name__ == "__main__":
    dfs = []
    for j in range(2):
        temp_dfs=run_training(j)
        dfs.append(temp_dfs)
    fin_valid_df=pd.concat(dfs)
    print(fin_valid_df.shape)
    fin_valid_df.to_csv('../model_preds/lr_cnt.csv', index=False)