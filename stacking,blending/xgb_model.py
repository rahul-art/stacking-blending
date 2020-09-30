import glob
import pandas as pd
import  numpy as np
from functools import partial
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
def run_training(pred_df, fold):
    #pdf = pred_df.copy(deep=True)
    train_df = pred_df[pred_df.kfold !=fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold ==fold].reset_index(drop=True)

    xtrain = train_df[["lr_pred", "lr_cnt_pred", "rf_svd_pred"]].values
    xvalid= valid_df[["lr_pred", "lr_cnt_pred", "rf_svd_pred"]].values

    xgb = XGBClassifier()

    xgb.fit(xtrain, train_df.sentiment.values)

    preds = xgb.predict_proba(xvalid)[:,1]
    auc = roc_auc_score(valid_df.sentiment.values,preds)
    print(f"{fold}, {auc}")
    valid_df.loc[:, "xgb_pred"] = preds
    return valid_df


if __name__=='__main__':
    files = glob.glob('../model_preds/*.csv')
    df = None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df=df.merge(temp_df,on='id', how='left')
    targets= df.sentiment.values
    pred_cols= ['lr_pred', "lr_cnt_pred", "rf_svd_pred"]
    dfs=[]
    for j in range(2):
       dfs.append(run_training(df,j))

    fin_vald_df = pd.concat(dfs)
    print(roc_auc_score(fin_vald_df.sentiment.values, fin_vald_df.xgb_pred.values))
