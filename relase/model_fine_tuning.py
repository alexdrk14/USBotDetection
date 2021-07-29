from sklearn.metrics import f1_score, roc_curve, roc_auc_score, auc, precision_recall_curve, accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import pandas as pd
import numpy as np
from collections import defaultdict
import argparse, sys
from datetime import datetime
from model_params import params
import random

#Data scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

#oversampling
from imblearn.combine import SMOTETomek
from sklearn.utils import shuffle


# scale  data of each portion of dataset (train, evaluation and test)
def scale_data(train, evaluation, test):
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
    # scale train and test
    X_eval = pd.DataFrame(scaler.transform(evaluation), columns=evaluation.columns)
    X_test = pd.DataFrame(scaler.transform(test), columns=test.columns)
    return X_train, X_eval, X_test


# oversample dataset portion
# used in separated form for train and evalution datasets
# do not use in test data at all
def oversample(data, target):
    smk = SMOTETomek()
    data, target = smk.fit_sample(data, target)
    return data, target


def read_data(filename, verbose=False, scale=False):
    # read the csv file that was combined with word2vec and graph features
    df = pd.read_csv(filename, header=0)

    # random shuffle the dataframe
    df = shuffle(df)

    # extract user id from dataframe
    user_ids = df["user_id"]

    # extract target from datafrmae
    target = df["target"]

    df = df.drop(["user_id", "target"], axis="columns")

    # make stratified train and test split 80/20
    X_train, X_test, y_train, y_test = train_test_split(df,
                                                        target,
                                                        test_size=0.2,
                                                        stratify=target)
    """
    # scale features values with MinMaxScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    # scale train and test
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # after scaling cast from np.array to dataframe again
    X_test = pd.DataFrame(X_test, columns=df.columns)
    X_train = pd.DataFrame(X_train, columns=df.columns)
    """
    return X_train, y_train, X_test, y_test




def get_xgboost_model(objective,learning_rate,n_estimators,max_depth,colsample_bytree,eval_metric, num_class=2,gpu=False):
    if gpu:
        return XGBClassifier(objective=objective,
                              num_class=num_class,
                              learning_rate=learning_rate,
                              n_estimators=n_estimators,
                              max_depth=max_depth,
                              colsample_bytree=colsample_bytree,
                              eval_metric=eval_metric,
                              tree_method="gpu_hist", #SERVER ONLY
                              predictor = 'gpu_predictor',#SERVER ONLY
                              use_label_encoder=False)
    else:
        return XGBClassifier(objective=objective,
                              num_class=num_class,
                              learning_rate=learning_rate,
                              n_estimators=n_estimators,
                              max_depth=max_depth,
                              colsample_bytree=colsample_bytree,
                              eval_metric=eval_metric,
                              use_label_encoder=False)

def get_svm_model(kernel, C):
    return SVC(kernel=kernel, C=C, probability=True)

def get_rfor_model(n_estimators,criterion,ccp_alpha,min_samples_split):
    return RandomForestClassifier(n_estimators=n_estimators,
                                criterion=criterion,ccp_alpha=ccp_alpha,
                                min_samples_split=min_samples_split)

def fine_tune_xgboost(gpu,X_train, X_val, Y_train, Y_val):
    global params
    #results of folds
    fold_results = defaultdict(lambda: {"auc_train": 0,
                                        "auc_val": 0,
                                        "f1_train": 0,
                                        "f1_val": 0})

    for objective in params["xgboost"]["objective"]:
        for learning_rate in params["xgboost"]["learning_rate"]:
            for n_estimators in params["xgboost"]["n_estimators"]:
                for max_depth in params["xgboost"]["max_depth"]:
                    for colsample_bytree in params["xgboost"]["colsample_bytree"]:
                        for eval_metric in params["xgboost"]["eval_metric"]:

                            #create model with selected parameters
                            model = get_xgboost_model(objective, learning_rate,
                                                      n_estimators, max_depth,
                                                      colsample_bytree, eval_metric,
                                                      num_class=2, gpu=gpu)

                            # measure model performance in: auc_train, auc_val, f1_train, f1_val
                            auc_train, auc_val, f1_train, f1_val = measure_model_performance(model,
                                                                                             X_train, X_val,
                                                                                             Y_train, Y_val)
                            model_param = "-XGBoost- Obj:{} LR:{} nEst:{} mDepth:{} cSample:{} eval:{}".format(
                                                                objective, learning_rate, n_estimators,
                                                                max_depth, colsample_bytree,eval_metric)
                            fold_results[model_param]["auc_train"] = auc_train
                            fold_results[model_param]["auc_val"] = auc_val
                            fold_results[model_param]["f1_train"] = f1_train
                            fold_results[model_param]["f1_val"] = f1_val
    return fold_results

def fine_tune_svm(X_train, X_val, Y_train, Y_val):
    global params
    fold_results = defaultdict(lambda: {"auc_train": 0, "auc_val": 0, "f1_train": 0, "f1_val": 0})
    for kernel in params["svm"]["kernel"]:
        for C in params["svm"]["C"]:

            model = get_svm_model(kernel=kernel, C=C)

            # measure model performance in: auc_train, auc_val, f1_train, f1_val
            auc_train, auc_val, f1_train, f1_val = measure_model_performance(model,
                                                                             X_train, X_val,
                                                                             Y_train, Y_val)
            model_param = "-SVM- kernel:{} C:{}".format(
                                                kernel, C)
            fold_results[model_param]["auc_train"] = auc_train
            fold_results[model_param]["auc_val"] = auc_val
            fold_results[model_param]["f1_train"] = f1_train
            fold_results[model_param]["f1_val"] = f1_val
    return fold_results

def fine_tune_rfor(X_train, X_val, Y_train, Y_val):
    global params
    fold_results = defaultdict(lambda: {"auc_train": 0, "auc_val": 0, "f1_train": 0, "f1_val": 0})

    for n_estimators in params["rfor"]["n_estimators"]:
        for criterion in params["rfor"]["criterion"]:
            for ccp_alpha in params["rfor"]["ccp_alpha"]:
                for min_samples_split in params["rfor"]["min_samples_split"]:

                    model = get_rfor_model(n_estimators=n_estimators,
                                           criterion=criterion,
                                           ccp_alpha=ccp_alpha,
                                           min_samples_split=min_samples_split)

                    #measure model performance in: auc_train, auc_val, f1_train, f1_val
                    auc_train, auc_val, f1_train, f1_val = measure_model_performance(model,
                                                                                     X_train, X_val,
                                                                                     Y_train, Y_val)

                    model_param = "-Rfor- nEst:{} criterion:{} cAplha:{} mSplit:{}".format(
                                                        n_estimators, criterion,
                                                        ccp_alpha, min_samples_split)
                    fold_results[model_param]["auc_train"] = auc_train
                    fold_results[model_param]["auc_val"] = auc_val
                    fold_results[model_param]["f1_train"] = f1_train
                    fold_results[model_param]["f1_val"] = f1_val
    return fold_results


def measure_model_performance(model, X_train, X_val, Y_train, Y_val):
    # fit the train data into the model
    fitted_opt = model.fit(X_train, Y_train.ravel())

    # get probabilities for each data portion
    train_probs = fitted_opt.predict_proba(X_train)
    val_probs = fitted_opt.predict_proba(X_val)

    # keep probabilities for the positive outcome only
    train_probs = train_probs[:, 1]
    val_probs = val_probs[:, 1]

    # ROC-AUC
    auc_train = roc_auc_score(Y_train, train_probs)
    auc_val = roc_auc_score(Y_val, val_probs)

    # F1 Score
    f1_train = f1_score(Y_train, model.predict(X_train))
    f1_val = f1_score(Y_val, model.predict(X_val))

    return auc_train, auc_val, f1_train, f1_val

def monte_carlo(models, iterations, filename, scale=False, gpu=False):
    #measure execution time starting from execution start
    start_time = datetime.now()

    #compute number of iterations = monte carlo runs * K folds
    number_of_comp = iterations * 5

    computation_ind = 0

    #datastucture that stores all the results
    model_results = defaultdict(lambda: defaultdict(lambda: []))

    for it in range(iterations):
        #read data, shuffle them and split into train/validation and testing
        df, target, df_test, y_test = read_data(filename, scale=False, verbose=True)

        #perform 5 Fold-Cross validation
        kf = StratifiedKFold(n_splits=5, shuffle=True)

        for train_index, test_index in kf.split(df, target):
            X_train, X_val = df.iloc[train_index], df.iloc[test_index]
            y_train, y_val = target.iloc[train_index], target.iloc[test_index]

            # Scale each data category (Train, Validaion and Test)by same scaller
            X_test = df_test
            if scale:
                X_train, X_val, X_test = scale_data(X_train, X_val, X_test)

            # Oversample the Train data portions
            X_train, y_train = oversample(X_train, y_train)

            for model in models:
                if model == "xgboost":
                    fold_results = fine_tune_xgboost(gpu, X_train, X_val, y_train, y_val)
                if model == "svm":
                    fold_results = fine_tune_svm(X_train, X_val, y_train, y_val)
                if model == "rfor":
                    fold_results = fine_tune_rfor(X_train, X_val, y_train, y_val)

                for model_params in fold_results:
                    for metric in ["auc_train", "auc_val", "f1_train", "f1_val"]:
                        model_results[model_params][metric].append(fold_results[model_params][metric])
                del(fold_results)
            computation_ind += 1
            spend_time = (datetime.now() - start_time).seconds / 360.0
            print("Time spend:{} h. left comp: {} left time:{} h.".format(spend_time, number_of_comp - computation_ind,
                                                                         (spend_time / computation_ind) * (number_of_comp - computation_ind)))

    for model_params in model_results:
        for metric in ["auc_train", "auc_val", "f1_train", "f1_val"]:
            model_results[model_params][metric] = sum(model_results[model_params][metric]) / float(len(model_results[model_params][metric]))
    return model_results

def store_results(models, results):
    f_out = open("model_{}_fine_tune_results.txt".format("-".join(models)), "w+")
    models_params = {"rfor": [], "svm": [], "xgboost": []}
    best_by_f1 = defaultdict(lambda: 0)
    best_by_AUC = defaultdict(lambda: 0)
    best_both = defaultdict(lambda: 0)
    for model_params in results:
        model = model_params.split(" ")[0]
        if model == "-Rfor-":
            models_params["rfor"].append(model_params)
        elif model == "-SVM-":
            models_params["svm"].append(model_params)
        elif model == "-XGBoost-":
            models_params["xgboost"].append(model_params)

    for model in models_params:
        f_out.write("############ Model:{} #############\n".format(model))

        for model_setup in models_params[model]:
            f_out.write("-----------------------------------\n")
            f_out.write("Params: {}\n".format(model_setup))
            f_out.write("Performance ROC-AUC-Train:{} ROC-AUC-Valid.:{} F1-Train:{} F1-Test:{}\n".format(
                results[model_setup]["auc_train"], results[model_setup]["auc_val"],
                results[model_setup]["f1_train"], results[model_setup]["f1_val"]))
            if results[model_setup]["auc_val"] > best_by_AUC[model+"_score"]:
                best_by_AUC[model] = model_setup
                best_by_AUC[model + "_score"] = results[model_setup]["auc_val"]
            if results[model_setup]["f1_val"] > best_by_f1[model + "_score"]:
                best_by_f1[model] = model_setup
                best_by_f1[model + "_score"] = results[model_setup]["f1_val"]
            if (results[model_setup]["auc_val"] + results[model_setup]["f1_val"]) / 2.0 > best_both[model + "_score"]:
                best_both[model] = model_setup
                best_both[model + "_score"] = (results[model_setup]["auc_val"] + results[model_setup]["f1_val"]) / 2.0
    f_out.write("-----------------------------------\n")
    f_out.write("BEST SETUP BY MODEL SCORES:\n")
    for model in models_params:
        f_out.write("-----------------------------------\n")
        f_out.write("Model:{}\n".format(model))
        f_out.write("By AUC:{} params:{}\n".format(best_by_AUC[model+"_score"], best_by_AUC[model]))
        f_out.write("By F1:{} params:{}\n".format(best_by_f1[model + "_score"], best_by_f1[model]))
        f_out.write("By both:{} params:{}\n".format(best_both[model + "_score"], best_both[model]))

    f_out.close()

parser = argparse.ArgumentParser(description='Model fine-tuning parameters')

parser.add_argument('--gpu', type=int,
                    dest="gpu", default=0,
                    help="Gpu flag , for usage of XGBoost performance 1 for utilization of gpu and 0 for not use the gpu.")
parser.add_argument('--model', dest='model', default="all",
                    help="Name of model used during fine_tuning. \nExample: --model xgboost\n          --model rfor\n          --model svm\n          --model all (for all 3 models fine tuning)")
parser.add_argument('--iter', type=int,
                    dest="iterations", default=10,
                    help="Number of Monte Carlo iterations")
parser.add_argument('--input',  dest="filename",
                    default="../data/september_old_labels.csv",
                    help="filename of file with df dataset")


args = parser.parse_args()
if __name__ == "__main__":
    models = []
    if args.model == "all":
        models = ["xgboost", "svm", "rfor"]
    elif args.model not in ["xgboost", "rfor", "svm"]:
        print("Model error:{}".format(args.model))
        sys.exit(-1)
    else:
        models.append(args.model)

    if args.iterations < 1:
        print("Wrong number of Monte Carlo iterations:{}".format(args.iterations))
        sys.exit(-1)
    if args.filename == "":
        print("Input filename is empty")
        sys.exit(-1)

    model_results = monte_carlo(models, args.iterations, args.filename, gpu=args.gpu)
    store_results(models, model_results)