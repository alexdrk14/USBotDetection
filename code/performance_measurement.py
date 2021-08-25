from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, auc, precision_recall_curve, accuracy_score, average_precision_score
import numpy as np
import shap
from utils import *
from xgboost import XGBClassifier

time_features = ["daily_rt_tw_0","daily_rt_0","daily_tw_0","hour_rt_tw_0","hour_tw_0",
"hour_rt_0","retweet_time_avg","retweet_time_min","retweet_time_max",
"retweet_time_std","daily_retweet_avg","daily_tweet_avg"]
add_cnt = []
for feature in time_features:
    if "0" in feature:
        if "daily" in feature:
            for i in range(1,7):
                add_cnt.append(feature.replace("0",str(i)))
        elif "hour" in feature:
            for i in range(1,24):
                add_cnt.append(feature.replace("0",str(i)))
time_features += add_cnt

aaai_features = ["statuses_count","followers_count","friends_count","favourites_count","listed_count",
                 "default_profile", "background_img","verified","tweet_by_age", "followers_by_age",
                 "friends_by_age","favourites_by_age","listed_by_age","foll_friends", "screen_name_length",
                 "screen_name_digits","name_length","name_digits","description_length","screen_name_likelihood"]

graph_features = ["rt_self","in_degree","out_degree", "w_in_degree","w_out_degree","w_degree" ]

statistical_features = ["statuses_count","followers_count","friends_count",
                        "favourites_count","listed_count","name_length","geolocation",
                        "protected","location","background_img","default_profile",
                        "verified","screen_name_length","description_length","entities_count",
                        "name_and_screen_name_similarity","tweet_retweet","screen_name_digits",
                        "name_digits","tweet_by_age","followers_by_age","friends_by_age",
                        "favourites_by_age","listed_by_age","foll_friends","screen_name_likelihood"]


context_features = ["N1_tweet_mentioned_tfidf","N2_tweet_mentioned_tfidf","N3_tweet_mentioned_tfidf",
                    "N1_retweet_mentioned_tfidf","N2_retweet_mentioned_tfidf","N3_retweet_mentioned_tfidf",
                    "N1_tweet_hastag_tfidf","N2_tweet_hastag_tfidf","N3_tweet_hastag_tfidf",
                    "N1_retweet_hastag_tfidf","N2_retweet_hastag_tfidf","N3_retweet_hastag_tfidf",
                    "tweet_number_of_urls_avg","tweet_number_of_urls_std","retweet_number_of_urls_avg",
                    "retweet_number_of_urls_std","tweet_number_of_hashtags_avg","tweet_number_of_hashtags_std",
                    "tweet_number_of_mentions_avg", "tweet_number_of_mentions_std","retweet_number_of_hashtags_avg",
                    "retweet_number_of_hashtags_std","retweet_number_of_mentions_avg","retweet_number_of_mentions_std",
                    "N1_tweet_mentioned_word_0","N2_tweet_mentioned_word_0","N3_tweet_mentioned_word_0",
                    "N1_retweet_mentioned_word_0","N2_retweet_mentioned_word_0","N3_retweet_mentioned_word_0",
                    "N1_tweet_hastag_word_0","N2_tweet_hastag_word_0","N3_tweet_hastag_word_0",
                    "N1_retweet_hastag_word_0","N2_retweet_hastag_word_0","N3_retweet_hastag_word_0",
                    "N1_tweet_word_0","N2_tweet_word_0","N3_tweet_word_0",
                    "N1_retweet_word_0","N2_retweet_word_0","N3_retweet_word_0"]
add_cnt = []
for feature in context_features:
    if "0" in feature:
        for i in range(1,10):
            add_cnt.append(feature.replace("0",str(i)))
context_features += add_cnt


our_features = ast.literal_eval(open("selected_features.txt", "r").read())


def measure():
    filename = "../data/us_2020_election_data.csv"
    feature_set = [aaai_features, statistical_features, context_features, time_features, graph_features, our_features]
    AUCEs = [None] * len(feature_set)
    F1Es = [None] * len(feature_set)
    precision = []
    recall = []
    FP = []
    TP = []
    ROC = []
    model_params = parse_model_params("xgboost")

    iterations = 10
    data_res = defaultdict(lambda: defaultdict(lambda: []))
    for it in range(iterations):
        df, target, X_test, Y_test = read_data(filename, known_features=False)

        X_train, X_val, Y_train, Y_val = train_test_split(df, target, test_size=0.30, stratify=target)


        ####################################################
        # Oversample train and validation portion separately#
        ####################################################

        # oversampling clean on training set
        X_train, Y_train = oversample(X_train, Y_train)



        for run in range(len(feature_set)):
            ########################################################################
            # Train model based on train portion of data with selected features only#
            ########################################################################

            selected_model = XGBClassifier(objective=model_params["objective"],
                                           num_class=2,
                                           learning_rate=model_params["learning_rate"],
                                           n_estimators=model_params["n_estimators"],
                                           max_depth=model_params["max_depth"],
                                           colsample_bytree=model_params["colsample_bytree"],
                                           eval_metric=model_params["eval_metric"],
                                           use_label_encoder=False)

            XGB_fitted_opt = selected_model.fit(X_train[feature_set[run]], Y_train)

            # XGB_fitted_opt = XGB_model.fit(XGB_X_train, y_train.ravel())

            # XGB_train_probs = XGB_fitted_opt.predict_proba(X_train[feature_set[run]])
            # XGB_val_probs = XGB_fitted_opt.predict_proba(X_val[feature_set[run]])
            XGB_test_probs = XGB_fitted_opt.predict_proba(X_test[feature_set[run]])

            # keep probabilities for the positive outcome only
            # XGB_train_probs = XGB_train_probs[:, 1]
            # XGB_val_probs = XGB_val_probs[:, 1]
            data_res[str(run)]["lr_probs"].append(XGB_test_probs)

            XGB_test_probs = XGB_test_probs[:, 1]
            data_res[str(run)]["RF_fitted_opt"].append(selected_model)

            data_res[str(run)]["X_test"].append(X_test[feature_set[run]])
            data_res[str(run)]["y_test"].append(Y_test)

            # XGB_precision_train, XGB_recall_train, _ = precision_recall_curve(y_train, XGB_train_probs)
            # XGB_precision, XGB_recall, _ = precision_recall_curve(y_val, XGB_val_probs)
            XGB_precision, XGB_recall, _ = precision_recall_curve(Y_test, XGB_test_probs)
            # precision.append(XGB_precision)
            # recall.append(XGB_recall)
            # XGB_precision_test, XGB_recall_test, _ = precision_recall_curve(target_test, XGB_test_probs)

            # AUC
            # XGB_auc_train = auc(XGB_recall_train, XGB_precision_train)
            XGB_auc = auc(XGB_recall, XGB_precision)
            # XGB_auc_test = auc(XGB_recall_test, XGB_precision_test)

            # F1
            XGB_f1 = f1_score(Y_test, selected_model.predict(X_test[feature_set[run]]))

            fpr, tpr, threshold = roc_curve(Y_test, XGB_test_probs)

            roc_auc = auc(fpr, tpr)

            ###########################
            # October data measurment #
            ###########################
            october_df = pd.read_csv("../data/us_2020_election_data_october.csv", header=0)

            october_target = october_df["target"]


            october_f1 = f1_score(october_target, selected_model.predict(october_df[feature_set[run]]))
            october_probs = XGB_fitted_opt.predict_proba(october_df[feature_set[run]])[:, 1]
            october_fpr, october_tpr, october_threshold = roc_curve(october_target, october_probs)

            october_roc_auc = auc(october_fpr, october_tpr)

            data_res[str(run)]["PREC"].append(XGB_precision)
            data_res[str(run)]["REC"].append(XGB_recall)

            data_res[str(run)]["FP"].append(fpr)
            data_res[str(run)]["TP"].append(tpr)
            data_res[str(run)]["ROC"].append(roc_auc)
            data_res[str(run)]["AUC"].append(XGB_auc)
            data_res[str(run)]["F1"].append(XGB_f1)
            data_res[str(run)]["F1_october"].append(october_f1)
            data_res[str(run)]["ROC_october"].append(october_roc_auc)
            

            FP.append(fpr)
            TP.append(tpr)
            ROC.append(roc_auc)
            # AUCEs[run] = XGB_auc
            # F1Es[run] = XGB_f1
            precision.append(XGB_precision)
            recall.append(XGB_recall)
    return data_res


def roc_curves(data_res):
    plt.rcParams.update({'font.size': 11})
    fig = plt.figure(figsize=(6, 6), dpi=600)
    model_labels = ["Statistical General only", "Statistical", "Context ", "Time", "Graph", "Our Model"]

    # plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    for c_type in range(0, 6):
        tprs = []
        aucs = []

        mean_fpr = np.linspace(0, 1, 100)

        PR_prs = []
        PR_aucs = []
        PR_mean_recall = np.linspace(0, 1, 100)

        y_real = []
        precision_array = []
        y_proba = []
        precision_array_SMOTE = []
        y_proba_SMOTE = []

        for MonteCarlo_i in range(0, 10):
            # ------------------ Load parameters needed for plotting purposes ----------
            fpr = data_res[str(c_type)]["FP"][MonteCarlo_i]
            tpr = data_res[str(c_type)]["TP"][MonteCarlo_i]

            # fpr = new_dict["0"]['FPR_MC', MonteCarlo_i]
            # tpr = new_dict[len(new_dict)-1]['TPR_MC', MonteCarlo_i]
            roc_auc = auc(fpr, tpr)
            # predict probabilities
            lr_probs = data_res[str(c_type)]["lr_probs"][MonteCarlo_i]

            # keep probabilities for the positive outcome only
            lr_probs = lr_probs[:, 1]


            # -------------- Plot Precision-Recall curves ------------------------------
            RF_fitted_opt = data_res[str(c_type)]["RF_fitted_opt"][MonteCarlo_i]
            X_test = data_res[str(c_type)]["X_test"][MonteCarlo_i]
            y_test = data_res[str(c_type)]["y_test"][MonteCarlo_i]



            # predict class values
            yhat = RF_fitted_opt.predict(X_test)
            lr_precision, lr_recall, _ = precision_recall_curve(y_test.values, lr_probs)
            lr_f1, lr_auc = f1_score(y_test.values, yhat), auc(lr_recall, lr_precision)
            averagePrecision = average_precision_score(y_test.values, lr_probs)



            # ------------------------ average values ----------------
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            # Compute PR curve and area the curve------------------
            lr_precision, lr_recall, _ = precision_recall_curve(y_test.values, lr_probs)
            PR_prs.append(np.interp(PR_mean_recall, lr_precision, lr_recall))
            pr_auc = auc(lr_recall, lr_precision)
            PR_aucs.append(pr_auc)

            # --- used to compute below, the mean PR curves------------------
            y_real.append(y_test.values)

            y_proba.append(lr_probs)
            lr_precision, lr_recall = lr_precision[::-1], lr_recall[::-1]  # reverse order of results
            precision_array = np.interp(PR_mean_recall, lr_recall, lr_precision)

        # ----------------------- mean ROC curves --------------------------------------
        aucs1 = aucs

        std_auc1 = np.std(aucs1)

        tprs1 = tprs
        mean_tpr1 = np.mean(tprs1, axis=0)

        mean_fpr1 = mean_fpr
        mean_auc1 = auc(mean_fpr1, mean_tpr1)

        std_tpr1 = np.std(tprs1, axis=0)
        tprs_upper1 = np.minimum(mean_tpr1 + std_tpr1, 1)
        tprs_lower1 = np.maximum(mean_tpr1 - std_tpr1, 0)

        collors = [4, 0, 1, 5, 3, 2]
        col = collors[c_type]

        plt.plot(mean_fpr1, mean_tpr1, color='C{}'.format(col),
                 label=r'{} (AUC: {:.2f} $\pm$ {:.3f}'.format(model_labels[c_type], mean_auc1, std_auc1),
                 lw=2, alpha=.8)

        plt.fill_between(mean_fpr1, tprs_lower1, tprs_upper1, color='C{}'.format(col), alpha=.1)
        # label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive rate', fontsize=11)  # ('1-Specificity', fontsize=13)
    plt.ylabel('True Positive rate ', fontsize=11)  # ('Sensitivity', fontsize=13)
    plt.title('ROC curves ', fontsize=13)
    plt.legend(loc='best')

    plt.savefig("../plots/roc_mean.png", bbox_inches='tight', dpi=600, facecolor='w')

def pr_curves(data_res):
    plt.rcParams.update({'font.size': 11})
    fig = plt.figure(figsize=(6, 6), dpi=600)
    model_labels = ["Statistical General only", "Statistical", "Context ", "Time", "Graph", "Our Model"]

    for c_type in range(0, 6):
        tprs = []
        aucs = []
        f1 = []
        mean_fpr = np.linspace(0, 1, 100)

        PR_prs = []
        PR_aucs = []
        PR_mean_recall = np.linspace(0, 1, 100)

        y_real = []
        precision_array = []
        y_proba = []
        precision_array_SMOTE = []
        y_proba_SMOTE = []

        for MonteCarlo_i in range(0, 10):
            # ------------------ Load parameters needed for plotting purposes ----------
            fpr = data_res[str(c_type)]["FP"][MonteCarlo_i]
            tpr = data_res[str(c_type)]["TP"][MonteCarlo_i]
            f1.append(data_res[str(c_type)]["F1"][MonteCarlo_i])

            roc_auc = auc(fpr, tpr)
            # predict probabilities
            lr_probs = data_res[str(c_type)]["lr_probs"][MonteCarlo_i]

            # keep probabilities for the positive outcome only
            lr_probs = lr_probs[:, 1]

            # rf_opt_pred_prob = new_dict[len(new_dict)-1]['RF_opt_PredictedProbabilities', MonteCarlo_i]
            # Test_CM = new_dict[len(new_dict)-1]['TEST_Conf_matrix', MonteCarlo_i]

            # -------------- Plot Precision-Recall curves ------------------------------
            RF_fitted_opt = data_res[str(c_type)]["RF_fitted_opt"][MonteCarlo_i]
            X_test = data_res[str(c_type)]["X_test"][MonteCarlo_i]
            y_test = data_res[str(c_type)]["y_test"][MonteCarlo_i]


            # predict class values
            yhat = RF_fitted_opt.predict(X_test)
            lr_precision, lr_recall, _ = precision_recall_curve(y_test.values, lr_probs)
            lr_f1, lr_auc = f1_score(y_test.values, yhat), auc(lr_recall, lr_precision)
            averagePrecision = average_precision_score(y_test.values, lr_probs)

            # ------------------------ average values ----------------
            tprs.append(np.interp(mean_fpr, lr_precision, lr_recall))
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            # Compute PR curve and area the curve------------------
            lr_precision, lr_recall, _ = precision_recall_curve(y_test.values, lr_probs)
            PR_prs.append(np.interp(PR_mean_recall, lr_precision, lr_recall))
            pr_auc = auc(lr_recall, lr_precision)
            PR_aucs.append(pr_auc)

            # --- used to compute below, the mean PR curves------------------
            y_real.append(y_test.values)

            y_proba.append(lr_probs)
            lr_precision, lr_recall = lr_precision[::-1], lr_recall[::-1]  # reverse order of results
            precision_array = np.interp(PR_mean_recall, lr_recall, lr_precision)

        # ----------------------- mean ROC curves --------------------------------------
        aucs1 = aucs

        std_auc1 = np.std(aucs1)

        tprs1 = tprs
        mean_tpr1 = np.mean(tprs1, axis=0)

        mean_fpr1 = mean_fpr
        mean_auc1 = auc(mean_fpr1, mean_tpr1)

        std_tpr1 = np.std(tprs1, axis=0)
        tprs_upper1 = np.minimum(mean_tpr1 + std_tpr1, 1)
        tprs_lower1 = np.maximum(mean_tpr1 - std_tpr1, 0)

        collors = [4, 0, 1, 5, 3, 2]
        col = collors[c_type]

        plt.plot(mean_fpr1, mean_tpr1, color='C{}'.format(col),
                 label=r'{} (AUC: {:.2f} $\pm$ {:.3f}'.format(model_labels[c_type], mean_auc1, std_auc1),
                 lw=2, alpha=.8)

        plt.fill_between(mean_fpr1, tprs_lower1, tprs_upper1, color='C{}'.format(col), alpha=.1)
        # label=r'$\pm$ 1 std. dev.')

        ####
        # plt.plot(mean_fpr2, mean_tpr2, color='C{}'.format(c_type+1), linestyle='--',
        #        label=r'Mean ROC, $M_2$ (AUC = %0.2f $\pm$ %0.3f)' % (mean_auc2, std_auc2),
        #        lw=2, alpha=.8)

        # plt.fill_between(mean_fpr2, tprs_lower2, tprs_upper2, color='C{}'.format(c_type+1), alpha=.1)
        #                # label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Precision', fontsize=11)  # ('1-Specificity', fontsize=13)
    plt.ylabel('Recall', fontsize=11)  # ('Sensitivity', fontsize=13)
    plt.title('Precision vs Recall curves ', fontsize=13)
    plt.legend(loc='best')

    plt.savefig("../plots/prec_recal_mean.png", bbox_inches='tight', dpi=600, facecolor='w')

def shap_values():
    filename = "../data/us_2020_election_data.csv"
    model_params = parse_model_params("xgboost")

    df, target, X_test, Y_test = read_data(filename, known_features=True)

    X_train, X_val, Y_train, Y_val = train_test_split(df, target, test_size=0.30, stratify=target)

    ####################################################
    # Oversample train and validation portion separately#
    ####################################################

    # oversampling clean on training set
    X_train, Y_train = oversample(X_train, Y_train)

    # Flush both models, in order to use the from scratch
    XGB_model = XGBClassifier(objective=model_params["objective"],
                               num_class=2,
                               learning_rate=model_params["learning_rate"],
                               n_estimators=model_params["n_estimators"],
                               max_depth=model_params["max_depth"],
                               colsample_bytree=model_params["colsample_bytree"],
                               eval_metric=model_params["eval_metric"],
                               use_label_encoder=False)


    XGB_model.fit(X_train, Y_train.ravel())

    shap_values_xgb = shap.TreeExplainer(XGB_model).shap_values(X_test)

    fig = plt.figure()
    shap.summary_plot(shap_values_xgb[1], X_test)
    fig.savefig("../plots/shap.png", bbox_inches='tight', dpi=600, facecolor='w')

if __name__ == "__main__":
    res = measure()
    pr_curves(res)
    roc_curves(res)
    model_labels = ["Statistical General only", "Statistical", "Context ", "Time", "Graph", "Our Model"]
    for i in res:
        print("{} F1:{}".format(model_labels[int(i)], sum(res[i]["F1"]) / len(res[i]["F1"])))
    print("Our model features performance of october data: F1:{} ROC-AUC:{}".format(
        sum(res["5"]["F1_october"]) / len(res["5"]["F1_october"]),
        sum(res["5"]["ROC_october"]) / len(res["5"]["ROC_october"])))
    shap_values()