import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
import json
import yaml
import os
import scipy
import time
import matplotlib.pyplot as plt
import json
import matplotlib.pyplot as plt
import importlib.util
import yaml
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from random import shuffle
import pickle
import cv2
import copy
from sklearn.metrics import r2_score
import kwcoco
from sklearn.metrics import average_precision_score
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from matplotlib.backends.backend_pdf import PdfPages

# angel imports
from angel_system.impls.detect_activities.detections_to_activities.utils import (
    obj_det2d_set_to_feature,
)
from angel_system.activity_hmm.eval import save_matrix_image, viz_step_finished_conf


pred_fnames = [
    "/angel_workspace/ros_bags/m2_all_data_cleaned_fixed_with_steps/m2_all_data_cleaned_fixed_with_steps_results_test.mscoco.json",
    "/angel_workspace/ros_bags/m2_all_data_cleaned_fixed_with_steps/m2_all_data_cleaned_fixed_with_steps_results_train_activity.mscoco.json",
    "/angel_workspace/ros_bags/m2_all_data_cleaned_fixed_with_steps/m2_all_data_cleaned_fixed_with_steps_results_val.mscoco.json",
]
act_label_yaml = "/angel_workspace/config/activity_labels/medical_tourniquet.v2.yaml"

with open(act_label_yaml, "r") as stream:
    act_labels = yaml.safe_load(stream)

if False:
    with open(trimmed_recipe_yaml, "r") as stream:
        trimmed_recipe = yaml.safe_load(stream)

    trim_act_map = {}
    for step in trimmed_recipe["steps"]:
        ids = step["activity_id"]
        try:
            ids = ids.split(",")
            for i in ids:
                trim_act_map[int(i)] = int(step["id"])
        except:
            trim_act_map[int(ids)] = int(step["id"])


def sanitize_str(txt):
    txt = txt.lower()

    # remove (step 1)
    try:
        txt = txt[: txt.index("(") - 1]
    except ValueError:
        pass

    if txt[-1] == ".":
        txt = txt[:-1]

    return txt


# Description to ID map.
act_map = {}
inv_act_map = {}
for step in act_labels["labels"]:
    act_map[sanitize_str(step["label"])] = step["id"]
    inv_act_map[step["id"]] = step["label"]

if 0 not in act_map.values():
    act_map["background"] = 0
    inv_act_map[0] = "Background"

act_str_list = [inv_act_map[key] for key in range(max(inv_act_map.keys()) + 1)]

dat = None
for pred_fname in pred_fnames:
    dat_ = kwcoco.CocoDataset(pred_fname)

    if dat is not None:
        dat = dat.union(dat_)
    else:
        dat = dat_

if False:
    activities = set()
    for img_id in dat.imgs:
        im = dat.imgs[img_id]
        try:
            activities.add(im["activity_gt"])
        except KeyError:
            pass


# dat.category_annotation_frequency()


image_activity_gt = {}
image_id_to_dataset = {}
for img_id in dat.imgs:
    im = dat.imgs[img_id]
    image_id_to_dataset[im["id"]] = os.path.split(im["file_name"])[0]

    if im["activity_gt"] is None:
        continue

    image_activity_gt[im["id"]] = act_map[sanitize_str(im["activity_gt"])]

dsets = sorted(list(set(image_id_to_dataset.values())))
image_id_to_dataset = {
    i: dsets.index(image_id_to_dataset[i]) for i in image_id_to_dataset
}


set(image_activity_gt.values())


min_cat = min([dat.cats[i]["id"] for i in dat.cats])
num_act = len(dat.cats)
label_to_ind = {dat.cats[i]["name"]: dat.cats[i]["id"] - min_cat for i in dat.cats}
act_id_to_str = {dat.cats[i]["id"]: dat.cats[i]["name"] for i in dat.cats}


ann_by_image = {}
for i in dat.anns:
    ann = dat.anns[i]
    if ann["image_id"] not in ann_by_image:
        ann_by_image[ann["image_id"]] = [ann]
    else:
        ann_by_image[ann["image_id"]].append(ann)


X = []
y = []
dataset_id = []
last_dset = 0
for image_id in sorted(list(ann_by_image.keys())):
    label_vec = []
    left = []
    right = []
    top = []
    bottom = []
    label_confidences = []
    obj_obj_contact_state = []
    obj_obj_contact_conf = []
    obj_hand_contact_state = []
    obj_hand_contact_conf = []

    for ann in ann_by_image[image_id]:
        label_vec.append(act_id_to_str[ann["category_id"]])
        left.append(ann["bbox"][0])
        right.append(ann["bbox"][1])
        top.append(ann["bbox"][2])
        bottom.append(ann["bbox"][3])
        label_confidences.append(ann["confidence"])

        try:
            obj_obj_contact_state.append(ann["obj-obj_contact_state"])
            obj_obj_contact_conf.append(ann["obj-obj_contact_conf"])
            obj_hand_contact_state.append(ann["obj-hand_contact_state"])
            obj_hand_contact_conf.append(ann["obj-hand_contact_conf"])
        except KeyError:
            pass

    feature_vec = obj_det2d_set_to_feature(
        label_vec,
        left,
        right,
        top,
        bottom,
        label_confidences,
        None,
        obj_obj_contact_state,
        obj_obj_contact_conf,
        obj_hand_contact_state,
        obj_hand_contact_conf,
        label_to_ind,
        version=1,
    )

    X.append(feature_vec.ravel())
    try:
        dataset_id.append(image_id_to_dataset[image_id])
        last_dset = dataset_id[-1]
    except:
        dataset_id.append(last_dset)

    try:
        # y.append(trim_act_map[image_activity_gt[image_id]])
        y.append(image_activity_gt[image_id])
    except:
        y.append(0)

X = np.array(X)
y = np.array(y)
dataset_id = np.array(dataset_id)

if False:
    ind = np.argsort(dataset_id)
    X = X[ind]
    y = y[ind]
    dataset_id = dataset_id[ind]

if True:
    # Carve out final test set.
    val_fract = 0.2
    final_test_dataset_ids = sorted(list(set(dataset_id)))
    i = final_test_dataset_ids[int(np.round((len(final_test_dataset_ids) * val_fract)))]
    ind = dataset_id <= i
    X_final_test = X[ind]
    y_final_test = y[ind]
    dataset_id_final_test = dataset_id[ind]
    X = X[~ind]
    y = y[~ind]
    dataset_id = dataset_id[~ind]
else:
    X_final_test = X
    y_final_test = y
    dataset_id_final_test = dataset_id

if False:
    # Get rid of Not Started and Finish and just call them background.
    ind = y == 1
    y[ind] = 0
    ind = y == 17
    y[ind] = 0

if True:
    ref = set(range(len(act_map)))
    # Make sure every dataset has at least one example of each step.
    for i in sorted(list(set(dataset_id))):
        # ind = dataset_id == i
        # for j in ref.difference(set(y[ind])):
        for j in ref:
            y = np.hstack([y, j])
            X = np.vstack([X, np.zeros(X[0].shape)])
            dataset_id = np.hstack([dataset_id, i])

if False:
    # Check which examples show up in which datasets.
    coverage = []
    for dataset_id_ in sorted(list(set(dataset_id))):
        ind = dataset_id == dataset_id_
        coverage.append([class_id in set(y[ind]) for class_id in range(max(y) + 1)])

    coverage = np.array(coverage, dtype=float)
    fname = "/angel_workspace/ros_bags/dataset_class_coverage.png"
    save_matrix_image(
        coverage, fname, max_w=1000, aspect_ratio=1, first_ind=0, col_labels=True
    )

if False:
    ind = list(range(len(y)))
    shuffle(ind)
    X = X[ind]
    y = y[ind]
    dataset_id = dataset_id[ind]


if False:
    # Give credit to interstitial to prior step or background.
    interstitial_ids = [1, 3, 5, 7, 9, 11, 13, 15]
    X2 = X.copy()
    y2 = y.copy()
    dataset_id2 = dataset_id.copy()
    for interstitial_id in interstitial_ids:
        ind = y == interstitial_id
        X2 = np.vstack([X2, X[ind]])
        y2 = np.hstack([y2, y[ind] - 1])
        dataset_id2 = np.hstack([dataset_id2, dataset_id[ind]])
        # X2 = np.vstack([X2, X[ind]])
        # y2 = np.hstack([y2, y[ind] + 1])
        # dataset_id2 = np.hstack([dataset_id2, dataset_id[ind]])

        if interstitial_id > 0:
            # Give an example where it is considered background.
            X2 = np.vstack([X2, X[ind]])
            y2 = np.hstack([y2, y[ind] * 0])
            dataset_id2 = np.hstack([dataset_id2, dataset_id[ind]])

    X0 = X.copy()
    y0 = y.copy()
    dataset_id0 = dataset_id.copy()
    X = X2
    y = y2
    dataset_id = dataset_id2


plt.imshow(np.cov(X.T))
plt.colorbar()

y_ = y.tolist()
x_ = list(range(max(y) + 1))
counts = [y_.count(i) for i in x_]
plt.close("all")
fig = plt.figure(num=None, figsize=(14, 10), dpi=80)
plt.rc("font", **{"size": 22})
plt.rc("axes", linewidth=4)
plt.bar(x_, counts)
plt.xticks(x_)
plt.ylabel("Counts", fontsize=34)
plt.xlabel("Ground Truth Steps", fontsize=34)
plt.tight_layout()
plt.savefig("/angel_workspace/ros_bags/gt_counts.png")


def fit(clf, X, y, dataset_id, n_splits=2, n_components=50):
    group_kfold = GroupKFold(n_splits=n_splits)
    y_test = []
    ypred_test = []
    score = []
    for i, (train_index, test_index) in enumerate(group_kfold.split(X, y, dataset_id)):
        # print(f"Fold {i}:")
        # print(f"  Train: index={train_index}, group={dataset_id[train_index]}")
        # print(f"  Test:  index={test_index}, group={dataset_id[test_index]}")
        X_ = X[train_index]
        y_ = y[train_index]

        if isinstance(clf, RandomForestClassifier):
            pipe = clf
        elif True:
            # X_ = PCA(whiten=True, n_components='mle').fit_transform(X_)
            # pipe = make_pipeline(PCA(whiten=True, n_components='mle'), clf)
            pipe = make_pipeline(PCA(whiten=True, n_components=n_components), clf)
        else:
            pipe = make_pipeline(RobustScaler(), clf)

        if isinstance(clf, MLPClassifier):
            clf.solver = "adam"
            clf.warm_start = False
            clf.max_iter = 200
            pipe.fit(X_, y_)
            clf.solver = "lbfgs"
            clf.warm_start = True
            clf.max_iter = 10000
            pipe.fit(X_, y_)
        elif False:
            pipe.fit(X_, y_)

        print(set(y[train_index]))
        if isinstance(clf, RandomForestClassifier):
            pipe.fit(X[train_index], y[train_index])
        if False:
            score.append(accuracy_score(y[test_index], pipe.predict(X[test_index])))
        else:
            n = min([n_splits, len(set(dataset_id[train_index]))])
            cv = GroupKFold(n_splits=n).split(
                X[train_index], y[train_index], dataset_id[train_index]
            )
            pipe = CalibratedClassifierCV(pipe, cv=cv)
            pipe.fit(X[train_index], y[train_index])

        y_score = pipe.predict_proba(X[test_index])
        lb = preprocessing.LabelBinarizer()
        y_true = lb.fit(range(y_score.shape[1])).transform(y[test_index])
        score.append(average_precision_score(y_true, y_score))

        print(score[-1], y_score.shape)

        y_test.append(y[test_index])
        # ypred_test.append(pipe.predict(X[test_index]))
        ypred_test.append(y_score)

    y_test = np.hstack(y_test)
    ypred_test = np.vstack(ypred_test)

    # s = acc_score
    # s = accuracy_score(y_test, ypred_test)
    return np.mean(score), y_test, ypred_test


best_models = {}


def check_score(clf, X, y, n_splits=5, n_components=50):
    score_ = fit(clf, X, y, dataset_id, n_splits, n_components)[0]
    best_models[score_] = copy.deepcopy(clf)
    return score_


# Logistic regression.
clf = LogisticRegression(
    random_state=0,
    max_iter=1000,
    penalty="elasticnet",
    fit_intercept=False,
    solver="saga",
)
Cs = np.logspace(-6, 3, 20)
l1_ratios = np.linspace(0, 1, 10)
l1_ratios = [0.1]
score_image = np.zeros((len(l1_ratios), len(Cs)))
for i, l1_ratio in enumerate(l1_ratios):
    for j, C in enumerate(Cs):
        clf.C = C
        clf.l1_ratio = l1_ratio
        s = check_score(clf, X, y)
        score_image[i, j] = s
        print("Score", clf.C, clf.l1_ratio, s)

clf = best_models[max(best_models.keys())]
x = sorted(best_models.keys())
plt.plot([best_models[x_].C for x_ in x])
clf = LinearSVC(random_state=0, tol=1e-5)

Cs = np.logspace(-3, 3, 20)
for j, C in enumerate(Cs):
    clf.C = C
    s = check_score(clf, X, y)
    print("Score", clf.C, s)

x = sorted(best_models.keys())
plt.plot([best_models[x_].C for x_ in x], x, "o")


clf = RandomForestClassifier(
    max_depth=7, random_state=0, n_estimators=100, max_features=0.2, bootstrap=True
)
n = np.arange(1, 11)
for j, n_ in enumerate(n):
    clf.max_depth = n_
    s = check_score(clf, X, y)
    print("Score", n_, s)

x = sorted(best_models.keys())
plt.plot([best_models[x_].max_depth for x_ in x], x, "o")


clf = best_models[max(best_models.keys())]

plt.semilogx(C[: len(err)], err, "ro")


y_test, ypred_test = fit(clf, X, y, dataset_id, n_splits=5, n_components=50)[1:]


np.any(clf.coef_ != 0, axis=0)


scores = []
for n_components in range(5, 50):
    score_ = fit(clf, X, y, dataset_id, n_components=n_components)[0]
    print(n_components, score_)
    scores.append(score_)

plt.plot(range(5, 50), scores)

# ----------------------------------------------------------------------------
def fit_regr(clf, X, y, dataset_id, n_splits=2, n_components=50):
    group_kfold = GroupKFold(n_splits=n_splits)
    y_test = []
    ypred_test = []
    for i, (train_index, test_index) in enumerate(group_kfold.split(X, y, dataset_id)):
        # print(f"Fold {i}:")
        # print(f"  Train: index={train_index}, group={dataset_id[train_index]}")
        # print(f"  Test:  index={test_index}, group={dataset_id[test_index]}")
        X_ = X[train_index]
        y_ = y[train_index]

        if True:
            # X_ = PCA(whiten=True, n_components='mle').fit_transform(X_)
            # pipe = make_pipeline(PCA(whiten=True, n_components='mle'), clf)
            pipe = make_pipeline(PCA(whiten=True, n_components=n_components), clf)
        else:
            pipe = make_pipeline(RobustScaler(), clf)

        pipe.fit(X_, y_)

        print(r2_score(y[test_index], pipe.predict(X[test_index])))

        y_test.append(y[test_index])
        ypred_test.append(pipe.predict(X[test_index]))

    y_test = np.hstack(y_test)
    ypred_test = np.hstack(ypred_test)
    return r2_score(y_test, ypred_test), y_test, ypred_test


from sklearn.linear_model import ElasticNet

regr = ElasticNet(random_state=0)
alphas = np.logspace(-1, 0, 20)
scores = []
for alpha in alphas:
    regr.alpha = alpha
    scores.append(fit_regr(regr, X, y, dataset_id)[0])
    print(alpha, scores[-1])

regr.alpha = alphas[np.argmax(scores)]
plt.plot(alphas, scores)

y_test, ypred_test = fit_regr(regr, X, y, dataset_id, n_splits=5, n_components=50)[1:]
ypred_test = np.round(ypred_test)
ypred_test[ypred_test < 0] = 0
ypred_test[ypred_test > max(y_test)] = max(y_test)

# ----------------------------------------------------------------------------
# GaussianNB
best_models = {}


def check_score(clf, X, y):
    score_ = fit(clf, X, y, dataset_id)[0]
    best_models[score_] = copy.deepcopy(clf)
    print("Score", score_)
    return score_


from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

kernel = 1.0 * RBF(0.1)
clf = GaussianProcessClassifier(kernel=kernel, random_state=0)
check_score(clf)

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
check_score(clf)

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

clf = QuadraticDiscriminantAnalysis(reg_param=1)
check_score(clf)

clf = RandomForestClassifier(
    max_depth=None, random_state=0, n_estimators=1000, bootstrap=True
)
check_score(clf)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)
check_score(clf)

from sklearn.ensemble import AdaBoostClassifier

clf = LogisticRegression(random_state=0, max_iter=1000)
clf = AdaBoostClassifier(clf, n_estimators=100, random_state=0)
check_score(clf)

from sklearn.naive_bayes import ComplementNB

clf = ComplementNB(force_alpha=True)
check_score(clf)


clf = SVC(gamma=0.025)
check_score(clf)
clf = LinearSVC(random_state=0, tol=1e-5)
check_score(clf)

C = np.logspace(-2, 2, 20)
score = []
for C_ in C:
    clf.C = C_
    score_ = check_score(clf)
    score.append(score_)
    print(C_, score[-1])

plt.semilogx(C[: len(score)], score, "ro")
# plt.semilogx(C[:len(score)], score, 'ro', label='gamma=%s' % clf.gamma)
plt.legend()
# ----------------------------------------------------------------------------


y_test, ypred_test = fit(clf, X, y, dataset_id, n_splits=5, n_components=50)[1:]
pipe = make_pipeline(PCA(whiten=True, n_components=50), clf)

class_count = {i: np.count_nonzero(y == i) for i in set(y)}
sample_weight = [class_weight[i] for i in y]
pipe.fit(X, y)

# Calibrate model.

calibrated_clf = CalibratedClassifierCV(pipe, cv=5)
calibrated_clf.fit(X, y)

y_score = pipe.predict_proba(X_final_test)

fname = "/angel_workspace/ros_bags/y_pred_.png"
save_matrix_image(
    y_score[np.argsort(y_final_test)].T, fname, max_w=8000, aspect_ratio=4, first_ind=0
)

lb = preprocessing.LabelBinarizer()
y_true = lb.fit(range(y_score.shape[1])).transform(y_final_test)
s1 = average_precision_score(y_true, y_score)

y_score = calibrated_clf.predict_proba(X_final_test)
s2 = average_precision_score(y_true, y_score)

print(s1, s2)


if True:
    y_test = y_final_test
    ypred_test = pipe.predict(X_final_test)
else:
    y_test = y
    ypred_test = pipe.predict(X)


accuracy_score(y_test, ypred_test)

act_str_list = [inv_act_map[key] for key in sorted(list(set(y)))]
fname = "/angel_workspace/model_files/recipe_m2_apply_tourniquet_v0.052.2.pkl"
with open(fname, "wb") as of:
    pickle.dump([label_to_ind, 1, pipe, act_str_list], of)


# y2 = pipe.named_steps['pca'].transform(X)


fname = "/angel_workspace/ros_bags/activity_hmm_training_data.pkl"
with open(fname, "wb") as of:
    pickle.dump([pipe.predict_proba(X), y], of)


N = 25
hidden_layer_sizes = [N, N, N, N, N, N, N, N]
alpha = 1e-6
activation = "relu"
solver = "adam"
solver = "lbfgs"
clf = MLPClassifier(
    solver=solver,
    alpha=alpha,
    activation=activation,
    hidden_layer_sizes=hidden_layer_sizes,
    learning_rate_init=0.001,
    max_iter=100,
    tol=1e-3,
    n_iter_no_change=np.inf,
    early_stopping=False,
    verbose=True,
)
# pipe = make_pipeline(StandardScaler(), clf)
# pipe.fit(X, y)
# print(pipe.score(X, y))

alphas = np.logspace(-4, 4, 50)
shuffle(alphas)
errs = []
for alpha in alphas:
    clf.alpha = alpha
    err, y_test, ypred_test = fit(clf)
    errs.append(err)
    print(alpha, err)
    fname = "/mnt/data10tb/projects/ptg/new_contact_process/models/%s.p" % str(
        int(err * 1000000)
    ).zfill(6)
    with open(fname, "wb") as of:
        pickle.dump(clf, of)

plt.semilogx(alphas[: len(errs)], errs, "ro")


fname = "/mnt/data10tb/projects/ptg/new_contact_process/models/432301.p"
with open(fname, "rb") as of:
    clf2 = pickle.load(of)


def fit_mlp(x, y):
    solver = "adam"
    alpha = 10 ** (-(np.random.rand(1)) * 10)
    # ind = int(np.random.randint(low=0, high=3))

    ind = 2
    if ind == 0:
        activation = "logistic"
    elif ind == 1:
        activation = "tanh"
    elif ind == 2:
        activation = "relu"

    alpha = 1e-8
    hidden_layer_sizes = [50] * np.random.randint(1, 4)
    clf = MLPClassifier(
        solver=solver,
        alpha=alpha,
        activation=activation,
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=0.01,
        max_iter=200,
        tol=1e-6,
        n_iter_no_change=np.inf,
        early_stopping=True,
        verbose=True,
    )
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(X, y)

    clf.solver = "lbfgs"
    # clf.solver = 'adam'
    clf.verbose = True
    # clf.verbose = False
    clf.tol = 1e-6
    # clf.verbose = False
    clf.warm_start = True
    clf.max_iter = 10000

    while True:
        pipe.fit(X, y)
        print(pipe.score(X, y))


y_pred = pipe.predict(X)


A = confusion_matrix(y_test, ypred_test, labels=range(max(y) + 1), normalize="pred")

plt.close("all")
fig = plt.figure(num=None, figsize=(14, 10), dpi=80)
plt.rc("font", **{"size": 22})
plt.rc("axes", linewidth=4)
plt.imshow(A)
cbar = plt.colorbar()
cbar.set_label("Percent", rotation=270, fontsize=34)
cbar.set_label("Percent", labelpad=+30, rotation=270, fontsize=34)
plt.xticks(range(0, len(A)))
plt.yticks(range(0, len(A)))
plt.ylabel("True Steps", fontsize=34)
plt.xlabel("Predicted Steps", fontsize=34)
plt.tight_layout()
plt.savefig("/angel_workspace/ros_bags/tourniquet_step_classification.png")
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
fname = "/angel_workspace/ros_bags/kitware_test_results_test.mscoco.json"
with open(pred_train_fname, "r") as f:
    dat = json.load(f)

ann_by_image = {}
for ann in dat["annotations"]:
    if ann["image_id"] not in ann_by_image:
        ann_by_image[ann["image_id"]] = [ann]
    else:
        ann_by_image[ann["image_id"]].append(ann)

# Generate features using code above.

fname = "/angel_workspace/model_files/recipe_m2_apply_tourniquet_v0.052.pkl"
with open(fname, "rb") as of:
    label_to_ind, feat_ver, clf, act_str_list = pickle.load(of)


label_to_ind = {cat["name"]: cat["id"] - min_cat for cat in dat["categories"]}
act_id_to_str = {cat["id"]: cat["name"] for cat in dat["categories"]}


X = []
y = []
dataset_id = []
last_dset = 0
for image_id in ann_by_image:
    label_vec = []
    left = []
    right = []
    top = []
    bottom = []
    label_confidences = []
    obj_obj_contact_state = []
    obj_obj_contact_conf = []
    obj_hand_contact_state = []
    obj_hand_contact_conf = []

    for ann in ann_by_image[image_id]:
        label_vec.append(act_id_to_str[ann["category_id"]])
        left.append(ann["bbox"][0])
        right.append(ann["bbox"][1])
        top.append(ann["bbox"][2])
        bottom.append(ann["bbox"][3])
        label_confidences.append(ann["confidence"])
        try:
            obj_obj_contact_state.append(ann["obj-obj_contact_state"])
            obj_obj_contact_conf.append(ann["obj-obj_contact_conf"])
            obj_hand_contact_state.append(ann["obj-hand_contact_state"])
            obj_hand_contact_conf.append(ann["obj-hand_contact_conf"])
        except KeyError:
            pass

    if len(label_vec) > 10:
        raise Exception()

    feature_vec = obj_det2d_set_to_feature(
        label_vec,
        left,
        right,
        top,
        bottom,
        label_confidences,
        None,
        obj_obj_contact_state,
        obj_obj_contact_conf,
        obj_hand_contact_state,
        obj_hand_contact_conf,
        label_to_ind,
        version=1,
    )

    X.append(feature_vec.ravel())
    try:
        dataset_id.append(image_id_to_dataset[image_id])
        last_dset = dataset_id[-1]
    except:
        dataset_id.append(last_dset)

    try:
        # y.append(trim_act_map[image_activity_gt[image_id]])
        y.append(image_activity_gt[image_id])
    except:
        y.append(0)

X = np.array(X)
y = np.array(y)
dataset_id = np.array(dataset_id)


# ----------------------------------------------------------------------------
y_pred = clf.predict_proba(X)

fname = "/angel_workspace/ros_bags/det_step_class.png"
save_matrix_image(X.T, fname, max_w=8000, aspect_ratio=1, first_ind=1)

fname = "/angel_workspace/ros_bags/step_class.png"
save_matrix_image(
    clf.predict_proba(X).T, fname, max_w=8000, aspect_ratio=4, first_ind=0
)


# h2 = w


# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Score detections themselves.
label_str = {
    0: "Background",
    1: "Not started.",
    2: "Place tourniquet over affected extremity 2-3 inches above wound site (step 1).",
    3: "In between step 1 and step 2.",
    4: "Pull tourniquet tight (step 2).",
    5: "In between step 2 and step 3.",
    6: "Apply strap to strap body (step 3).",
    7: "In between step 3 and step 4.",
    8: "Turn windless clock wise or counter clockwise until hemorrhage is controlled (step 4).",
    9: "In between step 4 and step 5.",
    10: "Lock windless into the windless keeper (step 5).",
    11: "In between step 5 and step 6.",
    12: "Pull remaining strap over the windless keeper (step 6).",
    13: "In between step 6 and step 7.",
    14: "Secure strap and windless keeper with keeper securing device (step 7).",
    15: "In between step 7 and step 8.",
    16: "Mark time on securing device strap with permanent marker (step 8).",
    17: "Finished.",
}

y_ = np.zeros((len(y), len(act_id_to_str)))
acts = [act_id_to_str[key] for key in sorted(list(act_id_to_str.keys()))]
for i in range(len(y)):
    for j in range(len(acts)):
        act_str = acts[j]
        if y[i] in [0, 1, 3, 5, 7, 9, 11, 13, 15, 17]:
            # Background
            if (
                "(finished)" in act_str
                or "(before)" in act_str
                or "1.5" in act_str
                or "2.5" in act_str
                or "3.5" in act_str
                or "4.5" in act_str
                or "5.5" in act_str
                or "6.5" in act_str
                or "7.5" in act_str
            ):
                y_[i, j] = True
        elif y[i] == 2:
            # Step 1
            if "(step 1)" in act_str:
                y_[i, j] = True
        elif y[i] == 4:
            # Step 2
            if "(step 2)" in act_str:
                y_[i, j] = True
        elif y[i] == 6:
            # Step 3
            if "(step 3)" in act_str:
                y_[i, j] = True
        elif y[i] == 8:
            # Step 4
            if "(step 4)" in act_str:
                y_[i, j] = True
        elif y[i] == 10:
            # Step 5
            if "(step 5)" in act_str:
                y_[i, j] = True
        elif y[i] == 12:
            # Step 6
            if "(step 6)" in act_str:
                y_[i, j] = True
        elif y[i] == 14:
            # Step 7
            if "(step 7)" in act_str:
                y_[i, j] = True
        elif y[i] == 16:
            # Step 8
            if "(step 8)" in act_str:
                y_[i, j] = True
        else:
            raise Exception()

maps = [average_precision_score(y_[:, i], X[:, i]) for i in range(y_.shape[1])]
examples = [np.mean(y_[:, i]) for i in range(len(maps))]
maps = np.array(maps)
examples = np.array(examples)
[
    "MAP: %0.1f%% (%0.0f%% examples) - %s"
    % (maps[i] * 100, (examples[i]) * 100, acts[i])
    for i in np.argsort(maps / examples)[::-1]
]

fname = "/angel_workspace/ros_bags/pr_curves.pdf"
with PdfPages(fname) as pdf:
    for i in np.argsort(maps / examples)[::-1]:
        fig = plt.figure(figsize=(3, 3))
        print("%0.1f%% true examples" % ((np.mean(y_[:, i])) * 100))
        plt.plot(*precision_recall_curve(y_[:, i], X[:, i])[:2][::-1], label="Detector")
        plt.plot(
            *precision_recall_curve(y_[:, i], np.random.rand(len(y_)))[:2][::-1],
            label="Guessing",
        )
        plt.legend()
        plt.ylim([0, 1])
        plt.title(acts[i])
        pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig

    plt.close("all")
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
import angel_system
from angel_system.activity_hmm.core import ActivityHMMRos

config_fname = "/angel_workspace/config/tasks/task_steps_config-recipe_m2_apply_tourniquet_v0.051.yaml"
print(f"Loading HMM with recipe {config_fname}")
live_model = ActivityHMMRos(config_fname)

if True:
    fname = "/angel_workspace/model_files/recipe_m2_apply_tourniquet_v0.052.2.pkl"
    with open(fname, "rb") as of:
        ret = pickle.load(of)
        label_to_ind, feat_ver, pipe, act_str_list = ret
        feat_ver = 1

if False:
    # Train model on test set.
    clf = RandomForestClassifier(
        max_depth=10,
        random_state=0,
        n_estimators=1000,
        max_features=0.1,
        bootstrap=True,
    )
    clf.fit(X, y)
elif False:
    clf = LogisticRegression(
        random_state=0,
        C=0.0006951927961775606,
        l1_ratio=0.1,
        max_iter=1000,
        penalty="elasticnet",
        fit_intercept=False,
        solver="saga",
    )
    clf = make_pipeline(PCA(whiten=True, n_components=40), clf)
    clf = CalibratedClassifierCV(clf, cv=5)
    clf.fit(X, y)
elif False:
    act_str_list = [inv_act_map[key] for key in sorted(list(set(y)))]
    fname = "/angel_workspace/model_files/recipe_m2_apply_tourniquet_v0.051.pkl"
    with open(fname, "wb") as of:
        pickle.dump([label_to_ind, 1, clf, act_str_list], of)

# live_model.class_std_conf = np.maximum(live_model.class_std_conf, 5)
live_model.med_class_duration = np.ones(len(live_model.med_class_duration)) * 10

# y_pred = clf.predict_proba(X)
y_pred_final_test = clf.predict_proba(X_final_test)
# plt.imshow(np.cov(y_pred.T)); plt.colorbar()
# plt.imshow(np.cov(y_pred[y == i].T)); plt.colorbar()

class_mean_conf = []
class_std_conf = []
for i in range(max(y) + 1):
    ind = y_final_test == i
    v = y_pred_final_test[ind]
    class_mean_conf.append(np.mean(v, axis=0))

    if True:
        cov = np.cov(v.T)
        cov = cov + np.identity(len(cov)) * 1e-6
        class_std_conf.append(cov)
    else:
        class_std_conf.append(np.std(v, axis=0))

class_mean_conf = np.array(class_mean_conf)
class_std_conf = np.array(class_std_conf)
# plt.imshow(class_mean_conf); plt.colorbar()
# plt.imshow(class_std_conf[1]); plt.colorbar()

if False:
    # class_mean_conf = np.identity(len(live_model.class_mean_conf))
    # class_std_conf += 0.1
    interstitial_ids = [1, 3, 5, 7, 9, 11, 13, 15]
    # mask = np.zeros(class_mean_conf.shape, dtype=bool)
    for i in interstitial_ids:
        class_std_conf[i][:] *= 5

if False:
    # class_mean_conf = np.identity(len(live_model.class_mean_conf))
    # class_std_conf += 0.1
    interstitial_ids = [6]
    # mask = np.zeros(class_mean_conf.shape, dtype=bool)
    for i in interstitial_ids:
        class_std_conf[i][:] = np.identity(len(class_std_conf[i])) * 2

live_model.set_hmm_mean_and_std(class_mean_conf, class_std_conf)

if False:
    fname = "/angel_workspace/model_files/recipe_m2_apply_tourniquet.3.npy"
    tmp = np.array([None, None], dtype=object)
    tmp[0] = class_mean_conf
    tmp[1] = class_std_conf
    np.save(fname, tmp, allow_pickle=True)


if False:
    fname = "/angel_workspace/ros_bags/hhm_class_mean_conf.png"
    save_matrix_image(
        live_model.class_mean_conf, fname, max_w=4000, aspect_ratio=1, first_ind=0
    )

    fname = "/angel_workspace/ros_bags/hhm_class_std_conf.png"
    a = live_model.class_mean_conf / live_model.class_std_conf
    a = a / np.max(a)
    save_matrix_image(a, fname, max_w=4000, aspect_ratio=1, first_ind=0)


y_pred_final_test = clf.predict_proba(X_final_test)
for ds_id in sorted(list(set(dataset_id_final_test))):
    ind = dataset_id_final_test == ds_id
    y_ = y_final_test[ind]
    y_pred_ = y_pred_final_test[ind]

    if False:
        ind = random.permutation(range(len(y_)))
        y_ = y_[ind]
        y_pred_ = y_pred_[ind]
        ind = np.argsort(y_)
        y_ = y_[ind]
        y_pred_ = y_pred_[ind]

    if False:
        fname = "/angel_workspace/ros_bags/y_pred_.png"
        save_matrix_image(y_pred_.T, fname, max_w=8000, aspect_ratio=4, first_ind=0)

    dt = 1 / 30
    t0 = 0
    N = len(y_pred_)
    time_windows = np.zeros((N, 2))
    time_windows[:, 0] = np.linspace(t0, t0 + dt * (N - 1), N)
    time_windows[:, 1] = time_windows[:, 0] + (dt - 1e-6)
    valid = np.ones(len(time_windows), dtype=bool)

    save_fname = (
        "/angel_workspace/ros_bags/new/hmm-ds=%s.png" % dsets[ds_id].split("/")[-3]
    )
    viz_step_finished_conf(
        live_model, y_pred_, y_, time_windows, valid, save_fname=save_fname
    )
