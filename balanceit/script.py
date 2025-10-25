import string
from unidecode import unidecode
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler, KMeansSMOTE
from imblearn.under_sampling import OneSidedSelection, RandomUnderSampler
from imblearn.ensemble import BalancedBaggingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imbens.ensemble import AdaUBoostClassifier, AsymBoostClassifier, CompatibleAdaBoostClassifier, SMOTEBoostClassifier, RUSBoostClassifier, AdaCostClassifier, OverBoostClassifier, SelfPacedEnsembleClassifier
import json

def clean_text(text, stop_words):
    cleaned_text = unidecode(text.encode('ascii', 'ignore').decode('utf-8'))
    cleaned_text = ''.join(char for char in cleaned_text if char not in string.punctuation)
    cleaned_text = ' '.join(cleaned_text.split())
    cleaned_text = ' '.join(word for word in cleaned_text.split() if word.lower() not in stop_words)
    cleaned_text = cleaned_text.lower()
    return cleaned_text

def compute_metrics(y_test, y_pred, y_score):
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'gmean': geometric_mean_score(y_test, y_pred, average='weighted'),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_score, multi_class='ovr')
    }

def get_sampler(sampler_name, params):
    # If params is a string, try to parse it as JSON
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            params = {}
    
    # If params is None or not a dictionary, use an empty dictionary
    if not isinstance(params, dict):
        params = {}

    if sampler_name == "none":
        return None
    elif sampler_name == "ADASYN":
        return ADASYN(
            sampling_strategy=params.get("sampling_strategy", "auto"),
            n_neighbors=int(params.get("n_neighbors", 5)),
            random_state=int(params.get("random_state")) if params.get("random_state") else None
        )
    elif sampler_name == "SMOTE":
        return SMOTE(
            sampling_strategy=params.get("sampling_strategy", "auto"),
            k_neighbors=int(params.get("k_neighbors", 5)),
            random_state=int(params.get("random_state")) if params.get("random_state") else None
        )
    elif sampler_name == "RandomOverSampler":
        return RandomOverSampler(
            sampling_strategy=params.get("sampling_strategy", "auto"),
            random_state=int(params.get("random_state")) if params.get("random_state") else None,
            shrinkage=float(params.get("shrinkage")) if params.get("shrinkage") else None
        )
    elif sampler_name == "OneSidedSelection":
        return OneSidedSelection(
            sampling_strategy=params.get("sampling_strategy", "auto"),
            n_neighbors=int(params.get("n_neighbors")) if params.get("n_neighbors") else None,
            n_seeds_S=int(params.get("n_seeds_S", 1)),
            random_state=int(params.get("random_state")) if params.get("random_state") else None,
        )
    elif sampler_name == "KMeansSMOTE":
        return KMeansSMOTE(
            sampling_strategy=params.get("sampling_strategy", "auto", "minority", "not minority", "all", "not majority") if params.get("sampling_strategy") else "auto",
            k_neighbors=int(params.get("k_neighbors", 2)),
            random_state=int(params.get("random_state")) if params.get("random_state") else None,
            cluster_balance_threshold=float(params.get("cluster_balance_threshold")) if params.get("cluster_balance_threshold") else "auto",
            density_exponent=float(params.get("density_exponent")) if params.get("density_exponent") else "auto",
        )
    elif sampler_name == "RandomUnderSampler":
        return RandomUnderSampler(
            sampling_strategy=params.get("sampling_strategy", "auto", "majority", "not minority", "not majority", "all") if params.get("sampling_strategy") else "auto",
            random_state=int(params.get("random_state")) if params.get("random_state") else None,
            replacement=params.get("replacement", False),
        )
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")
    
def get_model(model_name, params):
    # If params is a string, try to parse it as JSON
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            params = {}
    
    # If params is None or not a dictionary, use an empty dictionary
    if not isinstance(params, dict):
        params = {}

    if model_name == "none":
        return None
    elif model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", None)) if params.get("max_depth") else None,
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            random_state=int(params.get("random_state")) if params.get("random_state") else None,
            min_weight_fraction_leaf=float(params.get("min_weight_fraction_leaf", 0.0)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", None)) if params.get("max_leaf_nodes") else None,
            min_impurity_decrease=float(params.get("min_impurity_decrease", 0.0)),
            class_weight=params.get("class_weight", None),
            ccp_alpha=float(params.get("ccp_alpha", 0.0)),
            max_samples=float(params.get("max_samples", None)) if params.get("max_samples") else None,
            warm_start=params.get("warm_start", False),
            verbose=2,
            )
    elif model_name == "Logistic Regression":
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 100)),

        )
    elif model_name == "XGBClassifier":
        return XGBClassifier(
            n_estimators=int(params.get("n_estimators", 1)),
            booster=params.get("booster", "gbtree", "gblinear", "dart") if params.get("booster") else "gbtree",
            max_depth=int(params.get("max_depth", 6)),
            dart_normalized_type=params.get("dart_normalized_type", "TREE", "FOREST") if params.get("dart_normalized_type") else "TREE",
            tree_method=params.get("tree_method", "auto", "exact", "approx", "hist") if params.get("tree_method") else "auto",
            learning_rate=float(params.get("learning_rate", 0.3)),
            gamma=float(params.get("gamma", 0.0)),
            min_child_weight=int(params.get("min_child_weight", 1)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            colsample_bylevel=float(params.get("colsample_bylevel", 1.0)),
            colsample_bynode=float(params.get("colsample_bynode", 1.0)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            scale_pos_weight=float(params.get("scale_pos_weight", 1.0)),
            max_iterations=int(params.get("max_iterations", 20)),
            tol = float(params.get("tol", 0.01)),
            enable_global_explain=params.get("enable_global_explain", False),

        )
    elif model_name == "LGBMClassifier":
        return LGBMClassifier(
            boosting = params.get("boosting", "gbdt", "rf", "dart") if params.get("boosting") else "gbdt",
            data_sample_strategy = params.get("data_sample_strategy", "bagging", "goss") if params.get("data_sample_strategy") else "bagging",
            num_iterations = int(params.get("num_iterations", 100)),
            learning_rate = float(params.get("learning_rate", 0.1)),
            num_leaves = int(params.get("num_leaves", 31)),
            seed = int(params.get("seed", 0)),
        )
    elif model_name == "AdaBoostClassifier":
        return AdaBoostClassifier(
            n_estimators=int(params.get("n_estimators", 50)),
            learning_rate=float(params.get("learning_rate", 1.0)),
            algorithm=params.get("algorithm", "SAMME.R", "SAMME") if params.get("algorithm") else "SAMME.R",
            random_state=int(params.get("random_state")) if params.get("random_state") else None,
        )
    elif model_name == "AdaUBoostClassifier":
        return AdaUBoostClassifier(
            n_estimators=int(params.get("n_estimators", 50)),
            learning_rate=float(params.get("learning_rate", 1.0)),
            random_state=int(params.get("random_state")) if params.get("random_state") else None,
            early_termination=params.get("early_termination", False),
            algorithm=params.get("algorithm", "SAMME.R", "SAMME") if params.get("algorithm") else "SAMME.R",

        )
    elif model_name == "AsymBoostClassifier":
        return AsymBoostClassifier(
            n_estimators=int(params.get("n_estimators", 50)),
            learning_rate=float(params.get("learning_rate", 1.0)),
            random_state=int(params.get("random_state")) if params.get("random_state") else None,
            early_termination=params.get("early_termination", False),
            algorithm=params.get("algorithm", "SAMME.R", "SAMME") if params.get("algorithm") else "SAMME.R",
        )
    elif model_name == "CompatibleAdaBoostClassifier":
        return CompatibleAdaBoostClassifier(
            n_estimators=int(params.get("n_estimators", 50)),
            learning_rate=float(params.get("learning_rate", 1.0)),
            random_state=int(params.get("random_state")) if params.get("random_state") else None,
            early_termination=params.get("early_termination", False),
            algorithm=params.get("algorithm", "SAMME.R", "SAMME") if params.get("algorithm") else "SAMME.R",
        )
    elif model_name == "GradientBoostingClassifier":
        return GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            loss=params.get("loss", "logloss", "exponential") if params.get("loss") else "logloss",
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            subsample=float(params.get("subsample", 1.0)),
            max_features=params.get("max_features", None),
            random_state=int(params.get("random_state")) if params.get("random_state") else None,
            ccp_alpha=float(params.get("ccp_alpha", 0.0)),
            criterion=params.get("criterion", "friedman_mse", "squared_error") if params.get("criterion") else "friedman_mse",
            min_weight_fraction_leaf=float(params.get("min_weight_fraction_leaf", 0.0)),
            min_impurity_decrease=float(params.get("min_impurity_decrease", 0.0)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", None)) if params.get("max_leaf_nodes") else None,
            warm_start=params.get("warm_start", False),
            validation_fraction=float(params.get("validation_fraction", 0.1)),
            n_iter_no_change=int(params.get("n_iter_no_change", None)) if params.get("n_iter_no_change") else None,
            tol=float(params.get("tol", 1e-4)),
        ),
    elif model_name == "HistGradientBoostingClassifier":
        return HistGradientBoostingClassifier(
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_iter=int(params.get("max_iter", 100)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
            max_depth=int(params.get("max_depth")) if params.get("max_depth") else None,
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            l2_regularization=float(params.get("l2_regularization", 0.0)),
            max_bins=int(params.get("max_bins", 255)),
            max_features=params.get("max_features", 1.0),
            warm_start=params.get("warm_start", False),
            early_stopping=params.get("early_stopping", "auto", "valid", "train") if params.get("early_stopping") else "auto",
            scoring=params.get("scoring", "loss", "accuracy") if params.get("scoring") else "loss",
            validation_fraction=float(params.get("validation_fraction", 0.1)),
            n_iter_no_change=int(params.get("n_iter_no_change", 10)),
            tol=float(params.get("tol", 1e-7)),
            random_state=int(params.get("random_state")) if params.get("random_state") else None,
        )
    elif model_name == "SMOTEBoostClassifier":
        return SMOTEBoostClassifier(
            n_estimators=int(params.get("n_estimators", 50)),
            k_neighbors=int(params.get("k_neighbors", 5)),
            learning_rate=float(params.get("learning_rate", 1.0)),
            random_state=int(params.get("random_state")) if params.get("random_state") else None,
            early_termination=params.get("early_termination", False),
            algorithm=params.get("algorithm", "SAMME.R", "SAMME") if params.get("algorithm") else "SAMME.R",
        )
    elif model_name == "RUSBoostClassifier":
        return RUSBoostClassifier(
            n_estimators=int(params.get("n_estimators", 50)),
            learning_rate=float(params.get("learning_rate", 1.0)),
            random_state=int(params.get("random_state")) if params.get("random_state") else None,
            algorithm=params.get("algorithm", "SAMME.R", "SAMME") if params.get("algorithm") else "SAMME.R",
            sampling_strategy=params.get("sampling_strategy", "auto", "not majority", "not minority", "not majority", "all") if params.get("sampling_strategy") else "auto",
            replacement=params.get("replacement", False), 
        )
    elif model_name == "AdaCostClassifier":
        return AdaCostClassifier(
            n_estimators=int(params.get("n_estimators", 50)),
            learning_rate=float(params.get("learning_rate", 1.0)),
            random_state=int(params.get("random_state")) if params.get("random_state") else None,
            algorithm=params.get("algorithm", "SAMME.R", "SAMME") if params.get("algorithm") else "SAMME.R",
            early_termination=params.get("early_termination", False),
            cost_matrix=params.get("cost_matrix", None),
        )
    elif model_name == "OverBoostClassifier":
        return OverBoostClassifier(
            n_estimators=int(params.get("n_estimators", 50)),
            learning_rate=float(params.get("learning_rate", 1.0)),
            random_state=int(params.get("random_state")) if params.get("random_state") else None,
            algorithm=params.get("algorithm", "SAMME.R", "SAMME") if params.get("algorithm") else "SAMME.R",
            early_termination=params.get("early_termination", False),
        )
    elif model_name == "SelfPacedEnsembleClassifier":
        return SelfPacedEnsembleClassifier(
            n_estimators=int(params.get("n_estimators", 50)),
            random_state=int(params.get("random_state")) if params.get("random_state") else None,
            k_bins=int(params.get("k_bins", 5)),
            soft_resample=params.get("soft_resample", False),
            replacement=params.get("replacement", True),
        )
    elif model_name == "BaggingClassifier":
        return BaggingClassifier(
            n_estimators=int(params.get("n_estimators", 10)),
            max_samples=float(params.get("max_samples", 1.0)),
            max_features=float(params.get("max_features", 1.0)),
            bootstrap=params.get("bootstrap", True),
            bootstrap_features=params.get("bootstrap_features", False),
            oob_score=params.get("oob_score", False),
            warm_start=params.get("warm_start", False),
            random_state=int(params.get("random_state")) if params.get("random_state") else None,
        )
    elif model_name == "BalancedBaggingClassifier":
        return BalancedBaggingClassifier(
            n_estimators=int(params.get("n_estimators", 10)),
            max_samples=float(params.get("max_samples", 1.0)),
            max_features=float(params.get("max_features", 1.0)),
            bootstrap=params.get("bootstrap", True),
            bootstrap_features=params.get("bootstrap_features", False),
            oob_score=params.get("oob_score", False),
            warm_start=params.get("warm_start", False),
            random_state=int(params.get("random_state")) if params.get("random_state") else None,
            replacement=params.get("replacement", False),
            sampling_strategy=params.get("sampling_strategy", "auto", "majority", "not minority", "all", "not majority") if params.get("sampling_strategy") else "auto",
        )
    elif model_name == "SVC":
        return SVC(
            C=float(params.get("C", 1.0)),
            kernel=params.get("kernel", "rbf", "linear", "poly", "sigmoid", "precomputed") if params.get("kernel") else "rbf",
            degree=int(params.get("degree", 3)),
            gamma=params.get("gamma", "scale", "auto") if params.get("gamma") else "scale",
            coef0=float(params.get("coef0", 0.0)),
            shrinking=params.get("shrinking", True),
            probability=params.get("probability", False),
            tol=float(params.get("tol", 1e-3)),
            max_iter=int(params.get("max_iter", -1)),
            decision_function_shape=params.get("decision_function_shape", "ovr", "ovo") if params.get("decision_function_shape") else "ovr",
            break_ties=params.get("break_ties", False),
            random_state=int(params.get("random_state")) if params.get("random_state") else None,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")