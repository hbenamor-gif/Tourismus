
# ==============================================================================
# 1. IMPORTS & DEPENDENCIES
# ==============================================================================
# Data manipulation and utilities
import os
import time
import numpy as np
import pandas as pd

# Preprocessing and modeling pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
import xgboost as xgb

# Model serialization and MLflow experiment tracking
import joblib
import mlflow
import mlflow.sklearn
from mlflow.entities import Metric, Param
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


# ==============================================================================
# 2. MLFLOW CONFIGURATION
# ==============================================================================
# Set tracking server URI and target experiment name
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Tourism Development")


# ==============================================================================
# 3. DATA LOADING & CLEANING
# ==============================================================================
# Load prepared train and test splits
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

# Drop non-predictive identifier columns if still present
Xtrain = Xtrain.drop(columns=["CustomerID", "Unnamed: 0"], errors="ignore")
Xtest = Xtest.drop(columns=["CustomerID", "Unnamed: 0"], errors="ignore")


# ==============================================================================
# 4. FEATURE DEFINITIONS
# ==============================================================================
# 4.1 Nominal categorical features (text labels -> OneHotEncoder)
nominal_cols = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
]

# 4.2 Binary and ordinal categorical features (already numeric -> passthrough for trees)
binary_ordinal_cols = [
    "CityTier",
    "PreferredPropertyStar",
    "Passport",
    "OwnCar",
]

# 4.3 Continuous and count numerical features (median imputation + scaling)
numeric_cols = [
    "Age",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "NumberOfTrips",
    "PitchSatisfactionScore",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]

# ==============================================================================
# 5. PREPROCESSING & MODEL PIPELINE SETUP
# ==============================================================================
# Compute positive class weight dynamically to handle class imbalance (~4.18)
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Build column transformer aligned with all three feature groups
preprocessor = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_cols),
    (SimpleImputer(strategy="most_frequent"), binary_ordinal_cols),
    (
        make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore"),
        ),
        nominal_cols,
    ),
    remainder="drop",
)

# Initialize base XGBoost classifier with imbalance scaling
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
)

# Define hyperparameter grid for cross-validated search
param_grid = {
    "xgbclassifier__n_estimators": [50, 100, 200],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__colsample_bytree": [0.7, 0.85, 1.0],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__reg_lambda": [1.0, 5.0],
}

# Chain category cleaner -> preprocessor -> classifier into an end-to-end pipeline
model_pipeline = make_pipeline(
    preprocessor,
    xgb_model,
)



# ==============================================================================
# 6. HYPERPARAMETER TUNING & MLFLOW EXPERIMENT LOGGING
# ==============================================================================
# Initialize MLflow client for batch-logging child runs
client = MlflowClient()

with mlflow.start_run(run_name="XGBoost_GridSearch") as parent_run:
    parent_id = parent_run.info.run_id
    exp_id = parent_run.info.experiment_id

     # --------------------------------------------------------------------------
    # 6.1 Execute GridSearchCV Optimizing F1
    # --------------------------------------------------------------------------
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        scoring="f1",
        return_train_score=True,
        n_jobs=-1,
    )
    grid_search.fit(Xtrain, ytrain)


    # --------------------------------------------------------------------------
    # 6.2 Batch-Log All Candidate Runs as Nested Child Runs
    # --------------------------------------------------------------------------
    results = grid_search.cv_results_
    sorted_indices = results["rank_test_score"].argsort()

    for idx in sorted_indices:
        param_set = results["params"][idx]
        rank_f1 = int(results["rank_test_score"][idx])
        ts = int(time.time() * 1000)

        # Create child run linked to the parent run
        child = client.create_run(
            experiment_id=exp_id,
            tags={
                "mlflow.parentRunId": parent_id,
                "mlflow.runName": f"Candidate_{idx+1:03d}_Rank_{rank_f1}",
            },
        )
        # Log parameters and all CV metrics in a single batch request
        client.log_batch(
            run_id=child.info.run_id,
            metrics=[
                Metric("cv_rank_f1", float(rank_f1), ts, 0),
                Metric("cv_f1_mean", float(results["mean_test_score"][idx]), ts, 0),
                Metric("cv_f1_std", float(results["std_test_score"][idx]), ts, 0),
                Metric("cv_train_f1_mean", float(results["mean_train_score"][idx]), ts, 0),
                Metric(
                    "cv_overfit_gap",
                    float(results["mean_train_score"][idx] - results["mean_test_score"][idx]),
                    ts,
                    0,
                ),
            ],
            params=[Param(k, str(v)) for k, v in param_set.items()],
        )
        # Mark child run status as completed
        client.set_terminated(child.info.run_id, status="FINISHED")

    # --------------------------------------------------------------------------
    # 6.3 Best Model Evaluation & Parent Run Logging
    # --------------------------------------------------------------------------
    best_idx = grid_search.best_index_

    # Log optimal hyperparameters, CV summary, and decision threshold
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metrics({
        "best_cv_f1_mean": float(results["mean_test_score"][best_idx]),
        "best_cv_f1_std": float(results["std_test_score"][best_idx]),
        "best_cv_train_f1_mean": float(results["mean_train_score"][best_idx]),
    })

    classification_threshold = 0.45
    mlflow.log_param("classification_threshold", classification_threshold)

    # Retrieve best fitted pipeline
    best_model = grid_search.best_estimator_

    # Generate probability predictions and apply custom threshold
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Compute classification reports for train and test splits
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log train and test performance metrics to the parent run
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"],
    })


    # ==========================================================================
    # 7. MODEL SERIALIZATION & ARTIFACT LOGGING
    # ==========================================================================
    # Save best model pipeline locally as joblib for deployment scripts
    model_path = "tourism_project/deployment/best_model.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="joblib_model")

    # Log standard MLflow sklearn model with input/output signature for registry & serving
    signature = infer_signature(Xtrain, y_pred_train)
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        signature=signature,
        input_example=Xtrain.iloc[:5],
    )

    print(f"Model saved locally to {model_path} and logged to MLflow.")
    print(f"Best CV F1: {grid_search.best_score_:.4f}  {results['std_test_score'][best_idx]:.4f}")
    print(f"Test F1:    {test_report['1']['f1-score']:.4f}")
