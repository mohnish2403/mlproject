
import os
import sys
import pickle

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)

        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object saved successfully: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a saved Python object from a file.
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        logging.info(f"Object loaded successfully: {file_path}")

        return obj

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    param
):
    """
    Train and evaluate multiple regression models.
    Return a dictionary containing test R2 scores.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            # Get parameters for the current model
            para = param.get(model_name, {})

            # Perform GridSearchCV only when parameters exist
            if para:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=para,
                    cv=3,
                    n_jobs=1,
                    error_score="raise",
                    scoring="r2"
                )

                gs.fit(X_train, y_train)

                # Apply the best parameters
                model.set_params(**gs.best_params_)

                logging.info(
                    f"Best parameters for {model_name}: "
                    f"{gs.best_params_}"
                )

            # Train the selected model
            model.fit(X_train, y_train)

            # Predict on test data
            y_test_pred = model.predict(X_test)

            # Calculate test R2 score
            test_model_score = r2_score(
                y_test,
                y_test_pred
            )

            report[model_name] = float(test_model_score)

            logging.info(
                f"{model_name} test R2 score: "
                f"{test_model_score:.4f}"
            )

        return report

    except Exception as e:
        raise CustomException(e, sys)
