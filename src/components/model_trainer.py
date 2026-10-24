
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts", "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Splitting training and testing arrays")

            # Separate input features and target column
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Define models
            models = {
                "Random Forest": RandomForestRegressor(
                    random_state=42,
                    n_jobs=1
                ),

                "Decision Tree": DecisionTreeRegressor(
                    random_state=42
                ),

                "Gradient Boosting": GradientBoostingRegressor(
                    random_state=42
                ),

                "Linear Regression": LinearRegression(),

                "XGBRegressor": XGBRegressor(
                    random_state=42,
                    n_jobs=1,
                    verbosity=0
                ),

                "CatBoosting Regressor": CatBoostRegressor(
                    verbose=False,
                    random_seed=42,
                    thread_count=1,
                    allow_writing_files=False
                ),

                "AdaBoost Regressor": AdaBoostRegressor(
                    random_state=42
                )
            }

            # Hyperparameter grids
            # Corrected: friedman_mse removed from DecisionTreeRegressor
            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "absolute_error"
                    ]
                },

                "Random Forest": {
                    "n_estimators": [32, 64]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [32, 64],
                    "subsample": [0.8]
                },

                "Linear Regression": {},

                "XGBRegressor": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [32, 64]
                },

                "CatBoosting Regressor": {
                    "depth": [4, 6],
                    "learning_rate": [0.05],
                    "iterations": [50]
                },

                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.5],
                    "n_estimators": [32, 64]
                }
            }

            logging.info("Starting model evaluation")

            # Train and evaluate models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            if not model_report:
                raise ValueError(
                    "No model scores were returned."
                )

            # Select model with highest R2 score
            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(
                f"Best model: {best_model_name}, "
                f"R2 score: {best_model_score:.4f}"
            )

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Final test-set evaluation
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            logging.info(
                f"Model training completed. "
                f"Test R2 score: {r2_square:.4f}"
            )

            print(f"\nBest Model: {best_model_name}")
            print(f"Best Model R2 Score: {best_model_score:.4f}")
            print(f"Final Test R2 Score: {r2_square:.4f}")

            return r2_square

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)
