import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
import pickle

from .generate_feature import GenerateFeature
from .feature_engineer import FeatureEngineer
from .constants import (
    LGBM_PREDICTION_MODEL_PATH,
    CAT_PREDICTION_MODEL_PATH,
    BETA_CALIBRATOR_PATH,
    RANKING_THRESHOLD_PATH,
    RANK,
)


class PredictScore:

    def __init__(
        self,
    ):

        self.lgbm_prediction_model = None
        self.cat_prediction_model = None
        self.beta_calibrator = None
        self.ranking_threshold = None

        self.feature_generator = GenerateFeature()

        self.feature_engineer = FeatureEngineer()

        self.lgbm_prediction_model = lgb.Booster(model_file=LGBM_PREDICTION_MODEL_PATH)
        self.cat_prediction_model = CatBoostClassifier().load_model(
            CAT_PREDICTION_MODEL_PATH
        )
        with open(BETA_CALIBRATOR_PATH, "rb") as f:
            self.beta_calibrator = pickle.load(f)

        self.ranking_threshold = pd.read_csv(RANKING_THRESHOLD_PATH)["0"].to_list()

    def get_full_feature(self, data: dict):
        inquiry_date = data["inquiry_date"]
        age = data["age"]
        annual_income = data["annual_income"]
        org_cf2_extract = pd.DataFrame(data["match_data"]["cf2"])
        org_ff_extract = pd.DataFrame(data["match_data"]["ff"])
        org_hom_extract = pd.DataFrame(data["match_data"]["hom"])
        if len(org_cf2_extract) == 0:
            org_cf2_extract = pd.DataFrame(data["similar_data"]["cf2"])
            org_ff_extract = pd.DataFrame(data["similar_data"]["ff"])
            org_hom_extract = pd.DataFrame(data["similar_data"]["hom"])
        generate_feature = self.feature_generator.create_feature(
            inquiry_date,
            annual_income,
            org_cf2_extract,
            org_hom_extract,
        )
        engineered_feature = self.feature_engineer.create_feature(
            inquiry_date,
            age,
            annual_income,
            cf2_df=org_cf2_extract,
            ff_df=org_ff_extract,
            hom_df=org_hom_extract,
        )
        full_data = pd.concat([engineered_feature, generate_feature], axis=1)
        full_data = full_data.map(lambda x: f"{x:.15f}")
        full_data = full_data.astype(float)
        return full_data

    def get_prediction(self, data):
        lgbm = self.lgbm_prediction_model.predict(
            data, num_threads=1, num_iteration=self.lgbm_prediction_model.best_iteration
        )
        cat = self.cat_prediction_model.predict_proba(
            data, ntree_end=self.cat_prediction_model.get_best_iteration()
        )[:, 1]
        final_score = self.beta_calibrator.predict(np.mean([lgbm, cat], 0))
        return np.clip(final_score, 0, 1)

    def assign_rank(self, score: float):
        if score <= self.ranking_threshold[0]:
            return RANK[0]
        elif score <= self.ranking_threshold[1]:
            return RANK[1]
        elif score <= self.ranking_threshold[2]:
            return RANK[2]
        elif score <= self.ranking_threshold[3]:
            return RANK[3]
        else:
            return RANK[4]

    def get_score(self, data_dict: dict):
        full_data = self.get_full_feature(data_dict).reset_index(drop=True)
        prediction = self.get_prediction(full_data)[0]
        rank = self.assign_rank(prediction)
        return {"score": prediction, "rank": rank}
