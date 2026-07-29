import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import torch
from .model import DeployFeatureExtractionModel
from typing import List
from .constants import (
    CF2_NORMALIZER_PATH,
    HOM_NORMALIZER_PATH,
    FEATURE_GEN_MODEL_ARGS,
    FEATURE_GEN_MODEL_PATH,
    VALID_PRODUCT_CAT,
    PRODUCT_CODE_CAT_TO_NUM,
    ENDING_COMMENT_CAT_TO_NUM,
    PAYMENT_HISTORY_ENCODING_DICT,
    INFORMATION_TYPE_DEFAULT,
    PAYMENT_HISTORY_LENGTH,
    FINAL_FEATURE_LENGTH,
    FINAL_FEATURE_SHIFT,
    MIN,
    VALID_MEMBER_INDUSTRY_CLASSIFICATION,
    VALID_CONTRACT_MANAGEMENT_CLASSIFICATION,
    VALID_GUARANTOR_CLASSIFICATION,
    VALID_INQUIRY_TYPE,
)


class GenerateFeature:
    def __init__(
        self,
        cf2_cat_order: List[str] = [
            "member_industry_classification",
            "contract_management_classification",
            "product_code",
            "ending_comment",
            "guarantor_classification",
        ],
        cf2_num_order: List[str] = [
            "number_of_payments",
            "contract_amount",
            "outstanding_debt_amount",
            "cumulative_payment_amount",
            "information_type",
            "additional_comment_n",
            "estimate_loan_age",
            "estimate_report_age",
        ],
        cf2_deposit_col: str = "payment_history",
        hom_cat_order: List[str] = [
            "member_industry_classification",
            "inquiry_type",
            "product_code_1",
        ],
        hom_num_order: List[str] = [
            "application_amount",
            "number_of_payments",
            "estimate_application_age",
        ],
        cf2_num_col_normalize: List[str] = [
            "contract_amount",
            "outstanding_debt_amount",
            "cumulative_payment_amount",
        ],
        hom_num_col_normalize: List[str] = ["application_amount"],
    ):
        self.cf2_normalizer = None
        self.hom_normalizer = None
        self.feature_gen_model = DeployFeatureExtractionModel(**FEATURE_GEN_MODEL_ARGS)

        with open(CF2_NORMALIZER_PATH, "rb") as f:
            self.cf2_normalizer = pickle.load(f)
        with open(HOM_NORMALIZER_PATH, "rb") as f:
            self.hom_normalizer = pickle.load(f)
        self.feature_gen_model.load_state_dict(torch.load(FEATURE_GEN_MODEL_PATH))
        self.feature_gen_model.double()
        self.feature_gen_model.eval()

        self.cf2_cat_order = cf2_cat_order
        self.cf2_num_order = cf2_num_order
        self.cf2_deposit_col = cf2_deposit_col
        self.hom_cat_order = hom_cat_order
        self.hom_num_order = hom_num_order
        self.cf2_num_col_normalize = cf2_num_col_normalize
        self.hom_num_col_normalize = hom_num_col_normalize

    def preprocess_cf2(
        self, inquiry_date: datetime, annual_income: int, cf2_df: pd.DataFrame
    ):
        out_df = pd.DataFrame()

        # number of payments
        out_df["number_of_payments"] = cf2_df["number_of_payments"].replace(
            {888: 0, -999: -1}
        )

        # contract amount
        out_df["contract_amount"] = cf2_df["contract_amount"].replace(-999999, np.nan)
        out_df["credit_limit"] = cf2_df["credit_limit"].replace(
            [-999999, -777777], np.nan
        )
        out_df["cash_advance_credit_limit"] = cf2_df[
            "cash_advance_credit_limit"
        ].replace([-999999, -777777], np.nan)
        out_df["contract_amount"] = (
            out_df["contract_amount"]
            .fillna(out_df["credit_limit"])
            .fillna(out_df["cash_advance_credit_limit"])
            .fillna(-1)
        )
        out_df = out_df.drop(columns=["credit_limit", "cash_advance_credit_limit"])

        # remaining debt amount
        out_df["outstanding_debt_amount"] = (
            cf2_df["outstanding_debt_amount"]
            .replace({-999999: np.nan, -444444: 0.5})
            .fillna(-1)
        )

        # cumulative deposit amount
        out_df["cumulative_payment_amount"] = (
            cf2_df["cumulative_payment_amount"].replace(-999999, 0).fillna(0.0)
        )

        # contract management classification
        out_df["contract_management_classification"] = (
            cf2_df["contract_management_classification"]
            .fillna(99.0)
            .apply(lambda x: x if x in VALID_CONTRACT_MANAGEMENT_CLASSIFICATION else 99)
            .replace({21.0: 8.0, 99.0: 0})
        )

        # information type
        out_df["information_type"] = cf2_df["information_type"].apply(
            lambda x: 1 if str(x) in INFORMATION_TYPE_DEFAULT else 0
        )

        # guarantor classification
        out_df["guarantor_classification"] = (
            cf2_df["guarantor_classification"]
            .fillna(9)
            .apply(lambda x: x if x in VALID_GUARANTOR_CLASSIFICATION else 9)
            .replace(9, 0)
        )

        # ending comment
        out_df["ending_comment"] = (
            cf2_df["ending_comment"]
            # .fillna(0)
            .apply(lambda x: x if x in ENDING_COMMENT_CAT_TO_NUM.keys() else 0).apply(
                lambda x: ENDING_COMMENT_CAT_TO_NUM[x]
            )
        )

        # Product Code
        out_df["product_code"] = cf2_df["product_code"].fillna(cf2_df["product_code_1"])
        out_df["product_code"] = (
            out_df["product_code"].fillna("").apply(lambda x: x.strip())
        )
        out_df["product_code"] = (
            out_df["product_code"]
            .fillna("")
            .apply(lambda x: x.strip())
            .apply(lambda x: x[0] if len(x) > 0 else x)
            .apply(lambda x: x if x in VALID_PRODUCT_CAT else "Other")
            .apply(lambda x: PRODUCT_CODE_CAT_TO_NUM[x])
        )
        # membership classification
        out_df["member_industry_classification"] = (
            cf2_df["member_industry_classification"]
            .apply(lambda x: x if x in VALID_MEMBER_INDUSTRY_CLASSIFICATION else 90)
            .apply(lambda x: (x / 10) - 1)
        )
        # additional comments
        out_df["additional_comment_n"] = (
            cf2_df["additional_comment_n"]
            .fillna("")
            .apply(lambda x: x.strip())
            .replace("", pd.NA)
            .astype("object")
        )
        out_df["additional_comment_n"] = (
            ~(out_df["additional_comment_n"].isna())
        ).astype(int)

        # contract date
        out_df["contract_date"] = pd.to_datetime(cf2_df["contract_date"])

        # reporting date
        out_df["reporting_date"] = pd.to_datetime(cf2_df["reporting_date"])

        out_df["estimate_report_age"] = out_df.apply(
            lambda row: (
                -1
                if pd.isnull(row["reporting_date"])
                else (inquiry_date - row["reporting_date"]).days
            ),
            axis=1,
        )
        out_df["estimate_loan_age"] = out_df.apply(
            lambda row: (
                -1
                if pd.isnull(row["contract_date"])
                else (inquiry_date - row["contract_date"]).days
            ),
            axis=1,
        )
        out_df.loc[out_df["estimate_report_age"] < -1, "estimate_report_age"] = -1

        out_df["payment_history"] = (
            cf2_df["payment_history"]
            .apply(lambda x: x.replace(" ", "-") if not pd.isnull(x) else "")
            .apply(lambda x: x.replace("B", "A").replace("C", "A"))
            .apply(
                lambda x: [
                    c if c in PAYMENT_HISTORY_ENCODING_DICT.keys() else "-" for c in x
                ]
            )
            .apply(
                lambda x: ("".join([PAYMENT_HISTORY_ENCODING_DICT[c] for c in x]))[::-1]
            )
        )

        income_k = annual_income / 1000
        for c in self.cf2_num_col_normalize:
            out_df[c] = out_df[c] / income_k
            out_df.loc[out_df[c] < 0, c] = -1

        return out_df

    def preprocess_hom(self, inquiry_date: datetime, income: int, hom_df: pd.DataFrame):
        out_df = pd.DataFrame()

        # inquiry date
        out_df["inquiry_date"] = pd.to_datetime(hom_df["inquiry_date"])

        # inquiry time
        out_df["inquiry_time"] = pd.to_datetime(hom_df["inquiry_time"])

        # inquiry category
        out_df["inquiry_type"] = hom_df["inquiry_type"].apply(
            lambda x: x if x in VALID_INQUIRY_TYPE else 0
        )

        # product code
        out_df["product_code_1"] = (
            hom_df["product_code_1"]
            .fillna("")
            .apply(lambda x: x.strip())
            .apply(lambda x: x[0] if len(x) > 0 else x)
            .apply(lambda x: x if x in VALID_PRODUCT_CAT else "Other")
            .apply(lambda x: PRODUCT_CODE_CAT_TO_NUM[x])
        )

        # application amount
        out_df["application_amount"] = (
            hom_df["application_amount"].replace(-999999, np.nan).fillna(-1)
        )

        # number of payments
        out_df["number_of_payments"] = (
            hom_df["number_of_payments"].replace({-999: np.nan, 888: -1}).fillna(-2)
        )

        # member industry classification
        out_df["member_industry_classification"] = (
            hom_df["member_industry_classification"]
            .apply(lambda x: x if x in VALID_MEMBER_INDUSTRY_CLASSIFICATION else 90)
            .apply(lambda x: (x / 10) - 1)
        )

        # estimate application age
        out_df["estimate_application_age"] = out_df["inquiry_date"].apply(
            lambda x: (inquiry_date - x).days
        )

        out_df = out_df.sort_values(
            by=["inquiry_date", "inquiry_time"], ascending=[False, False]
        )
        out_df = out_df.drop(columns=["inquiry_date", "inquiry_time"])

        income_k = income / 1000
        for c in self.hom_num_col_normalize:
            out_df[c] = out_df[c] / income_k
            out_df.loc[out_df[c] < 0, c] = -1

        return out_df

    def model_preprocess(
        self,
        cf2_num: np.array,
        cf2_cat: np.array,
        cf2_deposit: np.array,
        hom_num: np.array,
        hom_cat: np.array,
    ):

        cf2_num = torch.tensor(cf2_num, dtype=torch.float64)
        cf2_cat = torch.tensor(cf2_cat, dtype=torch.int32)
        cf2_deposit_hist = []
        cf2_deposit_hist_mask = []
        for l in cf2_deposit:
            cf2_deposit_hist.append(
                (
                    (([0] * PAYMENT_HISTORY_LENGTH) + [int(c) for c in l])[
                        -PAYMENT_HISTORY_LENGTH:
                    ]
                )
            )
            cf2_deposit_hist_mask.append(
                (([MIN] * PAYMENT_HISTORY_LENGTH) + ([1] * len(l)))[
                    -PAYMENT_HISTORY_LENGTH:
                ]
            )
        cf2_deposit_hist = torch.tensor(cf2_deposit_hist)
        cf2_deposit_hist_mask = torch.tensor(cf2_deposit_hist_mask)
        hom_num = torch.tensor(hom_num, dtype=torch.float64)
        hom_cat = torch.tensor(hom_cat, dtype=torch.int32)
        cf2_idx = [len(cf2_num)]
        hom_idx = [len(hom_num)]
        return {
            "cf2_num": cf2_num,
            "cf2_cat": cf2_cat,
            "cf2_deposit_hist": cf2_deposit_hist,
            "cf2_deposit_hist_mask": cf2_deposit_hist_mask,
            "hom_num": hom_num,
            "hom_cat": hom_cat,
            "cf2_idx": cf2_idx,
            "hom_idx": hom_idx,
        }

    def generate_feature(self, data_dict):
        with torch.no_grad():
            cf2_idx = data_dict["cf2_idx"]
            hom_idx = data_dict["hom_idx"]
            cf2_num = data_dict["cf2_num"]
            cf2_cat = data_dict["cf2_cat"]
            cf2_deposit_hist = data_dict["cf2_deposit_hist"]
            cf2_deposit_hist_mask = data_dict["cf2_deposit_hist_mask"]
            hom_num = data_dict["hom_num"]
            hom_cat = data_dict["hom_cat"]

            outputs = self.feature_gen_model(
                cf2_idx,
                hom_idx,
                cf2_num,
                cf2_cat,
                cf2_deposit_hist,
                cf2_deposit_hist_mask,
                hom_num,
                hom_cat,
            )
            outputs = outputs.view(outputs.size(1))
            return outputs

    def create_feature(
        self,
        inquiry_date: datetime,
        annual_income: int,
        cf2: pd.DataFrame,
        hom: pd.DataFrame,
    ) -> pd.DataFrame:
        clean_cf2 = self.preprocess_cf2(inquiry_date, annual_income, cf2)
        clean_hom = self.preprocess_hom(inquiry_date, annual_income, hom)
        clean_cf2_num = clean_cf2[self.cf2_num_order]
        clean_cf2_cat = clean_cf2[self.cf2_cat_order].to_numpy(dtype=np.int32)
        clean_cf2_deposit = clean_cf2[self.cf2_deposit_col].to_numpy()
        clean_hom_num = clean_hom[self.hom_num_order]
        clean_hom_cat = clean_hom[self.hom_cat_order].to_numpy(dtype=np.int32)
        clean_cf2_num = self.cf2_normalizer.transform(clean_cf2_num)
        if len(clean_hom) > 0:
            clean_hom_num = self.hom_normalizer.transform(clean_hom_num)
        else:
            clean_hom_num = clean_hom_num.to_numpy(dtype=np.float64)
        data = self.model_preprocess(
            clean_cf2_num,
            clean_cf2_cat,
            clean_cf2_deposit,
            clean_hom_num,
            clean_hom_cat,
        )

        feature = self.generate_feature(data).numpy()
        col = [
            f"feature_{i}"
            for i in range(
                FINAL_FEATURE_SHIFT, FINAL_FEATURE_LENGTH + FINAL_FEATURE_SHIFT
            )
        ]
        feature_df = pd.DataFrame([feature], columns=col)
        return feature_df