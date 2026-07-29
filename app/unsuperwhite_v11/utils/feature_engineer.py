import pandas as pd
import numpy as np
from datetime import datetime
from .constants import (
    POSTAL_TO_PREFECTURE_PATH,
    PREFECTURE_TO_DEFAULT_PATH,
    ENGINEERED_FEATURE_COL_ORDER,
    INFORMATION_TYPE_DEFAULT,
    ENDING_COMMENT_COMPLETE,
    MONTH_INTERVAL,
)


class FeatureEngineer:
    def __init__(
        self,
        postal_to_prefecture_postal_col="postal_code",
        postal_to_prefecture_prefecture_col="prefecture",
        prefecture_to_default_prefecture_col="prefecture",
        prefecture_to_default_default_col="default12",
    ):
        self.postal_to_prefecture_dict = {}
        self.prefecture_to_default_dict = {}
        postal_to_prefecture_df = pd.read_csv(POSTAL_TO_PREFECTURE_PATH)
        self.postal_to_prefecture_dict = postal_to_prefecture_df.set_index(
            postal_to_prefecture_postal_col
        )[postal_to_prefecture_prefecture_col].to_dict()

        prefecture_to_default_df = pd.read_csv(PREFECTURE_TO_DEFAULT_PATH)
        self.prefecture_to_default_dict = prefecture_to_default_df.set_index(
            prefecture_to_default_prefecture_col
        )[prefecture_to_default_default_col].to_dict()

    def weighted_mean(self, values_series, weights_series):
        weighted_values = values_series * weights_series
        sum_weighted_values = weighted_values.sum()
        sum_weights = weights_series.sum()
        if sum_weights == 0:
            return np.nan
        return sum_weighted_values / sum_weights

    def full_cf2_feature_engineering(
        self, inquiry_date: datetime, annual_income: int, df_cf2: pd.DataFrame
    ):
        processed_df = pd.DataFrame()
        out_df = pd.DataFrame()

        processed_df["outstanding_debt_amount"] = df_cf2[
            "outstanding_debt_amount"
        ].replace([-999999.00, -444444.00], np.nan)

        processed_df["contract_date"] = pd.to_datetime(df_cf2["contract_date"])

        processed_df["from_contractDate"] = (
            (inquiry_date.year - processed_df["contract_date"].dt.year) * 365
            + (inquiry_date.month - processed_df["contract_date"].dt.month) * 30
            + (inquiry_date.day - processed_df["contract_date"].dt.day)
        )

        if len(processed_df) > 0:
            out_df["remainingDebt_sum"] = [
                processed_df["outstanding_debt_amount"].sum()
            ]
        else:
            out_df["remainingDebt_sum"] = [np.nan]

        out_df["from_contractDate"] = [processed_df["from_contractDate"].min()]

        processed_df["payment_history"] = df_cf2["payment_history"].apply(
            lambda x: x if pd.isna(x) else x.strip()
        )
        processed_df["numberOfdeposit"] = processed_df["payment_history"].apply(
            lambda x: 0 if pd.isna(x) else len(x)
        )
        processed_df["numberOfpay"] = processed_df["payment_history"].apply(
            lambda x: 0 if pd.isna(x) else x.count("$")
        )
        processed_df["percentPayment_24month"] = (
            processed_df["numberOfpay"] / processed_df["numberOfdeposit"]
        )

        out_df["percentPayment_max_24month"] = [
            processed_df["percentPayment_24month"].max()
        ]
        out_df["activeAcct"] = [
            (
                (processed_df["outstanding_debt_amount"] != 0)
                & (~processed_df["outstanding_debt_amount"].isna())
            ).sum()
        ]

        out_df["badAcct"] = [
            (df_cf2["information_type"].isin(INFORMATION_TYPE_DEFAULT)).sum()
        ]
        out_df["active_badAcct"] = [
            (
                (df_cf2["information_type"].isin(INFORMATION_TYPE_DEFAULT))
                & (df_cf2["outstanding_debt_amount"] != 0)
            ).sum()
        ]
        out_df["passingBadAcct"] = [
            processed_df[(df_cf2["information_type"].isin(INFORMATION_TYPE_DEFAULT))][
                "from_contractDate"
            ].min()
        ]
        out_df["annualBilling_sum"] = [
            df_cf2[processed_df["outstanding_debt_amount"] > 0]["annual_billing_amount"]
            .replace([-999999.0, -444444.0], np.nan)
            .sum()
        ]
        if len(df_cf2[processed_df["outstanding_debt_amount"] > 0]) == 0:
            out_df["annualBilling_sum"] = np.nan
        out_df["billingIncome_ratio"] = (
            (out_df["annualBilling_sum"].values[0]) * 1000 / annual_income
        )
        is_default_ending_comment = df_cf2["information_type"].isin(
            INFORMATION_TYPE_DEFAULT
        )

        processed_df["from_contractDate_m"] = (
            inquiry_date.year - processed_df["contract_date"].dt.year
        ) * 12 + (inquiry_date.month - processed_df["contract_date"].dt.month)

        has_remaining_debt = (processed_df["outstanding_debt_amount"] > 0) & ~(
            processed_df["outstanding_debt_amount"].isna()
        )

        processed_df["reporting_date"] = pd.to_datetime(df_cf2["reporting_date"])
        processed_df["contractDuration_m"] = (
            processed_df["reporting_date"].dt.year
            - processed_df["contract_date"].dt.year
        ) * 12 + (
            processed_df["reporting_date"].dt.month
            - processed_df["contract_date"].dt.month
        )
        if len(processed_df.dropna(subset=["contractDuration_m"])) > 0:
            out_df["contractDuration_avg"] = [
                processed_df["contractDuration_m"].sum()
                / len(processed_df.dropna(subset=["contractDuration_m"]))
            ]
            out_df["contractDuration_max"] = [processed_df["contractDuration_m"].max()]
        else:
            out_df["contractDuration_avg"] = [np.nan]
            out_df["contractDuration_max"] = [np.nan]
        processed_df["ending_comment"] = df_cf2["ending_comment"]
        out_df["bankkruptcy"] = [
            ((processed_df["ending_comment"] == 75).sum() > 0).sum()
        ]
        out_df["allContract"] = [len(processed_df)]
        out_df["maxDeposit"] = [
            df_cf2["payment_amount"].replace([-444444, -999999], np.nan).max()
        ]
        processed_df["product_code"] = df_cf2["product_code"]
        processed_df["product_code_1"] = df_cf2["product_code_1"]
        out_df["R04_count"] = [(processed_df["product_code"] == "R04").sum()]
        out_df["Q02_count"] = [
            (
                (processed_df["product_code"] == "Q02")
                | (processed_df["product_code_1"] == "S02")
            ).sum()
        ]
        out_df["QXX_count"] = [
            (
                processed_df["product_code"].apply(
                    lambda x: False if pd.isna(x) else "Q" in x
                )
            ).sum()
        ]

        processed_df["credit_limit"] = df_cf2["credit_limit"].replace(
            [-999999.00, -444444.00], np.nan
        )
        processed_df["balancePerLimit"] = (
            processed_df["outstanding_debt_amount"] / processed_df["credit_limit"]
        )
        processed_df["balancePerLimit"] = processed_df["balancePerLimit"].replace(
            np.inf, np.nan
        )

        out_df["avg_balancePerLimit"] = [processed_df["balancePerLimit"].mean()]
        out_df["max_balancePerLimit"] = [processed_df["balancePerLimit"].max()]
        out_df["retailWithBalance"] = [
            (
                (
                    (processed_df["product_code"] == "Q02")
                    | (processed_df["product_code_1"] == "S02")
                )
                & (processed_df["outstanding_debt_amount"] > 0)
            ).sum()
        ]

        out_df["completeAmount"] = [
            sum(processed_df["ending_comment"].isin(ENDING_COMMENT_COMPLETE))
        ]

        processed_df["weightContractDate"] = 1 / processed_df["from_contractDate"]

        active_old_record = processed_df[
            (processed_df["from_contractDate"] != 0)
            & (processed_df["outstanding_debt_amount"] > 0)
        ].copy()
        active_record = processed_df[processed_df["outstanding_debt_amount"] > 0].copy()

        has_active_record = len(active_record) > 0

        for r in MONTH_INTERVAL:

            active_record[f"$_{r}month"] = active_record["payment_history"].apply(
                lambda x: np.nan if pd.isna(x) else x[0:r].count("$")
            )

            active_record[f"A_{r}month"] = active_record["payment_history"].apply(
                lambda x: np.nan if pd.isna(x) else x[0:r].count("A")
            )

            active_old_record[f"$_{r}month"] = active_old_record[
                "payment_history"
            ].apply(lambda x: np.nan if pd.isna(x) else x[0:r].count("$"))

            active_old_record[f"A_{r}month"] = active_old_record[
                "payment_history"
            ].apply(lambda x: np.nan if pd.isna(x) else x[0:r].count("A"))

            if has_active_record:
                avg_default = active_record[f"A_{r}month"].mean()
                avg_pay = active_record[f"$_{r}month"].mean()
                max_default = active_record[f"A_{r}month"].max()
                out_df[f"avg_default{r}month"] = [avg_default]
                out_df[f"avg_pay{r}month"] = [avg_pay]
                out_df[f"max_default{r}month"] = [max_default]

                sum_default = active_record[f"A_{r}month"].sum()
                sum_pay = active_record[f"$_{r}month"].sum()
                out_df[f"sum_default{r}month"] = [sum_default]
                out_df[f"sum_pay{r}month"] = [sum_pay]

                avg_default2 = self.weighted_mean(
                    active_old_record[f"A_{r}month"],
                    active_old_record["weightContractDate"],
                )
                avg_pay2 = self.weighted_mean(
                    active_old_record[f"$_{r}month"],
                    active_old_record["weightContractDate"],
                )
                out_df[f"avg_default{r}month_weighted"] = [avg_default2]
                out_df[f"avg_pay{r}month_weighted"] = [avg_pay2]
            else:
                out_df[f"avg_default{r}month"] = [np.nan]
                out_df[f"avg_pay{r}month"] = [np.nan]
                out_df[f"max_default{r}month"] = [np.nan]
                out_df[f"sum_default{r}month"] = [np.nan]
                out_df[f"sum_pay{r}month"] = [np.nan]
                out_df[f"avg_default{r}month_weighted"] = [np.nan]
                out_df[f"avg_pay{r}month_weighted"] = [np.nan]

            out_df[f"NC_{r}month"] = [
                (processed_df["from_contractDate"] / 30 <= r).sum()
            ]

            out_df[f"defaultRecently_{r}month"] = [
                (
                    (processed_df["from_contractDate"] / 30 < r)
                    & is_default_ending_comment
                ).sum()
            ]

            out_df[f"active{r}Month"] = [
                (has_remaining_debt & (processed_df["from_contractDate_m"] < r)).sum()
            ]

        col_replace_with_nan = [
            "NC_3month",
            "NC_6month",
            "NC_9month",
            "NC_12month",
            "activeAcct",
            "badAcct",
            "active_badAcct",
            "passingBadAcct",
            "active3Month",
            "active6Month",
            "active9Month",
            "active12Month",
            "R04_count",
            "Q02_count",
            "QXX_count",
            "retailWithBalance",
            "allContract",
            "completeAmount",
        ]

        out_df[col_replace_with_nan] = out_df[col_replace_with_nan].replace(0, np.nan)

        if len(processed_df) > 0 and pd.isna(processed_df["from_contractDate"].min()):
            out_df["from_contractDate"] = [-999999999]

        return out_df

    def full_ff_feature_engineering(self, inquiry_date: datetime, annual_income, df_ff):

        processed_df = pd.DataFrame()
        out_df = pd.DataFrame()

        processed_df[
            ["balance", "cash_advance_balance", "credit_limit", "contract_amount"]
        ] = df_ff[
            ["balance", "cash_advance_balance", "credit_limit", "contract_amount"]
        ].replace(
            [-999999999, -777777777, -111111111, -444444444], np.nan
        )

        if len(df_ff) == 0:
            out_df["sum_balance"] = [np.nan]
            out_df["sum_cashAdvanceBalance"] = [np.nan]
        else:
            out_df["sum_balance"] = [processed_df["balance"].sum()]
            out_df["sum_cashAdvanceBalance"] = [
                processed_df["cash_advance_balance"].sum()
            ]

        out_df["avg_balance"] = [processed_df["balance"].mean()]
        out_df["avg_cashAdvanceBalance"] = [processed_df["cash_advance_balance"].mean()]

        out_df["all_loan"] = out_df["sum_balance"] + out_df["sum_cashAdvanceBalance"]

        out_df["loanTimes"] = [len(df_ff)]

        out_df["completeRP_times"] = [
            (
                (processed_df["balance"].fillna(0.0) == 0)
                & (processed_df["cash_advance_balance"].fillna(0.0) == 0)
            ).sum()
        ]
        out_df["completeRepay_rate"] = out_df["completeRP_times"] / out_df["loanTimes"]
        if len(df_ff) == 0:
            out_df["completeRP_times"] = [np.nan]
            out_df["completeRepay_rate"] = [np.nan]

        processed_df["loan_date"] = pd.to_datetime(df_ff["loan_date"])
        processed_df["From_latestLoan"] = (
            (inquiry_date.year - processed_df["loan_date"].dt.year) * 365
            + (inquiry_date.month - processed_df["loan_date"].dt.month) * 30
            + (inquiry_date.day - processed_df["loan_date"].dt.day)
        )
        out_df["From_latestLoan"] = [processed_df["From_latestLoan"].min()]

        processed_df["latest_payment_date"] = pd.to_datetime(
            df_ff["latest_payment_date"]
        )
        processed_df["From_latestPayment"] = (
            (inquiry_date.year - processed_df["latest_payment_date"].dt.year) * 365
            + (inquiry_date.month - processed_df["latest_payment_date"].dt.month) * 30
            + (inquiry_date.day - processed_df["latest_payment_date"].dt.day)
        )
        out_df["From_latestPayment"] = [processed_df["From_latestPayment"].min()]

        processed_df["credit_limit"] = processed_df["credit_limit"].fillna(
            processed_df["contract_amount"]
        )

        processed_df["extreme_annualIncome"] = (
            processed_df["credit_limit"] / annual_income
        )
        processed_df["balance_annualIncome"] = processed_df["balance"] / annual_income

        out_df["extreme_annualIncome_avg"] = [
            processed_df["extreme_annualIncome"].mean()
        ]
        out_df["extreme_annualIncome_max"] = [
            processed_df["extreme_annualIncome"].max()
        ]
        out_df["extreme_annualIncome_min"] = [
            processed_df["extreme_annualIncome"].min()
        ]
        out_df["balance_annualIncome_avg"] = [
            processed_df["balance_annualIncome"].mean()
        ]
        out_df["balance_annualIncome_max"] = [
            processed_df["balance_annualIncome"].max()
        ]
        out_df["balance_annualIncome_min"] = [
            processed_df["balance_annualIncome"].min()
        ]

        col_replace_with_nan = [
            "loanTimes",
        ]

        out_df[col_replace_with_nan] = out_df[col_replace_with_nan].replace(0, np.nan)

        return out_df

    def full_hom_feature_engineering(self, df_hom):
        out_df = pd.DataFrame()
        out_df["applyingTimes"] = [len(df_hom)]
        out_df["auto_apply"] = [
            (
                df_hom["product_code_1"].apply(
                    lambda x: False if pd.isna(x) else "R04" in x
                )
            ).sum()
        ]
        out_df["house_apply"] = [
            (
                df_hom["product_code_1"].apply(
                    lambda x: False if pd.isna(x) else "S04" in x
                )
            ).sum()
        ]
        out_df["Q_apply"] = [
            (
                df_hom["product_code_1"].apply(
                    lambda x: False if pd.isna(x) else "Q" in x
                )
            ).sum()
        ]
        out_df["NaN_apply"] = [(df_hom["product_code_1"].isna()).sum()]

        col_replace_with_nan = [
            "applyingTimes",
            "auto_apply",
            "house_apply",
            "Q_apply",
            "NaN_apply",
        ]

        out_df[col_replace_with_nan] = out_df[col_replace_with_nan].replace(0, np.nan)

        return out_df

    def added_feature_engineering(self, merge_data):
        merge_data["income_age"] = merge_data["annual_income"] / merge_data["age"]

        merge_data["activeRatio_3month"] = (
            merge_data["active3Month"] / merge_data["NC_3month"]
        )
        merge_data["activeRatio_6month"] = (
            merge_data["active6Month"] / merge_data["NC_6month"]
        )
        merge_data["activeRatio_9month"] = (
            merge_data["active9Month"] / merge_data["NC_9month"]
        )
        merge_data["activeRatio_12month"] = (
            merge_data["active12Month"] / merge_data["NC_12month"]
        )

        merge_data["remainingDebt_annualIncome"] = (
            merge_data["remainingDebt_sum"] / merge_data["annual_income"]
        )
        merge_data["allContract_applying"] = (
            merge_data["allContract"] / merge_data["applyingTimes"]
        )
        merge_data["complete_allContract"] = (
            merge_data["completeAmount"] / merge_data["allContract"]
        )

        merge_data["activeBad_active"] = (
            merge_data["active_badAcct"] / merge_data["activeAcct"]
        )
        merge_data["sumBalance_income"] = (
            merge_data["sum_balance"] / merge_data["annual_income"]
        )
        merge_data["loanTimes_allContract"] = (
            merge_data["loanTimes"] / merge_data["allContract"]
        )
        merge_data["contractDuration_max_all_loan"] = (
            merge_data["all_loan"] / merge_data["contractDuration_max"]
        )
        merge_data["From_latestPayment_passingBadAcct"] = (
            merge_data["From_latestPayment"] / merge_data["passingBadAcct"]
        )

        merge_data["Q_apply_applyingTimes"] = (
            merge_data["Q_apply"] / merge_data["applyingTimes"]
        )
        merge_data["auto_apply_applyingTimes"] = (
            merge_data["auto_apply"] / merge_data["applyingTimes"]
        )
        merge_data["NaN_apply_applyingTimes"] = (
            merge_data["NaN_apply"] / merge_data["applyingTimes"]
        )
        merge_data["house_apply_applyingTimes"] = (
            merge_data["house_apply"] / merge_data["applyingTimes"]
        )

        merge_data["defaultRecently_3month"] = merge_data[
            "defaultRecently_3month"
        ].fillna(0.0)
        merge_data["defaultRecently_6month"] = merge_data[
            "defaultRecently_6month"
        ].fillna(0.0)
        merge_data["defaultRecently_9month"] = merge_data[
            "defaultRecently_9month"
        ].fillna(0.0)
        merge_data["defaultRecently_12month"] = merge_data[
            "defaultRecently_12month"
        ].fillna(0.0)
        merge_data["from_contractDate"] = merge_data["from_contractDate"].fillna(
            -999999999.0
        )

        return merge_data

    def get_latest_prefecture(
        self,
        cf2,
        hom,
        cf2_contract_date_col_name="contract_date",
        cf2_postal_code_col_name="postal_code",
        hom_application_date_col_name="inquiry_date",
        hom_postal_code_col_name="postal_code",
    ):
        full_df = cf2[[cf2_contract_date_col_name, cf2_postal_code_col_name]].rename(
            columns={
                cf2_contract_date_col_name: "date",
                cf2_postal_code_col_name: "postal_code",
            }
        )

        if len(hom) > 0:
            full_df = pd.concat(
                [
                    full_df,
                    hom[
                        [hom_application_date_col_name, hom_postal_code_col_name]
                    ].rename(
                        columns={
                            hom_application_date_col_name: "date",
                            hom_postal_code_col_name: "postal_code",
                        }
                    ),
                ],
                ignore_index=True,
            )

        full_df["postal_code"] = full_df["postal_code"].apply(
            lambda x: x if ((not pd.isna(x)) or (len(x).strip() == 0)) else "other"
        )
        full_date_postal_unique = full_df.sort_values(
            by=["date", "postal_code"], ascending=[False, True]
        ).reset_index(drop=True)["postal_code"]
        for p in full_date_postal_unique:
            if p in self.postal_to_prefecture_dict:
                return self.postal_to_prefecture_dict[p]
        return np.nan

    def get_prefecture_default(
        self,
        cf2_key,
        hom_key,
        **kwargs,
    ):
        prefecture = self.get_latest_prefecture(cf2_key, hom_key, **kwargs)
        if prefecture in self.prefecture_to_default_dict.keys():
            return self.prefecture_to_default_dict[prefecture]
        return self.prefecture_to_default_dict["other"]

    def create_feature(
        self,
        inquiry_date: datetime,
        age: int,
        annual_income: int,
        cf2_df: pd.DataFrame,
        ff_df: pd.DataFrame,
        hom_df: pd.DataFrame,
    ):
        cf2 = (
            self.full_cf2_feature_engineering(inquiry_date, annual_income, cf2_df)
            .reset_index(drop=True)
            .astype(np.float64)
        )
        ff = (
            self.full_ff_feature_engineering(inquiry_date, annual_income, ff_df)
            .reset_index(drop=True)
            .astype(np.float64)
        )
        hom = (
            self.full_hom_feature_engineering(hom_df)
            .reset_index(drop=True)
            .astype(np.float64)
        )
        created_data = pd.concat([cf2, ff, hom], axis=1)
        prefecture_default = self.get_prefecture_default(
            cf2_df,
            hom_df,
        )
        created_data["prefecture_default_ratio"] = [prefecture_default]
        created_data["age"] = float(age)
        created_data["annual_income"] = float(annual_income)
        created_data = self.added_feature_engineering(created_data)
        created_data = created_data.drop(columns=["completeAmount"])
        created_data["NaN_count"] = created_data.isna().sum(axis=1)
        created_data = created_data[ENGINEERED_FEATURE_COL_ORDER]
        return created_data
