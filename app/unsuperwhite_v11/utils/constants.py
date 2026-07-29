import os

POSTAL_TO_PREFECTURE_PATH = os.path.join(
    "static", "models", "unsuperwhite_v11", "prefecture", "postal_to_prefecture.csv"
)
PREFECTURE_TO_DEFAULT_PATH = os.path.join(
    "static", "models", "unsuperwhite_v11", "prefecture", "prefecture_default_ratio.csv"
)

CF2_NORMALIZER_PATH = os.path.join(
    "static", "models", "unsuperwhite_v11", "scaler", "cf2_minmax_scaler.pkl"
)
HOM_NORMALIZER_PATH = os.path.join(
    "static", "models", "unsuperwhite_v11", "scaler", "hom_minmax_scaler.pkl"
)
FEATURE_GEN_MODEL_PATH = os.path.join(
    "static",
    "models",
    "unsuperwhite_v11",
    "feature_gen",
    "first_and_second_train_shuffle_32_noundersampling_new_architect_hom_random42_patience5_finaldim128_23_reduced.pth",
)

LGBM_PREDICTION_MODEL_PATH = os.path.join(
    "static",
    "models",
    "unsuperwhite_v11",
    "prediction_model",
    "lgbm_20260406.txt",
)
CAT_PREDICTION_MODEL_PATH = os.path.join(
    "static", "models", "unsuperwhite_v11", "prediction_model", "meow_20260403.cbm"
)

BETA_CALIBRATOR_PATH = os.path.join(
    "static", "models", "unsuperwhite_v11", "prediction_model", "beta_calibrator.pkl"
)

RANKING_THRESHOLD_PATH = os.path.join(
    "static", "models", "unsuperwhite_v11", "ranking.csv"
)

FEATURE_GEN_MODEL_ARGS = {
    "cf2_num_input_dim": 8,
    "hom_num_input_dim": 3,
    "base_num_input_dim": 2,
    "deposit_cat_num": 5,
    "deposit_embedding_dim": 4,
    "num_cat": [9, 9, 9, 6, 3],
    "embedding_dim": [8, 8, 8, 4, 2],
    "app_rnn_length": 10,
    "hidden_size": [16, 64],
    "num_layers": [2, 2],
    "cf2_final_dim": 128,
}

# member industry classification
VALID_MEMBER_INDUSTRY_CLASSIFICATION = [10, 20, 30, 40, 50, 60, 70, 80]

# contract management classification
VALID_CONTRACT_MANAGEMENT_CLASSIFICATION = [1, 2, 3, 4, 5, 6, 7, 21, 99]

# guarantor classification
VALID_GUARANTOR_CLASSIFICATION = [1, 2, 9]

# product code dict
VALID_PRODUCT_CAT = ["Q", "", "S", "U", "A", "R", "C", "Y", "Other"]
PRODUCT_CODE_CAT_TO_NUM = {
    "Q": 0,
    "": 1,
    "S": 2,
    "U": 3,
    "A": 4,
    "R": 5,
    "C": 6,
    "Y": 7,
    "Other": 8,
}

# ending comment dict
ENDING_COMMENT_CAT_TO_NUM = {0: 0, 71: 1, 73: 2, 74: 3, 75: 4, 70: 5}

# deposit history dict
PAYMENT_HISTORY_ENCODING_DICT = {
    "$": "0",
    "P": "1",
    "R": "2",
    "A": "3",
    "-": "4",
    "+": "5",
}

# inquiry type
VALID_INQUIRY_TYPE = [1, 2, 3, 4, 5, 6, 7]

CF2_NORMALIZER_COLUMN_MAP = {
    "number_of_payments": "Number of payments",
    "contract_amount": "Contract amount (next cash advance bill amount)",
    "outstanding_debt_amount": "Remaining debt amount",
    "cumulative_payment_amount": "Cumulative deposit amount",
    "information_type": "Information type (progress comments)",
    "additional_comment_n": "Additional comments (N)",
    "estimate_loan_age": "Estimate loan age",
    "estimate_report_age": "Estimate report age",
}

HOM_NORMALIZER_COLUMN_MAP = {
    "application_amount": "Application amount",
    "number_of_payments": "Number of payments",
    "estimate_application_age": "Estimate application age",
}

ENGINEERED_FEATURE_COL_ORDER = [
    "age",
    "annual_income",
    "income_age",
    "avg_default3month",
    "avg_pay3month",
    "max_default3month",
    "avg_default6month",
    "avg_pay6month",
    "max_default6month",
    "sum_default3month",
    "sum_pay3month",
    "sum_default6month",
    "sum_pay6month",
    "avg_default3month_weighted",
    "avg_pay3month_weighted",
    "avg_default6month_weighted",
    "avg_pay6month_weighted",
    "avg_default9month",
    "avg_pay9month",
    "max_default9month",
    "avg_default12month",
    "avg_pay12month",
    "max_default12month",
    "sum_default9month",
    "sum_pay9month",
    "sum_default12month",
    "sum_pay12month",
    "avg_default9month_weighted",
    "avg_pay9month_weighted",
    "avg_default12month_weighted",
    "avg_pay12month_weighted",
    "remainingDebt_sum",
    "from_contractDate",
    "NC_3month",
    "NC_6month",
    "NC_9month",
    "NC_12month",
    "percentPayment_max_24month",
    "activeAcct",
    "badAcct",
    "active_badAcct",
    "passingBadAcct",
    "annualBilling_sum",
    "billingIncome_ratio",
    "active3Month",
    "active6Month",
    "active9Month",
    "active12Month",
    "contractDuration_avg",
    "contractDuration_max",
    "bankkruptcy",
    "sum_balance",
    "sum_cashAdvanceBalance",
    "avg_balance",
    "avg_cashAdvanceBalance",
    "all_loan",
    "loanTimes",
    "completeRP_times",
    "completeRepay_rate",
    "From_latestLoan",
    "From_latestPayment",
    "extreme_annualIncome_avg",
    "balance_annualIncome_avg",
    "extreme_annualIncome_max",
    "balance_annualIncome_max",
    "extreme_annualIncome_min",
    "balance_annualIncome_min",
    "applyingTimes",
    "auto_apply",
    "Q_apply",
    "NaN_apply",
    "allContract",
    "maxDeposit",
    "R04_count",
    "Q02_count",
    "QXX_count",
    "avg_balancePerLimit",
    "max_balancePerLimit",
    "retailWithBalance",
    "defaultRecently_3month",
    "defaultRecently_6month",
    "defaultRecently_9month",
    "defaultRecently_12month",
    "activeRatio_3month",
    "activeRatio_6month",
    "activeRatio_9month",
    "activeRatio_12month",
    "remainingDebt_annualIncome",
    "allContract_applying",
    "house_apply",
    "complete_allContract",
    "activeBad_active",
    "sumBalance_income",
    "loanTimes_allContract",
    "contractDuration_max_all_loan",
    "From_latestPayment_passingBadAcct",
    "Q_apply_applyingTimes",
    "auto_apply_applyingTimes",
    "NaN_apply_applyingTimes",
    "house_apply_applyingTimes",
    "prefecture_default_ratio",
    "NaN_count",
]

# define magic number/text

RANK = ["A-500", "B-99", "B-50", "C-3", "X"]

PAYMENT_HISTORY_LENGTH = 24

FINAL_FEATURE_LENGTH = 192
FINAL_FEATURE_SHIFT = 0

INFORMATION_TYPE_DEFAULT = ["10", "11", "12", "13"]

ENDING_COMMENT_COMPLETE = [71, 71.0]

MONTH_INTERVAL = [3, 6, 9, 12]

MIN = 1e-9
