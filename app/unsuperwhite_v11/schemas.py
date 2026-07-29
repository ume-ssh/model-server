from pydantic import BaseModel, Field, validator, model_validator, field_validator
from typing import List, Union, Optional, Literal, Any, Dict
from datetime import datetime
import pandas as pd
from .utils.constants import RANK

japanese_year = {"R": 2018, "H": 1988, "S": 1925, "T": 1911, "M": 1867}


def parse_date_yyyymmdd(date_str):
    """Parse a date string in %Y%m%d format to a datetime object."""
    if date_str is None:
        return None
    if not isinstance(date_str, str):
        raise ValueError(f"Date must be string, but got {type(date_str)}")
    if len(date_str) != 8:
        raise ValueError(
            f"Date must contain exactly 8 elements, but got {len(date_str)}"
        )
    try:
        return datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        raise ValueError(f"Date must be in the format %Y%m%d, but got {date_str}")


def parse_date_hhmmss(date_str):
    """Parse a date string in %H%M%S format to a datetime object."""
    if date_str is None:
        return None
    if not isinstance(date_str, str):
        raise ValueError(f"Date must be string, but got {type(date_str)}")
    if len(date_str) != 6:
        raise ValueError(
            f"Time must contain exactly 6 elements, but got {len(date_str)}"
        )
    try:
        return datetime.strptime(date_str, "%H%M%S")
    except ValueError:
        raise ValueError(f"Date must be in the format %H%M%S, but got {date_str}")


class BaseData(BaseModel):
    @model_validator(mode="after")
    def check_list_length_equal(self) -> "Payload":
        list_len = -1
        diff_len = set()
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            len_value = len(value)
            if list_len == -1:
                list_len = len_value
                diff_len.add(list_len)
            else:
                if list_len != len_value:
                    diff_len.add(len_value)
        if len(diff_len) > 1:
            raise ValueError(
                f"All lists in the {self.__class__.__name__} must have the same length: found length of {', '.join(sorted([str(e) for e in diff_len]))}"
            )
        return self


class CF2Data(BaseData):
    postal_code: List[int | None] = Field(
        description="Java: cccfa208, 本人要件_郵便番号 | SAS: col10, 郵便番号"
    )
    member_industry_classification: List[int | None] = Field(
        description="Java: cccfc212, 契約内容_会員業種区分 | SAS: col29, 会員業種区分"
    )
    contract_date: List[datetime | None] = Field(
        description="Java: cccfa221, 契約内容_契約年月日 | SAS: col32, 契約年月日 -> in format %Y%m%d, for example '20241023'"
    )
    contract_management_classification: List[int | None] = Field(
        description="Java: cccfc213, 契約内容_契約管理区分 | SAS: col33, 契約管理区分"
    )
    product_code: List[str | None] = Field(
        description="Java: cccfc214, 契約内容_商品コード | SAS: col34, 商品コード"
    )
    number_of_payments: List[float | None] = Field(
        description="Java: cccfa222, 契約内容_支払回数 | SAS: col36, 支払回数"
    )
    contract_amount: List[float | None] = Field(
        description="Java: cccfk201, 契約内容_契約額(次回キャッシング請求額) | SAS: col38, 契約額（次回キャッシング請求額）"
    )
    credit_limit: List[float | None] = Field(description="Java: | SAS: col39, 極度額")
    cash_advance_credit_limit: List[float | None] = Field(
        description="Java: cccfk203, 契約内容_キャッシング極度額 | SAS: col40, キャッシング極度額"
    )
    reporting_date: List[datetime | None] = Field(
        description="Java: cccfa224, 契約内容_報告日 | SAS: col41, 報告日 -> in format %Y%m%d, for example '20241023'"
    )
    outstanding_debt_amount: List[float | None] = Field(
        description="Java: cccfk204, 契約内容_残債額 | SAS: col42, 残債額"
    )
    payment_amount: List[float | None] = Field(
        description="Java: cccfk207, 契約内容_入金額 | SAS: col45, 入金額"
    )
    cumulative_payment_amount: List[float | None] = Field(
        description="Java: cccfk208, 契約内容_入金累計額 | SAS: col46, 入金累計額"
    )
    payment_history: List[str | None] = Field(
        description="Java: cccfa225, 契約内容_入金履歴 | SAS: col47, 入金履歴"
    )
    information_type: List[str | None] = Field(
        description="Java: cccfc216, 契約内容_情報種別（経過コメント） | SAS: col48, 情報種別（経過コメント）"
    )
    ending_comment: List[int | None] = Field(
        description="Java: cccfc217, 契約内容_終了コメント | SAS: col50, 終了コメント"
    )
    guarantor_classification: List[int | None] = Field(
        description="Java: cccfc219, 契約内容_本人保証人区分 | SAS: col55, 本人保証人区分"
    )
    product_code_1: List[str | None] = Field(
        description="Java: cckfc229, 契約内容２＋付加コメント_商品コード１ | SAS: col410, 商品コード1"
    )
    annual_billing_amount: List[float | None] = Field(
        description="Java: cckfk224, 契約内容２＋付加コメント_年間請求予定額 | SAS: col101, 年間請求予定額"
    )
    additional_comment_n: List[str | None] = Field(
        description="Java: cckfn209, 契約内容２＋付加コメント_付加コメント(N) | SAS: col191, 付加コメント（N)"
    )

    @validator("contract_date", "reporting_date", each_item=True, pre=True)
    def validate_contract_date(cls, v):
        return parse_date_yyyymmdd(v)

    @validator("information_type", each_item=True)
    def validate_information_type(cls, v):
        if v is None:
            return None
        has_number = False
        for c in v:
            if "0" < c < "9":
                has_number = True
                break
        if not has_number:
            return v
        else:
            return str(int(float(v)))


class FFData(BaseData):
    contract_amount: List[float | None] = Field(
        description="Java: cfnfk211, 契約額 | SAS: col4, 契約額"
    )
    credit_limit: List[float | None] = Field(
        description="Java: cfnfk212, 極度額 | SAS: col5, 極度額"
    )
    latest_payment_date: List[datetime | None] = Field(
        description="Java: cfnfa230, 最新支払日 | SAS: col8, 最新支払日 -> in format %Y%m%d, for example '20241023'"
    )
    balance: List[float | None] = Field(
        description="Java: cfnfk214, 残高 | SAS: col10, 残高"
    )
    cash_advance_balance: List[float | None] = Field(
        description="Java: cfnfk215, キャッシング残高 | SAS: col11, キャッシング残高"
    )
    loan_date: List[datetime | None] = Field(
        description="Java: cfnfa232, 貸付日 | SAS: col18, 貸付日 -> in format %Y%m%d, for example '20241023'"
    )

    @validator("latest_payment_date", "loan_date", each_item=True, pre=True)
    def validate_loan_date(cls, v):
        return parse_date_yyyymmdd(v)


class HOMData(BaseData):
    postal_code: List[int | None] = Field(
        description="Java: rhofa411, 基本要件_郵便番号 | SAS: col7, 郵便番号"
    )
    inquiry_date: List[datetime | None] = Field(
        description="Java: rhofa417, 申込履歴_照会年月日 | SAS: col16, 照会年月日 -> in format %Y%m%d, for example '20241023'"
    )
    inquiry_time: List[datetime | None] = Field(
        description="Java: rhofa418, 申込履歴_照会時刻 | SAS: col17, 照会時刻 -> in format %H%M%S"
    )
    inquiry_type: List[int | None] = Field(
        description="Java: rhofc406, 申込履歴_照会区分 | SAS: col19, 照会区分"
    )
    product_code_1: List[str | None] = Field(
        description="Java: rhofc407, 申込履歴_商品コード1 | SAS: col20, 商品コード1"
    )
    application_amount: List[float | None] = Field(
        description="Java: rhofk401, 申込履歴_申込額 | SAS: col21, 申込額"
    )
    number_of_payments: List[float | None] = Field(
        description="Java: rhofa419, 申込履歴_支払回数 | SAS: col22, 支払回数"
    )
    member_industry_classification: List[int | None] = Field(
        description="Java: rhofc408, 申込履歴_会員業種区分 | SAS: col23, 会員業種区分"
    )

    @validator("inquiry_date", each_item=True, pre=True)
    def validate_inquiry_date(cls, v):
        return parse_date_yyyymmdd(v)

    @validator("inquiry_time", each_item=True, pre=True)
    def validate_inquiry_time(cls, v):
        return parse_date_hhmmss(v)


class CICData(BaseModel):
    cf2: CF2Data
    ff: FFData
    hom: HOMData


class UnsuperwhiteV11Data(BaseModel):
    inquiry_date: datetime
    age: float
    annual_income: float
    match_data: CICData
    similar_data: CICData

    @validator("inquiry_date", pre=True)
    def parse_inquiry_date(cls, v):
        return parse_date_yyyymmdd(v)

    @field_validator("age", "annual_income")
    def annual_income_must_be_positive(cls, v, info):
        if v <= 0:
            raise ValueError(f"{info.field_name} must be more than 0.")
        return v


class ScoreResponse(BaseModel):
    score: float
    rank: Literal[*RANK]