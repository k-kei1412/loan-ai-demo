import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier, Pool

# ======================
# タイトル
# ======================

st.title("ローン返済リスク予測AI")

st.write("企業のローン返済リスクをAIで予測します")

# ======================
# モデル読み込み
# ======================

model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

# ======================
# 入力フォーム
# ======================

gross_approval = st.number_input("融資額 (GrossApproval)", value=50000)

sba_guaranteed = st.number_input(
    "政府保証額 (SBAGuaranteedApproval)",
    value=30000
)

approval_year = st.number_input(
    "承認年度 (ApprovalFiscalYear)",
    value=2010
)

subprogram = st.selectbox(
    "ローンプログラム",
    [
        "7(a)",
        "FA$TRK (Small Loan Express)",
        "Express",
        "Community Advantage"
    ]
)

interest_rate = st.number_input(
    "金利 (InitialInterestRate)",
    value=5.0
)

interest_type = st.selectbox(
    "金利タイプ",
    ["Fixed", "Variable"]
)

term_months = st.number_input(
    "返済期間(月)",
    value=120
)

naics_sector = st.number_input(
    "産業セクター (NaicsSector)",
    value=44
)

district = st.number_input(
    "議会地区 (CongressionalDistrict)",
    value=10
)

business_type = st.selectbox(
    "企業タイプ",
    [
        "Corporation",
        "Partnership",
        "Sole Proprietorship",
        "LLC"
    ]
)

business_age = st.selectbox(
    "企業年齢",
    [
        "Startup",
        "Existing"
    ]
)

revolver_status = st.selectbox(
    "リボルビングローン",
    [
        "Y",
        "N"
    ]
)

jobs_supported = st.number_input(
    "雇用人数",
    value=5
)

collateral = st.selectbox(
    "担保あり",
    [
        "Y",
        "N"
    ]
)

# ======================
# 予測ボタン
# ======================

if st.button("予測する"):

    data = pd.DataFrame(
        [[
            gross_approval,
            sba_guaranteed,
            approval_year,
            subprogram,
            interest_rate,
            interest_type,
            term_months,
            naics_sector,
            district,
            business_type,
            business_age,
            revolver_status,
            jobs_supported,
            collateral
        ]],
        columns=[
            "GrossApproval",
            "SBAGuaranteedApproval",
            "ApprovalFiscalYear",
            "Subprogram",
            "InitialInterestRate",
            "FixedOrVariableInterestInd",
            "TermInMonths",
            "NaicsSector",
            "CongressionalDistrict",
            "BusinessType",
            "BusinessAge",
            "RevolverStatus",
            "JobsSupported",
            "CollateralInd"
        ]
    )

    # ======================
    # カテゴリ変数
    # ======================

    cat_features = [
        "Subprogram",
        "FixedOrVariableInterestInd",
        "BusinessType",
        "BusinessAge",
        "RevolverStatus",
        "CollateralInd"
    ]

    pool = Pool(data, cat_features=cat_features)

    # ======================
    # 予測
    # ======================

    proba = model.predict_proba(pool)[0][1]

    st.subheader("予測結果")

    st.write("デフォルト確率:", round(proba * 100, 2), "%")

    if proba > 0.5:
        st.error("デフォルトリスクが高い可能性があります")
    else:
        st.success("デフォルトリスクは低いと予測されます")
