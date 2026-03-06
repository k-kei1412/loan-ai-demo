import pandas as pd

def preprocess(df):
    df = df.copy()

    # idは意味がないので削除
    if "id" in df.columns:
        df = df.drop("id", axis=1)

    # 数値型
    numeric_cols = [
        "GrossApproval",
        "SBAGuaranteedApproval",
        "ApprovalFiscalYear",
        "InitialInterestRate",
        "TermInMonths",
        "NaicsSector",
        "CongressionalDistrict",
        "JobsSupported"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # カテゴリ
    cat_cols = [
        "Subprogram",
        "FixedOrVariableInterestInd",
        "BusinessType",
        "BusinessAge",
        "RevolverStatus",
        "CollateralInd"
    ]

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df, cat_cols
