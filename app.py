import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="ローン審査AI", layout="wide")
st.title("ローンデフォルト予測AI")

# ======================
# 1. リソースの読み込み
# ======================
@st.cache_resource
def load_resources():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    
    # 過去の学習データを読み込む
    try:
        # 学習に使ったCSVファイルを指定してください
        train_df = pd.read_csv("train.csv")
    except:
        st.error("train_data.csv が見つかりません。過去事例を表示できません。")
        train_df = pd.DataFrame()
    
    return model, train_df

model, train_df = load_resources()
expected_features = model.feature_names_

# ======================
# 2. 予測ロジック関数
# ======================
def get_predictions(input_df):
    # 型の強制変換（これまでのデバッグ結果を反映）
    numeric_cols = ["GrossApproval", "SBAGuaranteedApproval", "ApprovalFiscalYear", 
                    "InitialInterestRate", "TermInMonths", "CongressionalDistrict", 
                    "JobsSupported", "RevolverStatus"]
    
    df = input_df.copy()
    for col in df.columns:
        if col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        else:
            df[col] = df[col].astype(str)
            
    cat_features_idx = [i for i, col in enumerate(df.columns) if df[col].dtype == 'object']
    pool = Pool(df, cat_features=cat_features_idx)
    return model.predict_proba(pool)[0][1], df, cat_features_idx

# ======================
# 3. 入力フォーム
# ======================
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    with col1:
        gross = st.number_input("融資額 ($)", 0, 10000000, 50000)
        sba = st.number_input("保証額 ($)", 0, 10000000, 30000)
        year = st.number_input("承認年度", 1990, 2026, 2010)
        rate = st.number_input("金利 (%)", 0.0, 30.0, 5.0)
        term = st.number_input("返済期間 (月)", 1, 360, 120)
        jobs = st.number_input("雇用人数", 0, 1000, 5)
    with col2:
        subprogram = st.text_input("ローンプログラム", "7(a)")
        rate_type = st.selectbox("金利タイプ", ["Fixed", "Variable"])
        sector = st.number_input("産業セクター (NAICS)", 1, 99, 44)
        district = st.number_input("地区コード", 1, 60, 10)
        business_type = st.text_input("企業形態", "CORPORATION")
        business_age = st.selectbox("企業年齢", ["Startup", "Existing"])
        revolver = st.selectbox("リボルビングローン", ["Y", "N"])
        collateral = st.selectbox("担保の有無", ["Y", "N"])
    
    submit = st.form_submit_button("AI審査を開始")

# ======================
# 4. 実行結果
# ======================
if submit:
    # 入力データ作成
    revolver_val = 1.0 if revolver == "Y" else 0.0
    raw_input = {
        "GrossApproval": gross, "SBAGuaranteedApproval": sba, "ApprovalFiscalYear": year,
        "Subprogram": subprogram, "InitialInterestRate": rate, "FixedOrVariableInterestInd": rate_type,
        "TermInMonths": term, "NaicsSector": str(int(sector)), "CongressionalDistrict": district,
        "BusinessType": business_type, "BusinessAge": business_age, "RevolverStatus": revolver_val,
        "JobsSupported": jobs, "CollateralInd": collateral
    }
    input_df = pd.DataFrame([raw_input]).reindex(columns=expected_features)

    # 予測実行
    proba, final_df, cat_idx = get_predictions(input_df)

    # --- ① 確率表示 ---
    st.subheader("審査結果")
    st.metric("AI予測デフォルト確率", f"{proba * 100:.4f} %")

    # --- ② 影響度表示 ---
    st.write("### 💡 AIが判断の決め手とした項目")
    importances = model.get_feature_importance()
    feat_imp_df = pd.DataFrame({'項目': expected_features, '影響度': importances}).sort_values(by='影響度', ascending=False).head(10)
    st.table(feat_imp_df.T)

    # --- ③ 類似事例10件の検索と比較 ---
    if not train_df.empty:
        st.write("### 📂 過去の類似事例（近い順10件）")
        
        # 距離計算のための数値列抽出
        num_cols = ["GrossApproval", "SBAGuaranteedApproval", "InitialInterestRate", "TermInMonths"]
        train_num = train_df[num_cols].fillna(0)
        input_num = final_df[num_cols].fillna(0)
        
        # KNNで近い事例を探す
        nn = NearestNeighbors(n_neighbors=10)
        nn.fit(train_num)
        distances, indices = nn.kneighbors(input_num)
        
        similar_cases = train_df.iloc[indices[0]].copy()
        
        # 結果を見やすく整形（デフォルトしたかどうかを分かりやすく）
        if 'Default' in similar_cases.columns:
            similar_cases['結果'] = similar_cases['Default'].apply(lambda x: "❌ デフォルト" if x == 1 else "✅ 完済")
        
        st.dataframe(similar_cases)
        
        # 類似事例内でのデフォルト率を計算
        if 'Default' in similar_cases.columns:
            actual_risk = similar_cases['Default'].mean() * 100
            st.info(f"💡 過去の類似事例10件のうち、デフォルトが発生した割合は **{actual_risk:.1f}%** です。")
            st.caption("※AIの予測値とこの実際の割合に乖離がある場合、類似事例の「結果」を優先して判断してください。")
