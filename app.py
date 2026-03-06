# ======================
# 3. 予測実行
# ======================
if submit:
    # 1. データの作成（RevolverStatus を数値に変換するための準備）
    # もしモデルが RevolverStatus を数値(Float)として期待しているなら、
    # Y -> 1, N -> 0 のように変換して渡す必要があります。
    
    revolver_numeric = 1.0 if revolver == "Y" else 0.0

    input_data = {
        "GrossApproval": float(gross),
        "SBAGuaranteedApproval": float(sba),
        "ApprovalFiscalYear": float(year),
        "Subprogram": str(subprogram),
        "InitialInterestRate": float(rate),
        "FixedOrVariableInterestInd": str(rate_type),
        "TermInMonths": float(term),
        "NaicsSector": float(sector),
        "CongressionalDistrict": float(district),
        "BusinessType": str(business_type),
        "BusinessAge": str(business_age),
        "RevolverStatus": revolver_numeric,  # ここを文字列から数値(float)に変更！
        "JobsSupported": float(jobs),
        "CollateralInd": str(collateral)
    }

    input_df = pd.DataFrame([input_data])

    # 2. 列の順序をモデルに合わせる
    input_df = input_df.reindex(columns=expected_features)

    # 3. 型の最終確認と強制変換
    # 今回のエラーに基づき、RevolverStatus も数値列リストに含めます
    numeric_cols = [
        "GrossApproval", "SBAGuaranteedApproval", "ApprovalFiscalYear",
        "InitialInterestRate", "TermInMonths", "NaicsSector", 
        "CongressionalDistrict", "JobsSupported", "RevolverStatus" # 追加
    ]

    for col in input_df.columns:
        if col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').astype(float)
        else:
            input_df[col] = input_df[col].astype(str)

    # 4. カテゴリ変数のインデックスを取得
    cat_features_idx = [i for i, col in enumerate(input_df.columns) if input_df[col].dtype == 'object']

    try:
        # Poolの作成
        pool = Pool(input_df, cat_features=cat_features_idx)
        
        # 予測
        proba = model.predict_proba(pool)[0][1]

        # 結果表示
        st.markdown("---")
        st.subheader("審査結果")
        st.metric("デフォルト確率", f"{round(proba * 100, 2)} %")

        if proba > 0.5:
            st.error("【判定】融資危険")
        else:
            st.success("【判定】融資可能")

    except Exception as e:
        st.error(f"予測中にエラーが発生しました。")
        st.write(f"エラー詳細: {e}")
