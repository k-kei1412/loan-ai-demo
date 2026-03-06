予測中にエラーが発生しました。

エラー詳細を確認

CatBoostError: catboost/libs/data/model_dataset_compatibility.cpp:53: モデルではFloatと表示されるが、データセットでは異なるマークが付けられている特徴
トレースバック:
File "/mount/src/loan-ai-demo/app.py", line 94, in <module>
    proba = model.predict_proba(pool)[0][1]
            ~~~~~~~~~~~~~~~~~~~^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/catboost/core.py", line 5653, in predict_proba
    return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/catboost/core.py", line 2929, in _predict
    predictions = self._base_predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose, task_type)
File "/home/adminuser/venv/lib/python3.13/site-packages/catboost/core.py", line 1876, in _base_predict
    return self._object._base_predict(pool, prediction_type, ntree_start, ntree_end, thread_count, verbose, task_type)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "_catboost.pyx", line 5565, in _catboost._CatBoost._base_predict
File "_catboost.pyx", line 5572, in _catboost._CatBoost._base_predict
モデルが期待する特徴量:

[
0:「グロス・プラクソングレーション」
1:「SBAGuaranteedApproval」
2:「承認会計年度」
3:「サブプログラム」
4:「初期金利」
5:「固定または可変興味インド」
6:「タームインマンス」
7:「ナイクス・セクター」
8:「議会選挙区」
9:「ビジネスタイプ」
10:「ビジネスエイジ」
11:「リボルバーステータス」
12:「ジョブズサポートド」
13:「コラテラルインド」
]
送信されたデータの列:

[
0:「グロス・プラクソングレーション」
1:「SBAGuaranteedApproval」
2:「承認会計年度」
3:"Subprogram"
4:"InitialInterestRate"
5:"FixedOrVariableInterestInd"
6:"TermInMonths"
7:"NaicsSector"
8:"CongressionalDistrict"
9:"BusinessType"
10:"BusinessAge"
11:"RevolverStatus"
12:"JobsSupported"
13:"CollateralInd"
]
