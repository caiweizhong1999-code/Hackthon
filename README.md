# Hackthon
clinical_explorer/
├── app.py                  # Streamlit 主入口(多页面)
├── pages/
│   ├── 1_Cohort_Explorer.py
│   ├── 2_Visualizations.py
│   └── 3_Risk_Prediction.py
├── src/
│   ├── data_loader.py      # CSV 上传、校验、变量类型识别
│   ├── nl_parser.py        # 自然语言 → 结构化查询(LLM)
│   ├── query_engine.py     # 结构化查询 → Pandas 筛选/分组
│   ├── plots.py            # KM、分布、趋势图
│   ├── survival.py         # KM + log-rank
│   └── model.py            # XGBoost 训练、预测、SHAP
├── data/                   # 示例数据
├── models/                 # 训练好的 model.joblib
├── train_model.py          # 离线训练脚本
├── requirements.txt
└── README.md
