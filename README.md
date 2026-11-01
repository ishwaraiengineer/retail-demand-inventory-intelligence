Retail Demand Forecasting & Inventory Optimization

Predict how much of a product will sell, then turn that prediction into a concrete replenishment plan. This project pairs a machine learning forecasting pipeline with inventory formulas (EOQ, reorder point, safety stock) and ships with both a REST API and an interactive dashboard.

Credit: This project is based on an open-source repository by [ORIGINAL AUTHOR NAME] — [link to original repo]. I studied the codebase, ran the pipeline end to end, and use it as a learning project for demand forecasting and ML deployment.

What Problem Does It Solve?

A demand forecast on its own doesn't tell a store manager what to do. They need to know how much to order and when to reorder. This project closes that gap:

Train regression models on historical retail data and pick the best one.
Explain which features drive the predictions (SHAP).
Convert forecasted demand into EOQ, reorder point, daily demand and safety stock.
Expose everything through a FastAPI service and a Streamlit dashboard.
Highlights
Complete ML pipeline: ingestion → cleaning → feature engineering → preprocessing → training → evaluation → saved artifacts.
Best model: XGBoost with R² = 0.9671, MAE = 5.52, MSE = 63.92, far ahead of the linear baselines (R² ≈ 0.44).
Time-aware validation: chronological split plus TimeSeriesSplit, so the model is never trained on the future.
Leakage prevention: columns like Units Sold, Units Ordered and Inventory Level are dropped before modeling.
Explainability: SHAP summary plot generated automatically.
Config-driven: paths, model grids and training settings live in configs/config.yaml.
Two interfaces: REST API (single + batch CSV prediction) and a Streamlit dashboard with CSV export.
Dataset

76,000 retail records with product, store, price, promotion, weather, season, region and disruption information.

Column	Meaning
Date	Observation date
Store ID, Product ID	Store and product identifiers
Category, Region	Product category and sales region
Price, Discount	Retail price and discount %
Weather Condition	Weather context
Promotion	Whether a promotion was running
Competitor Pricing	Competitor price signal
Seasonality	Seasonal context
Epidemic	Disruption indicator
Demand	Target variable
Model Results
Model	MAE	MSE	R²
Linear Regression	25.69	1094.69	0.4368
Ridge	25.69	1094.77	0.4367
Lasso	25.69	1095.07	0.4366
ElasticNet	25.69	1096.23	0.4360
XGBRegressor	5.52	63.92	0.9671

The winning model is saved to models/best_model.pkl and reused by the API and dashboard. Full metrics are in artifacts/metrics/metrics.json.

How It Works
text
Raw CSV
  → Data Ingestion
  → Data Cleaning
  → Feature Engineering   (year/month/day/weekday, discounted price, leakage columns removed)
  → Preprocessing         (impute + scale numerics, impute + one-hot encode categoricals)
  → Model Training        (LinearRegression, Ridge, Lasso, ElasticNet, XGBRegressor)
  → Evaluation & Saving   (best_model.pkl, preprocessor.pkl, metrics.json, shap_summary.png)
  → Prediction Pipeline
  → Inventory Optimization (daily demand, annual demand, EOQ, reorder point, safety stock)
  → FastAPI / Streamlit
Inventory Formulas
text
Daily Demand      = Predicted Demand / Forecast Horizon
Annualized Demand = Daily Demand × 365
EOQ               = √( (2 × Annualized Demand × Ordering Cost) / Holding Cost )
Reorder Point     = (Daily Demand × Lead Time Days) + Safety Stock
Running the Project
1. Setup
powershell
git clone <your-repo-url>
cd retail-demand-inventory-intelligence

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

pip install -e .
# or, with uv:
uv sync
2. Train the model
powershell
python src/pipeline/train_pipeline.py

This produces models/best_model.pkl, the preprocessor, metrics, train/test feature files and the SHAP plot.

3. Launch the dashboard
powershell
streamlit run src/ui/main.py

Supports single-scenario prediction, batch CSV upload, adjustable ordering cost / holding cost / lead time / safety stock / horizon, KPI cards and CSV export.

4. Or start the API
powershell
python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
Method	Endpoint	Purpose
GET	/health	Check the API is up
POST	/predict	Demand + inventory metrics for one record
POST	/predict/batch	Upload a CSV, get optimized predictions back

Swagger docs: http://127.0.0.1:8000/docs

Sample request (/predict)

json
{
  "data": {
    "Date": "2026-01-15T00:00:00",
    "StoreID": "S001",
    "ProductID": "P0001",
    "Category": "Electronics",
    "Region": "North",
    "Price": 99.99,
    "Discount": 10.0,
    "WeatherCondition": "Sunny",
    "Promotion": true,
    "CompetitorPricing": 95.0,
    "Seasonality": "Winter",
    "Epidemic": 0
  },
  "optimization": {
    "ordering_cost": 50.0,
    "holding_cost": 2.0,
    "lead_time_days": 7,
    "safety_stock": 10.0,
    "horizon_days": 365
  }
}

Sample response

json
{
  "predicted_demand": 1240.58,
  "optimal_order_quantity": 249.06,
  "reorder_point": 33.79,
  "daily_demand": 3.4
}
Project Layout
text
.
├── artifacts/          # metrics, SHAP plot, preprocessor, train/test features
├── configs/config.yaml # all training and path settings
├── data/               # raw CSV and cleaned parquet
├── models/             # best_model.pkl
├── notebooks/          # EDA and model training notebooks
├── src/
│   ├── api/            # FastAPI app and Pydantic schemas
│   ├── components/     # ingestion, cleaning, features, preprocessing, training, evaluation, explainability
│   ├── optimization_strategy/  # EOQ / reorder point logic
│   ├── pipeline/       # train and predict pipelines
│   ├── services/       # forecast → inventory plan service
│   └── ui/             # Streamlit dashboard
├── pyproject.toml
└── uv.lock
Tech Stack
Area	Tools
Language	Python 3.12+
Data	pandas, NumPy
ML	scikit-learn, XGBoost, GridSearchCV, TimeSeriesSplit
Preprocessing	ColumnTransformer, SimpleImputer, StandardScaler, OneHotEncoder
Explainability	SHAP, Matplotlib
Serving	FastAPI, Pydantic, Uvicorn
UI	Streamlit
Config & packaging	PyYAML, pyproject.toml, uv
Key Entry Points
Task	File
Train models	src/pipeline/train_pipeline.py
Run inference	src/pipeline/predict_pipeline.py
API app / schemas	src/api/app.py, src/api/schemas.py
Dashboard	src/ui/main.py
Inventory formulas	src/optimization_strategy/optimization.py
Forecast → plan	src/services/inventory_service.py
Settings	configs/config.yaml
Notes
Train the model first if models/best_model.pkl is missing.
Batch CSVs must match the feature format the prediction pipeline expects.
The root main.py is not the app entrypoint; use the Streamlit or FastAPI entry points above.
