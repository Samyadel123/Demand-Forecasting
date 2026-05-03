# DDA Project Report

## Data Engineering

Our data engineering foundation centers on a robust, scalable pipeline leveraging **Apache Spark** and **Hadoop** ecosystem components. The architecture is designed for distributed processing, enabling us to handle millions of records efficiently.

### Data Ingestion and Storage
The lifecycle begins with an idempotent **Python ingestion script** (`load_dataset.py`). It authenticates with the Kaggle API, downloads the *Product Demand Forecasting* dataset, and streams it into a **MinIO** object store—serving as our S3-compatible data lake. This staging area ensures that our raw data is immutable and accessible to our distributed compute cluster.

### Distributed Cleaning Pipeline
We implemented a high-performance **PySpark cleaning pipeline** following a strict *Reader-Cleaner-Writer* pattern. The pipeline is orchestrated to transition raw CSV strings into optimized, schema-enforced **Apache Parquet** files. 

Key technical implementations include:
- **Schema Enforcement:** We treat all raw inputs as strings to prevent silent failures, applying explicit casts to `LongType` and `DateType` during the transformation phase.
- **Complex Parsing:** The pipeline addresses critical data-quality issues, such as converting accounting-style negative demand notation (e.g., `(1234)` to `-1234`) and unifying multiple date formats.
- **Data Quality & Profiling:** We utilize **IQR-based outlier detection** to flag extreme demand events and filter out "sparse" products (those with fewer than 10 records) that would otherwise degrade forecasting accuracy.
- **Hadoop S3A Optimization:** We leverage the **Hadoop S3A** connector for high-throughput data transfer. The final processed data is **repartitioned by `Warehouse`**, enabling downstream training jobs to perform efficient predicate pushdown and minimize data shuffling across the network.

## Data Analysis

We conducted an Exploratory Data Analysis (EDA) using PySpark and visualization libraries (Seaborn/Matplotlib) to uncover hidden patterns in the demand signal. The full analysis is available in `notebooks/eda.ipynb`.

### Key Findings
- **Pareto Volume Distribution:** A significant "Power Law" distribution was observed. **Category 019** acts as the "Whale" category, driving the vast majority of the total demand volume. The accuracy of our global forecasting model is heavily dependent on this single category.
- **Extreme Q4 Seasonality:** The heatmap analysis of *Month vs. Day-of-Week* revealed a massive surge in demand on **Mondays during the fourth quarter (Oct–Dec)**. Peak average demand in November reached nearly 6x the typical midweek volume.
- **Warehouse Specialization:** 
    - **WHSE_S** was identified as the high-volume distribution hub, maintaining consistent large-scale throughput.
    - **WHSE_C** handles specialized bulk shipments, characterized by a low frequency of orders but extremely high units per order (averaging over 42,000 units).
- **Weak Auto-Correlation:** Lag plots and ACF analysis confirmed that demand at day $T$ is a weak predictor for day $T+1$, suggesting that demand is not driven by simple momentum but by periodic inventory cycles and external seasonal factors. This informed our decision to focus on temporal features (day of week, month) rather than simple autoregressive lags.

## ML

Our machine learning strategy focuses on a "Tournament" approach, evaluating multiple model architectures against the high-volume demand signal of our "Whale" category.

### Feature Engineering
We transformed the cleaned dataset into a rich feature space using PySpark window functions:
- **Temporal Indicators:** Binary flags for **Monday** and **Q4 Seasonality**, directly addressing the spikes identified during EDA.
- **Lagged Demand:** **Lag-2** features were engineered to capture short-term momentum while avoiding direct leakage from the immediate previous day.
- **Rolling Statistics:** A **7-day rolling window** was used to calculate the moving average and standard deviation, providing the models with local context on trend and volatility.
- **Temporal Alignment:** All features were filled using a 0-fill strategy for series starts to ensure a consistent input matrix.

### Model Tournament & Evaluation
The training pipeline (`training_pipeline.py`) implements a strict **chronological split**, using data prior to 2017 for training and 2017 for testing. This simulates real-world forecasting where future demand is predicted from historical data.

We evaluated five distinct architectures:
1. **Baseline Linear:** A simple regressor to establish a performance floor.
2. **Prophet:** A decomposable additive model designed for time-series with strong seasonality.
3. **Random Forest:** A non-linear ensemble used to capture complex interactions between features.
4. **XGBoost & LightGBM:** High-performance gradient boosting machines optimized for tabular regression.

Models were scored on **Mean Absolute Error (MAE)** to measure forecasting precision. Additionally, we treated high-demand events (>10,000 units) as a classification sub-problem, monitoring **F1-Score and Accuracy** to ensure the models could reliably predict volume surges.

## Big Data

To handle the scale and variety of the demand forecasting dataset, we implemented several Big Data optimizations within the **Apache Spark** and **Hadoop** ecosystem.

### Compute & Storage Optimizations
- **Apache Parquet Transition:** We moved away from row-based CSV storage to **columnar Parquet files**. This allows for significantly faster analytical queries through **predicate pushdown**—the engine only reads the columns and partitions requested by the ML pipeline, drastically reducing I/O.
- **Data Partitioning:** The processed dataset is physically **partitioned by `Warehouse`** on the filesystem. This aligns the storage layout with our access patterns, as forecasting models are often warehouse-specific. By partitioning, Spark can bypass entire directories of irrelevant data during the read phase.
- **Shuffle Tuning:** We optimized Spark’s shuffle behavior by tuning `spark.sql.shuffle.partitions` and `spark.default.parallelism` to match our cluster’s core count. This minimizes "small file" problems and ensures balanced task distribution across the executors.
- **Efficient Windowing:** Feature engineering (lags/rolling means) was implemented using **Window partitions** (`partitionBy("Product_Code", "Warehouse")`). This ensures that time-series computations are performed locally within distributed partitions, avoiding expensive global shuffles.

### Hadoop & S3A Connectivity
We leveraged the **Hadoop S3A** protocol to provide a high-throughput bridge between our Spark compute nodes and the **MinIO** object store. This architecture allows the system to scale compute and storage independently, maintaining the fault tolerance of HDFS while providing the flexibility of an S3-compatible data lake.

## MLOps

We bridge the gap between Big Data engineering and model deployment through a containerized **MLOps lifecycle** that ensures reproducibility, auditability, and scalability.

### Experiment Tracking & Model Registry
- **Automated Logging:** We utilize **MLflow** for comprehensive experiment management. Every training run automatically captures hyperparameters, performance metrics (MAE, F1, Accuracy), and serialized model artifacts.
- **Unified Registry:** The training pipeline integrates with the **MLflow Model Registry**. Upon completion, the "Winning" model is automatically registered and promoted to a **`Production` alias**. This allows the serving layer to decouple from specific run IDs, always fetching the latest validated model version.
- **S3-Backed Artifacts:** All models and training metadata are persisted in **MinIO**, providing a central, versioned source of truth for the entire organization.

### Real-Time Inference with FastAPI
We developed a low-latency inference service using **FastAPI** to serve model predictions to business consumers.
- **Production Alias Loading:** On startup, the API queries the MLflow Registry to load the model currently tagged as `Production`.
- **Dynamic Feature Engineering:** To simplify client requests, the API performs **on-the-fly feature engineering**. A client sends a raw date and warehouse ID; the API then extracts temporal features (month, day of week) and simulates fetching historical context (lags/rolling means) before executing the prediction.
- **Strict Validation:** Using **Pydantic models**, the API ensures that incoming data adheres to the expected schema, preventing malformed requests from reaching the model.

### Containerized Orchestration
The entire ecosystem—Spark, MinIO, MLflow, and FastAPI—is orchestrated using **Docker Compose**. This modular approach allows for seamless transitions between local development, staging, and production environments, ensuring that the "Data Engineering -> ML -> Serving" loop is fully automated and robust.
