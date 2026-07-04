# End-to-End Motorcycle Market Intelligence Pipeline

A production-grade data engineering, predictive analytics, and statistical modeling platform designed to programmatically harvest unstructured web data, manage API token distribution via a custom stateful caching layer, map cross-brand pricing ecosystems using graph theory, and model continuous asset valuation metrics over temporal horizons.

---

## 1. Architectural Engineering & System Topology

The platform separates the raw data ingestion runtime from the down-stream analytical and regression modules. 

### A. Distributed Data Extraction & API Ingestion
* **Dynamic Content Extraction:** Implements `BeautifulSoup4` for high-throughput HTML DOM parsing and `Selenium Web Driver` configured in a headless browser state to circumvent client-side JavaScript execution barriers on primary web listing indices.
* **SerpApi Engine Execution:** Integrates programmatic query parsing utilizing `google-search-results` client APIs to systematically pool distributed, unstructured marketplace endpoints across primary manufacturing clusters including Harley-Davidson, Honda, Suzuki, Kawasaki, Yamaha, and BMW.

### B. Stateful File-System Caching Subsystem (`moto_cache`)
To insulate the execution routine from third-party rate limits, API token exhaustion, and network-induced transaction latency, the system utilizes a localized, decoupled caching mechanism:
* **Segmented Data Partitioning:** Raw payloads are parsed, cleansed, and serialized into isolated immutable states (`data_HarleyDavidson_save.csv`, `data_Honda_save.csv`, etc.) within a protected directory structure.
* **Temporal Tracking:** System lifecycle updates are anchored against an independent epoch configuration file (`last_scraped.txt`) evaluating data freshness.

### C. Network Graph Analytics & Non-Linear Mapping
* **Topological Market Structural Relationships:** Employs `NetworkX` graph theory frameworks to extract implicit relationships between pricing stratification vectors and manufacturing entities, representing the structural density of current market distributions as node-edge adjacencies.
* **Multi-Dimensional Matrix Projections:** Renders multi-dimensional continuous variable arrays using `Plotly Express` 3D Point Clouds, pairing spatial layout optimization with `Seaborn` kernel density estimations to isolate statistical pricing distribution anomalies.

### D. Machine Learning Predictive Core & Statistical Inference
* **Feature Optimization:** Temporal dates are transformed into absolute continuous variables using ordinal mapping conversions ($Date_{ordinal}$) to construct linear continuous input spaces.
* **Mathematical Optimization Architecture:** Leverages `scikit-learn` Linear Regression engines to minimize Ordinary Least Squares error metrics, modeling continuous pricing trends and trends metrics over a sliding temporal scale.
* **Statistical Performance Analysis:** Evaluates baseline data drift thresholds by dynamically outputting the Coefficient of Determination ($R^2$ Score) directly to the analytics framework.

---

## 2. Directory Structure

```text
├── notebooks/
│   └── 01_scraping_and_modeling_pipeline.py  # Initial prototyping, DOM parsing, and modeling notebooks
├── app.py                                    # UI Visualization Framework Layer
├── market_history.csv                        # Aggregated continuous time-series target dataset
├── moto_cache/                               # Stateful file-system caching directory
│   ├── data_save.csv
│   ├── data_HarleyDavidson_save.csv
│   ├── data_Honda_save.csv
│   └── last_scraped.txt
├── requirements.txt                          # Production software dependency requirements
└── .gitignore                                # Version control architectural safety exclusions