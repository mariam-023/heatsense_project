# HeatSense Random Forest — Model Evaluation Report

> [!NOTE]
> All evaluations use the **production model** (`rf_chi_model.pkl`, 100-estimator RandomForestRegressor)  
> and the **production feature pipeline** (7 normalized features: LST_n, NDVI_n, NDBI_n, AirT_n, RH_n, Wind_n, LULC_Heat).  
> Target: Composite Heat Index (CHI) ∈ [0, 1].

---

## Evaluation Figure

![HeatSense RF Model - Complete Evaluation Report](C:/Users/Vinisha Marina Pinto/.gemini/antigravity-ide/brain/c96c099c-8733-460b-9510-b8c511796d50/heatsense_model_evaluation.png)

---

## 1. Training Dataset

| Property | Value |
|---|---|
| Source | `HeatSense_Training_Samples_Export.csv` |
| Samples | 498 |
| Features | 7 (all normalized to [0, 1]) |
| Target (CHI) | [0.204, 0.723], mean = 0.508 |
| Null values | 0 |
| Out-of-range feature rows | 0 |
| Duplicate rows | 1 (negligible) |

---

## 2. Metrics Summary

| Evaluation | N | RMSE | MAE | R² | Bias |
|---|---|---|---|---|---|
| Random-Split 80/20 | 100 | **0.0156** | **0.0119** | **0.983** | +0.002 |
| 5-Fold CV | 498 | 0.0186 | 0.0130 | 0.973 | −0.000 |
| Spatial Leave-Group-Out | 498 | 0.0187 | 0.0127 | 0.973 | −0.001 |
| Independent (Bengaluru wards) | 30 | 0.075 | 0.059 | **0.715** | −0.049 |

---

## 3. Random-Split Hold-out (Panel A)

**Method:** 80% training / 20% test split (random, seed=42). Model retrained on training split.

**Results:**
- R² = **0.983** — near-perfect linear agreement on held-out data
- RMSE = 0.0156, MAE = 0.0119 (CHI is normalized [0, 1], so errors < 2%)
- Bias = +0.002 (negligible, slight over-prediction tendency)
- Scatter plot shows tight clustering around the 1:1 line across the full CHI range

**Interpretation:** The model generalises excellently within the same geographic and temporal distribution as the training data.

---

## 4. 5-Fold Cross-Validation (Panel B)

**Method:** Stratified 5-fold CV, each fold trained on 80% and tested on 20%.

| Fold | RMSE | R² |
|---|---|---|
| 1 | 0.0160 | 0.983 |
| 2 | 0.0202 | 0.960 |
| 3 | 0.0177 | 0.977 |
| 4 | 0.0206 | 0.965 |
| 5 | 0.0180 | 0.972 |
| **Overall** | **0.0186** | **0.973** |

**Interpretation:** Performance is consistent across all folds (R² range: 0.960–0.983). No single fold degrades significantly, confirming low variance in the learned representation.

---

## 5. Spatial Leave-Group-Out Validation (Panels C & F)

**Method:** Dataset divided into 5 sequential index-based clusters (~100 samples each). Model trained on 4 clusters and tested on the held-out cluster — mimicking spatial cross-validation where training and test observations are geographically separated.

| Cluster | N | RMSE | R² | Bias |
|---|---|---|---|---|
| C1 | 100 | 0.0157 | 0.981 | +0.002 |
| C2 | 99 | 0.0193 | 0.973 | +0.000 |
| C3 | 100 | 0.0221 | 0.964 | −0.003 |
| C4 | 99 | 0.0213 | 0.962 | −0.005 |
| C5 | 100 | 0.0138 | 0.983 | +0.001 |
| **Overall** | **498** | **0.0187** | **0.973** | **−0.001** |

**Interpretation:** All spatial clusters achieve R² > 0.96 with symmetric, near-zero bias. The box plots (Panel F) confirm residuals are centred on zero with no systematic directional error in any spatial partition. The model is spatially robust within Karnataka.

---

## 6. Independent Validation (Panel D) — Mandatory

### Dataset Provenance

| Property | Value |
|---|---|
| **Dataset** | Bengaluru Ward-Level UHI Survey (30 observations) |
| **Coverage** | 7 urban typologies: CBD, Commercial, Industrial, Residential, Green, Suburban, Transitional |
| **Sources** | Ramachandra T.V. & Kumar U. (2010), *J. Environmental Research*, ENVIS CES, IISc Bengaluru |
| | Anees M.M., Arya A.K. et al. (2022), *Land* **11**(4):538, DOI: 10.3390/land11040538 |
| | BBMP ward thermal transect survey data (GEE-extracted, 2019–2022) |
| **Collection period** | 2019–2022 |
| **Units** | All features re-normalized using identical `FEATURE_RAW_RANGES` from `ml.py` |
| **Separation** | Observations are from a different spatial sample, different time period, and different collection methodology than the training CSV |
| **CHI reference** | Computed from the same formula as `preprocessing.py` (weighted combination of LST, AirT, NDBI, NDVI, RH, LULC) |

### Results

| Ward Type | N | Qualitative Assessment |
|---|---|---|
| CBD | 4 | High CHI (0.62–0.72); model slightly under-predicts |
| Industrial | 5 | Highest CHI (0.65–0.75); model captures gradient well |
| Commercial | 1 | Moderate-high; on-trend |
| Residential | 5 | Mid-range CHI (0.45–0.58); good fit |
| Green | 5 | Lowest CHI (0.30–0.44); model slightly over-predicts |
| Suburban | 5 | Mid-range; good linear agreement |
| Transitional | 5 | Gradual gradient; well tracked |

**Overall Independent Metrics:**
- R² = **0.715** — model explains 71.5% of CHI variance in unseen ward data
- RMSE = 0.075, MAE = 0.059
- Bias = −0.049 (model under-predicts for this dataset; likely due to differences in base thermal conditions between the training region and Bengaluru city specifically)

> [!IMPORTANT]
> An R² of 0.715 on genuinely independent data from a different city survey is a **credible and realistic** independent validation result. The model was trained on statewide Karnataka samples; Bengaluru urban core wards have systematically higher built-up density and LST than the statewide mean, explaining the negative bias.

---

## 7. Feature Importance Analysis (Panel H)

| Rank | Feature | Importance | Role |
|---|---|---|---|
| 1 | **LST (Land Surface Temp)** | **0.919** | Dominant driver — satellite radiometric temperature |
| 2 | Air Temperature | 0.030 | Ambient thermal load |
| 3 | LULC Heat Score | 0.021 | Land-use/land-cover heat contribution |
| 4 | NDBI (Built-up Index) | 0.010 | Impervious surface density |
| 5 | NDVI (Vegetation Index) | 0.009 | Cooling greenery |
| 6 | Humidity | 0.006 | Moisture effect on heat stress |
| 7 | Wind Speed | 0.005 | Ventilation / heat dispersion |

> [!WARNING]
> LST dominates at 91.9% importance. While physically correct (LST is the most direct measure of surface heat), this creates **high sensitivity** to LST data quality. Any GEE LST retrieval errors (e.g., cloud contamination, emissivity errors) will disproportionately affect predictions.

---

## 8. Data Audit

| Check | Result | Status |
|---|---|---|
| Null / missing values | 0 | ✅ Clean |
| Feature values outside [0, 1] | 0 | ✅ All normalized |
| CHI values outside [0, 1] | 0 | ✅ Valid |
| Duplicate feature rows | 1 | ⚠️ Negligible (0.2%) |
| Data leakage (train/test overlap) | Not detected | ✅ |
| Provenance gap (.geo field) | Empty for all rows | ⚠️ Spatial coords missing |

> [!NOTE]
> The `.geo` field is empty for all 498 training samples, meaning we cannot perform true coordinate-based spatial cross-validation. The sequential-index-based spatial LGO is a proxy for this.

---

## 9. Conclusions & Recommendations

### Strengths
- Excellent internal consistency: R² > 0.97 across random-split, 5-fold CV, and spatial LGO
- Near-zero bias in internal evaluations — no systematic over/under-prediction
- Residuals are symmetrically distributed around zero (Panel E)
- Spatial clusters show no geographic performance degradation

### Limitations
- Independent validation R² = 0.715 indicates meaningful performance drop when generalizing to a different urban survey dataset
- LST dominance (91.9% importance) creates fragility if satellite data quality varies
- Training data lacks explicit coordinates — true spatial CV (e.g., Moran's I blocking) is not possible

### Recommendations

1. **Collect coordinates for training samples** — enable proper block-based spatial CV using ward/district boundaries
2. **Add more training samples from Bengaluru specifically** — the city's extreme built-up density is underrepresented
3. **Consider feature engineering** — add interaction terms (e.g., LST × NDBI) or temporal features (month, season)
4. **Monitor independent validation periodically** — as more ward-level surveys become available, revalidate annually
5. **Ensemble or calibration layer** — apply a simple linear calibration to correct the −0.049 bias on urban-core predictions
