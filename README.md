# MSITM 5134 (Management of Organizational Data) — Midterm Group Project (Group 5)

**Project Theme:** Java Source Code Vulnerability Detection  
**Assigned Group 5 Methods:**
- **Primary Method:** `PLBART + Classification Head` (`uclanlp/plbart-base` pretrained encoder–decoder Transformer)
- **Alternative Method:** `PLBART + Frozen Embeddings + Logistic Regression` (768-D frozen embeddings + cost-sensitive L₂ Logistic Regression)

**Team Members (Group 5):**
1. **Namazbek Bekzhanov** (Group 5 Lead — Lectures 1 & 7: Lifecycle Architecture, PLBART Modeling & ROC/PR Results)
2. **Didar Auesbay** (Lectures 2, 3 & 6: Data Governance RACI, 3NF SQLite Data Warehouse Schema & RBAC Data Security)
3. **Yeldos Anarbay** (Lectures 4, 5 & 7: ETL Pipeline, SHA-256 Deduplication, 6-Rule Data Quality Gate & DevSecOps Decisions)

---

## Key Experimental Results (Test Split `n = 4,765`, Rubric Target `Accuracy >= 93%`)

| Model / Method | Group 5 Role | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PLBART + Classification Head** | **Primary Method** | **96.94%** | `0.4588` | `0.6846` | `0.5494` | `0.9571` | `0.6406` | `412.6s` |
| **PLBART + Frozen Embeddings + LR** | **Alternative Method** | **95.74%** | `0.3392` | `0.5923` | `0.4314` | `0.9285` | `0.4888` | **`18.4s` (22.4x faster)** |
| **TF-IDF + Logistic Regression** | Lexical Baseline | `91.96%` | `0.1714` | `0.5077` | `0.2563` | `0.8158` | `0.2133` | `2.1s` |

---

## Repository Contents

- `group5_plbart_pipeline.py` — End-to-end Python ETL, six-rule Data Quality Gate, 3NF SQLite warehouse builder, and model evaluation pipeline.
- `requirements.txt` — Runtime dependencies for the pipeline.
- `outputs/` and `mood_group5_warehouse.db` — Generated locally when the pipeline completes; they are not committed to this repository.

## How to Reproduce

The pipeline expects these Lab 4 files in `../lab_04/data/` relative to this repository:

- `vulnerability_samples_raw.csv`
- `cwe_reference.csv`

Install the dependencies and run the pipeline:

```bash
pip install -r requirements.txt
python group5_plbart_pipeline.py
```

The command validates the input schema, quarantines invalid rows, builds the SQLite warehouse, and writes model metrics to `outputs/group5_model_metrics.csv`.
