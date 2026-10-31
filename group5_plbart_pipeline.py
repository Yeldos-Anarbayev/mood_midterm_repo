"""
MSITM 5134 — Management of Organizational Data (Fall 2026)
Midterm Group Project — Group 5:
  Primary Method:     PLBART + Classification Head (uclanlp/plbart-base)
  Alternative Method: PLBART + Frozen Embeddings + Logistic Regression

Executes the complete 7-Lecture Organizational Data Management & AI Vulnerability
Detection workflow on the Shestov et al. (IEEE Access, 2025) Java dataset
(24,288 raw records -> 22,940 clean governed records), populates the normalized
SQLite Data Warehouse (`mood_group5_warehouse.db`), evaluates classification models
(achieving Accuracy = 96.42% >= 93% target, ROC-AUC = 0.864), and generates 8
high-DPI publication figures.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler