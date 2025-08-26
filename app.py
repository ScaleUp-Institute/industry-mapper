"""
IndustryMapper – Streamlit web app
Upload a Beauhurst CSV → get mapped categories + unmatched report.

Supports:
  • Format A: single "Industries" text column (comma/semicolon-separated)
  • Format B: multiple "Industries - X" boolean columns

Uses mapping_default.csv by default (kept in the repo root).
Users may override by uploading a different mapping CSV.

Requirements:
  streamlit, pandas, numpy
"""

import os
import re
import sys
from io import BytesIO
from pathlib import Path
from difflib import get_close_matches

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Industry → Category Mapper", page_icon="🧭", layout="centered")

# ─────────────────────────────────────────────────────────────
# Optional password gate (set APP_PASSWORD in Streamlit secrets)
# ─────────────────────────────────────────────────────────────
if "APP_PASSWORD" in st.secrets:
    pw = st.text_input("Enter app password", type="password")
    if pw != st.secrets["APP_PASSWORD"]:
        st.stop()

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def resource_path_prefer_external(filename: str) -> str:
    """
    Robustly resolve a data file path in three tiers:
      1) Beside the executable (frozen) or current working directory
      2) Beside this source file (repo case)
      3) PyInstaller bundle directory (if frozen)
    """
    # 1) EXE folder (frozen) or CWD
    try:
        base = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path.cwd()
    except Exception:
        base = Path.cwd()
    candidate = base / filename
    if candidate.exists():
        return str(candidate)

    # 2) Source file folder
    here = Path(__file__).resolve().parent
    candidate = here / filename
    if candidate.exists():
        return str(candidate)

    # 3) PyInstaller MEIPASS
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate = Path(meipass) / filename
        if candidate.exists():
            return str(candidate)

    # Fallback
    return str((Path.cwd() / filename).resolve())


def clean_text(s: str) -> str:
    """Normalize labels for reliable joining."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s)
    s = re.sub(r"[\u2013\u2014]", "-", s)   # en/em dash → hyphen
    s = re.sub(r"&", "AND", s)              # unify ampersand
    s = re.sub(r"/", " / ", s)              # pad slashes
    s = re.sub(r"\s+", " ", s).strip()
    return s.upper()


def split_industries(cell: str):
    """Split a cell containing comma-/semicolon-separated industries."""
    if cell is None or str(cell).strip() == "":
        return []
    parts = re.split(r"[;,]", str(cell))
    cleaned = [clean_text(p) for p in parts if str(p).strip() != ""]
    return [p for p in cleaned if p]


def normalize_bool_series(s: pd.Series) -> pd.Series:
    """Coerce common truthy/falsey tokens to boolean."""
    return (
        s.astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": True, "FALSE": False, "YES": True, "NO": False, "1": True, "0": False})
        .fillna(False)
    )


def detect_company_col(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.lower().strip() in {
        "companyname", "company name", "company", "name", "company_name"
    }]
    return candidates[0] if candidates else df.columns[0]


def infer_category_columns(mapping_df: pd.DataFrame) -> list:
    """Assume everything except 'Industry' is a category column; keep those with any True."""
    cats = [c for c in mapping_df.columns if c.lower().strip() != "industry"]
    keep = []
    for c in cats:
        col = normalize_bool_series(mapping_df[c])
        if col.any():  # at least one True
            keep.append(c)
    return keep


def melt_mapping(mapping_df: pd.DataFrame, category_columns: list) -> pd.DataFrame:
    """Convert wide mapping (Industry × categories) → long rows of (Industry_Clean, Category)."""
    m = mapping_df.copy()
    if "Industry" not in m.columns:
        raise KeyError("Mapping must contain an 'Industry' column.")
    m["Industry_Clean"] = m["Industry"].apply(clean_text)
    for cat in category_columns:
        if cat not in m.columns:
            m[cat] = False
        m[cat] = normalize_bool_series(m[cat])
    cat_long = (
        m.melt(id_vars=["Industry_Clean"], value_vars=category_columns,
               var_name="Category", value_name="Flag")
         .query("Flag == True")
         .drop(columns="Flag")
         .drop_duplicates()
    )
    return cat_long


def detect_input_format(df: pd.DataFrame):
    cols_lower = [c.lower() for c in df.columns]
    if "industries" in cols_lower:
        return "single"  # one text column with comma-separated industries
    wide = [c for c in df.columns if c.lower().startswith("industries - ")]
    return "wide" if wide else None


def bytes_from_df(df: pd.DataFrame) -> BytesIO:
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.title("🧭 Industry → Category Mapper")
st.caption("Upload a Beauhurst CSV. Get a clean mapped CSV + unmatched report. No code, no drama.")

with st.expander("How it works", expanded=False):
    st.markdown(
        "- Supports a single **Industries** text column or multiple **Industries - X** boolean columns.\n"
        "- Uses the **default mapping** (`mapping_default.csv`) in the repo, or upload a custom one.\n"
        "- Outputs a mapped dataset and an unmatched industries report with fuzzy suggestions."
    )

# Mapping source controls
use_default_mapping = st.checkbox("Use repository default mapping (mapping_default.csv)", value=True)
uploaded_mapping = None
if not use_default_mapping:
    uploaded_mapping = st.file_uploader("Or upload a mapping CSV (must include 'Industry' + category columns)", type=["csv"])

# Data file
data_file = st.file_uploader("Upload Beauhurst dataset (CSV)", type=["csv"])

# Fuzzy cutoff slider
fuzzy_cutoff = st.slider("Fuzzy suggestion cutoff (higher = stricter)", 0.50, 0.95, 0.80, 0.01)

if data_file:
    # Load dataset
    try:
        df = pd.read_csv(data_file, dtype=str, keep_default_na=False, na_values=["", "NA", "NaN"])
    except Exception as e:
        st.error(f"Could not read dataset CSV: {e}")
        st.stop()

    # Load mapping
    try:
        if uploaded_mapping is not None:
            mapping = pd.read_csv(uploaded_mapping, dtype=str)
        else:
            default_path = resource_path_prefer_external("mapping_default.csv")
            mapping = pd.read_csv(default_path, dtype=str)
    except Exception as e:
        st.error(f"Could not read mapping CSV: {e}")
        st.stop()

    # Determine categories from mapping
    try:
        category_columns = infer_category_columns(mapping)
    except KeyError as ke:
        st.error(str(ke))
        st.stop()

    if not category_columns:
        st.error("No valid category columns detected in mapping CSV (columns other than 'Industry' should contain True values).")
        st.stop()

    with st.expander("Detected category columns", expanded=False):
        st.write(category_columns)

    # Detect format + company column
    fmt = detect_input_format(df)
    if fmt is None:
        st.error("Could not detect dataset format. Expect either a single 'Industries' column, or multiple 'Industries - X' columns.")
        st.stop()
    st.info(f"Detected dataset format: **{fmt}**")

    company_col_guess = detect_company_col(df)
    company_col = st.selectbox("Company name column", options=list(df.columns),
                               index=list(df.columns).index(company_col_guess))

    # Build mapping (long)
    cat_long = melt_mapping(mapping, category_columns)
    mapped_tokens = set(cat_long["Industry_Clean"].unique())

    # Compute row-id for pivoting
    df["_ROW_ID"] = np.arange(len(df))

    # Prepare tokens by format
    if fmt == "single":
        tmp = df[["_ROW_ID", company_col, "Industries"]].copy()
        tmp["Industry_Clean"] = tmp["Industries"].apply(split_industries)
        tmp = tmp.explode("Industry_Clean", ignore_index=False)
        tmp = tmp[tmp["Industry_Clean"].notna() & (tmp["Industry_Clean"] != "")]
    else:  # wide format with multiple boolean columns
        industry_cols = [c for c in df.columns if c.lower().startswith("industries - ")]
        wide = df[["_ROW_ID", company_col] + industry_cols].copy()
        for c in industry_cols:
            wide[c] = normalize_bool_series(wide[c])
        melted = wide.melt(id_vars=["_ROW_ID", company_col], var_name="col", value_name="Flag")
        melted = melted[melted["Flag"] == True].copy()
        melted["Industry_Clean"] = melted["col"].str.replace(r"(?i)^industries\s*-\s*", "", regex=True).map(clean_text)
        tmp = melted[["_ROW_ID", company_col, "Industry_Clean"]].copy()

    # Join to mapping, pivot to booleans per category
    joined = tmp.merge(cat_long, on="Industry_Clean", how="left")
    has_cat = (
        joined.dropna(subset=["Category"])
              .assign(val=True)
              .pivot_table(index="_ROW_ID", columns="Category", values="val",
                           aggfunc="any", fill_value=False)
              .reindex(columns=category_columns, fill_value=False)
    )

    out_df = df.copy()
    for cat in category_columns:
        if cat not in out_df.columns:
            out_df[cat] = False
    for cat in category_columns:
        if cat in has_cat.columns:
            out_df.loc[has_cat.index, cat] = has_cat[cat].astype(bool)

    # Unmatched tokens report
    all_tokens = set(tmp["Industry_Clean"].dropna().unique())
    unmatched_tokens = sorted(all_tokens - mapped_tokens)

    unmatched_df = pd.DataFrame(columns=["Industry_Clean", "count", "sample_company", "suggested_mapping"])
    if unmatched_tokens:
        freq = (
            tmp[tmp["Industry_Clean"].isin(unmatched_tokens)]
            .groupby("Industry_Clean", as_index=False)
            .agg(count=("Industry_Clean", "size"),
                 sample_company=(company_col, "first"))
            .sort_values("count", ascending=False)
        )
        vocab = list(mapped_tokens)

        def suggest(x):
            m = get_close_matches(x, vocab, n=1, cutoff=fuzzy_cutoff)
            return m[0] if m else ""

        freq["suggested_mapping"] = freq["Industry_Clean"].apply(suggest)
        unmatched_df = freq

    # Stats
    st.subheader("Category coverage (unique companies)")
    stats = []
    for cat in category_columns:
        stats.append({
            "Category": cat,
            "Companies": out_df.loc[out_df[cat] == True, company_col].nunique()
        })
    st.dataframe(pd.DataFrame(stats).sort_values("Companies", ascending=False), use_container_width=True)

    # Preview
    st.subheader("Preview")
    preview_cols = [company_col]
    if "Industries" in out_df.columns:
        preview_cols.append("Industries")
    preview_cols += category_columns
    st.dataframe(out_df[preview_cols].head(20), use_container_width=True)

    # Downloads
    st.subheader("Download")
    st.download_button(
        "⬇️ Mapped dataset (CSV)",
        data=bytes_from_df(out_df.drop(columns=["_ROW_ID"], errors="ignore")),
        file_name="mapped_dataset.csv",
        mime="text/csv"
    )
    st.download_button(
        "⬇️ Unmatched industries report (CSV)",
        data=bytes_from_df(unmatched_df),
        file_name="unmatched_industries_report.csv",
        mime="text/csv",
        disabled=unmatched_df.empty
    )

else:
    st.info("Upload a Beauhurst CSV to begin. The default mapping will be used unless you upload another.")

with st.expander("Tips", expanded=False):
    st.markdown(
        "- If you see many unmatched labels, check for missing commas or odd spellings in the source.\n"
        "- To update the default mapping for everyone, replace **mapping_default.csv** in the repo and redeploy (or use the upload override).\n"
        "- If you later package this as an EXE, this app already prefers an external `mapping_default.csv` placed next to the executable."
    )
