"""
IndustryMapper – Streamlit web app
Upload a Beauhurst CSV → get mapped categories + unmatched report.

Supports:
  • Format A: single "Industries" text column (Beauhurst now uses this;
    handles commas inside official names with smart parsing)
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

# ─────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Industry → Category Mapper", page_icon="🧭", layout="centered")

# Hide Streamlit chrome (toolbar, GH link, footer)
def _hide_chrome():
    st.markdown("""
    <style>
      [data-testid="stToolbar"] {display: none !important;}      /* top-right: Share/GitHub/etc */
      #MainMenu {visibility: hidden !important;}                  /* hamburger menu */
      footer {visibility: hidden !important;}                     /* footer */
      .viewerBadge_container__r5tak, .viewerBadge_link__xq4xk,
      .viewerBadge_slot__r4o9n {display: none !important;}       /* viewer badge */
      a[href*="github.com"] {display: none !important;}          /* any direct GH links */
    </style>
    """, unsafe_allow_html=True)

_hide_chrome()

# ─────────────────────────────────────────────────────────────
# Password modal (blur/lock the page) – set APP_PASSWORD in secrets
# ─────────────────────────────────────────────────────────────
APP_PW = st.secrets.get("APP_PASSWORD", "")

def _logout():
    st.session_state.pop("authed", None)
    st.rerun()

if APP_PW:
    if "authed" not in st.session_state:
        st.session_state["authed"] = False

    if not st.session_state["authed"]:
        if hasattr(st, "dialog"):
            @st.dialog("Restricted access", width="small")
            def password_modal():
                st.write("Enter the app password to continue.")
                pw = st.text_input("Password", type="password")
                c1, c2 = st.columns([1,1])
                with c1:
                    if st.button("Continue", type="primary"):
                        if pw == APP_PW:
                            st.session_state["authed"] = True
                            st.rerun()
                        else:
                            st.error("Incorrect password")
                with c2:
                    st.button("Cancel", on_click=st.stop)
            password_modal()
            st.stop()  # keep dialog in front until success
        else:
            pw = st.text_input("Enter app password", type="password")
            if pw != APP_PW:
                st.stop()
            st.session_state["authed"] = True
            st.rerun()

    st.sidebar.button("Log out", on_click=_logout)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def resource_path_prefer_external(filename: str) -> str:
    """
    Resolve data file path robustly:
      1) Beside the executable (frozen) or current working directory
      2) Beside this source file (repo case)
      3) PyInstaller bundle directory (if frozen)
    """
    try:
        base = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path.cwd()
    except Exception:
        base = Path.cwd()
    candidate = base / filename
    if candidate.exists():
        return str(candidate)

    here = Path(__file__).resolve().parent
    candidate = here / filename
    if candidate.exists():
        return str(candidate)

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate = Path(meipass) / filename
        if candidate.exists():
            return str(candidate)

    return str((Path.cwd() / filename).resolve())


def clean_text(s: str) -> str:
    """Normalize labels for reliable joining."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s)
    s = re.sub(r"[\u2013\u2014]", "-", s)   # en/em dash → hyphen
    s = re.sub(r"&", "AND", s)              # unify &
    s = re.sub(r"/", " / ", s)              # pad slashes
    s = re.sub(r"\s+", " ", s).strip()
    return s.upper()


def normalize_bool_series(s: pd.Series) -> pd.Series:
    """Coerce common truthy/falsey tokens to boolean."""
    return (
        s.astype(str).str.strip().str.upper()
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
        return "single"  # one text column with embedded commas
    wide = [c for c in df.columns if c.lower().startswith("industries - ")]
    return "wide" if wide else None


def bytes_from_df(df: pd.DataFrame) -> BytesIO:
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


# NEW: capitalization-aware tokenizer for comma-separated cells
def tokenize_by_caps(cell: str) -> list[str]:
    """
    Split on commas, but merge segments that start with lowercase (they're
    continuations of the previous industry name). New industry begins when the
    next segment starts with an uppercase letter or a digit (e.g., '3D printing').
    Handles abbreviations like IT, R&D, B&B naturally (start uppercase).
    """
    if cell is None or str(cell).strip() == "":
        return []
    parts = [p.strip() for p in str(cell).split(",")]
    parts = [p for p in parts if p != ""]
    if not parts:
        return []
    tokens = [parts[0]]
    for seg in parts[1:]:
        first = seg[:1]
        # continuation if leading char is lowercase or '&'
        if first and (first.islower() or first in {"&"}):
            tokens[-1] = tokens[-1] + ", " + seg
        else:
            tokens.append(seg)
    return tokens


# NEW: dictionary scan to backstop lower/number-led official names
def extract_industries_from_cell(cell: str, vocab: set[str]) -> list[str]:
    """
    Find complete industry names from the mapping vocabulary inside a free-text cell.
    Longest-first so multi-phrase (incl. commas) win; prevents partial substrings.
    """
    if cell is None or str(cell).strip() == "":
        return []
    txt = clean_text(cell)
    found, remaining = [], txt
    for name in sorted(vocab, key=len, reverse=True):
        pattern = r'(?<!\w)' + re.escape(name) + r'(?!\w)'
        if re.search(pattern, remaining):
            found.append(name)
            remaining = re.sub(pattern, " ", remaining)
    return found


def uniq_preserve(seq):
    """Uniq while preserving order."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def build_sector_summary(out_df: pd.DataFrame, company_col: str, category_columns: list, mode: str = "companies") -> pd.DataFrame:
    """
    mode = 'companies'    → denominator = unique companies in the dataset
    mode = 'assignments'  → denominator = total True flags across all categories (shares sum to 100%)
    """
    # counts per category (unique companies with True)
    counts = {cat: out_df.loc[out_df[cat] == True, company_col].nunique() for cat in category_columns}
    counts_df = pd.DataFrame([
        {"Category": cat, "Companies": counts[cat]} for cat in category_columns
    ])

    if mode == "assignments":
        denom = sum(counts.values()) or 1
    else:  # 'companies'
        denom = out_df[company_col].nunique() or 1

    counts_df["Share %"] = (counts_df["Companies"] / denom * 100).round(2)
    # a “Companies per 100” metric can be handy when denominator is companies
    if mode == "companies":
        counts_df["Companies per 100"] = counts_df["Share %"].round(2)
    return counts_df.sort_values(["Share %", "Companies"], ascending=[False, False]).reset_index(drop=True)



# ─────────────────────────────────────────────────────────────
# UI (clear copy)
# ─────────────────────────────────────────────────────────────
st.title("🧭 Industry → Category Mapper")
st.caption("Upload your Beauhurst CSV. We’ll add the company categories and give you two downloads. No coding required.")

with st.expander("How it works (60-second version)", expanded=True):
    st.markdown("""
**1) Upload your Beauhurst CSV.**  
We accept either format:
- **One column** named **Industries** with multiple items (Beauhurst’s new format). We handle commas inside official names.
- **Many columns** beginning with **Industries - ...** where each column is TRUE/FALSE.

**2) Category mapping (optional).**  
If you don’t upload one, we use the **built-in mapping** (`mapping_default.csv`).  
A mapping is a simple CSV with one column **Industry** and additional columns for each category marked **True/False**.

**3) Download your results.**  
- **mapped_dataset.csv** – your original file plus the category columns.  
- **unmatched_industries_report.csv** – any labels we couldn’t match (with suggestions).  
*Note:* Suggestions are advisory only; we don’t auto-apply them.
""")

# Mapping controls
use_default_mapping = st.checkbox("Use built-in category mapping (mapping_default.csv)", value=True)
uploaded_mapping = None
if not use_default_mapping:
    uploaded_mapping = st.file_uploader(
        "Or upload a category mapping CSV (has 'Industry' + one column per category with True/False)",
        type=["csv"]
    )

# Offer mapping template download
try:
    default_path_for_dl = resource_path_prefer_external("mapping_default.csv")
    with open(default_path_for_dl, "rb") as _f:
        st.download_button("Download mapping template (CSV)", data=_f.read(),
                           file_name="mapping_template.csv", mime="text/csv")
except Exception:
    pass

# Data file
data_file = st.file_uploader("Upload your Beauhurst dataset (CSV)", type=["csv"])

# Suggestion strictness (advisory only)
fuzzy_cutoff = st.slider(
    "Suggestion strictness (affects suggestions only)",
    0.50, 0.95, 0.80, 0.01
)

# ─────────────────────────────────────────────────────────────
# Processing
# ─────────────────────────────────────────────────────────────
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

    # Determine categories
    try:
        category_columns = infer_category_columns(mapping)
    except KeyError as ke:
        st.error(str(ke)); st.stop()

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

    # Build long mapping + vocab
    cat_long = melt_mapping(mapping, category_columns)
    mapped_tokens = set(cat_long["Industry_Clean"].unique())

    # Row IDs
    df["_ROW_ID"] = np.arange(len(df))

    # Prepare tokens per row
    if fmt == "single":
        vocab = set(cat_long["Industry_Clean"].unique())
        tmp = df[["_ROW_ID", company_col, "Industries"]].copy()

        def smart_parse(cell):
            # 1) capitalization-aware comma parsing
            caps_tokens = [clean_text(t) for t in tokenize_by_caps(cell)]
            # keep only known industries
            caps_tokens = [t for t in caps_tokens if t in vocab]
            # 2) dictionary scan backup (handles leading lowercase/digits etc.)
            dict_tokens = extract_industries_from_cell(cell, vocab)
            # merge, de-dup, preserve order
            return [t for t in uniq_preserve(caps_tokens + dict_tokens) if t]

        tmp["Industry_Clean"] = tmp["Industries"].apply(smart_parse)

        # Conservative fallback (never commas) if truly nothing matched
        def fallback_tokens(s):
            if pd.isna(s):
                return []
            parts = re.split(r"[;|]", str(s))
            return [clean_text(p) for p in parts if str(p).strip()]

        no_hits = tmp["Industry_Clean"].str.len().fillna(0) == 0
        tmp.loc[no_hits, "Industry_Clean"] = tmp.loc[no_hits, "Industries"].apply(fallback_tokens)

        tmp = tmp.explode("Industry_Clean", ignore_index=False)
        tmp = tmp[tmp["Industry_Clean"].notna() & (tmp["Industry_Clean"] != "")]

    else:  # wide format
        industry_cols = [c for c in df.columns if c.lower().startswith("industries - ")]
        wide = df[["_ROW_ID", company_col] + industry_cols].copy()
        for c in industry_cols:
            wide[c] = normalize_bool_series(wide[c])
        melted = wide.melt(id_vars=["_ROW_ID", company_col], var_name="col", value_name="Flag")
        melted = melted[melted["Flag"] == True].copy()
        melted["Industry_Clean"] = melted["col"].str.replace(r"(?i)^industries\s*-\s*", "", regex=True).map(clean_text)
        tmp = melted[["_ROW_ID", company_col, "Industry_Clean"]].copy()

    # Join to mapping → categories per row
    joined = tmp.merge(cat_long, on="Industry_Clean", how="left")

    # Pivot to booleans
    has_cat = (
        joined.dropna(subset=["Category"])
              .assign(val=True)
              .pivot_table(index="_ROW_ID", columns="Category", values="val",
                           aggfunc="any", fill_value=False)
              .reindex(columns=category_columns, fill_value=False)
    )

  # ─────────────────────────────────────────────────────────────
  # Sector percentages (table + chart + download)
  # ─────────────────────────────────────────────────────────────
  st.subheader("Sector coverage & percentages")
  
  mode = st.radio(
      "Percentage mode",
      options=["Companies (denominator = unique companies)", "Assignments (denominator = total category flags)"],
      index=0,
      horizontal=True
  )
  mode_key = "companies" if mode.startswith("Companies") else "assignments"
  
  summary_df = build_sector_summary(out_df, company_col, category_columns, mode=mode_key)
  
  # Optional: filter out tiny counts for readability
  min_companies = st.slider("Hide sectors with fewer than N companies", 0, int(summary_df["Companies"].max() or 0), 0)
  summary_view = summary_df[summary_df["Companies"] >= min_companies].copy()
  
  # Show table
  st.dataframe(summary_view, use_container_width=True)
  
  # Bar chart (Share %)
  chart_data = summary_view.set_index("Category")["Share %"]
  st.bar_chart(chart_data)
  
  # Download summary
  st.download_button(
      "⬇️ Download sector summary (CSV)",
      data=summary_df.to_csv(index=False),
      file_name=f"sector_summary_{mode_key}.csv",
      mime="text/csv"
  )
  
  # Quick totals note
  total_companies = out_df[company_col].nunique()
  st.caption(
      f"Total unique companies: **{total_companies}**. "
      + (f"Sum of shares ≈ {summary_df['Share %'].sum():.2f}% (companies mode doesn’t force to 100; firms can span sectors)."
         if mode_key == 'companies' else
         "Shares sum to 100% (assignments mode).")
  )


    # Build output
    out_df = df.copy()
    for cat in category_columns:
        if cat not in out_df.columns:
            out_df[cat] = False
    for cat in category_columns:
        if cat in has_cat.columns:
            out_df.loc[has_cat.index, cat] = has_cat[cat].astype(bool)

    # Unmatched report (advisory suggestions)
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
        vocab_list = list(mapped_tokens)
        def suggest(x):
            m = get_close_matches(x, vocab_list, n=1, cutoff=fuzzy_cutoff)
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

with st.expander("Help & tips", expanded=False):
    st.markdown(
        "- If you see many **unmatched** labels, there may be missing delimiters or typos in the source file.\n"
        "- To change categories for everyone, update **mapping_default.csv** in the repo (or upload your own mapping above).\n"
        "- The app auto-detects your file format. If your file has columns like `Industries - <something>`, it treats them as TRUE/FALSE industry flags."
    )
