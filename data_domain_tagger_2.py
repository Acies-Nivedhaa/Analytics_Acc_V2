"""
Data Domain Tagger
==================
Auto-tag database tables with their business data domains using semantic similarity.
Upload an Excel file where each sheet IS a table (with actual data rows).
The app reads column names + samples values to figure out what each table is about.

Features:
  - Per-table column quality detection (meaningful vs anonymous)
  - Value sampling + pattern detection ONLY for anonymous columns
  - Confidence threshold — low-confidence tables go to "Untagged"
  - User can reassign ANY table's domain (tagged or untagged)

Requirements:
    pip install streamlit pandas sentence-transformers scikit-learn openpyxl

Run:
    streamlit run data_domain_tagger.py
"""

import streamlit as st
import pandas as pd
import json
import re
from collections import Counter

import sqlite3
import uuid
from datetime import datetime

# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------
DB_PATH = "domain_tagger.db"

def db_init():
    """Create tables if they don't exist."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id      TEXT PRIMARY KEY,
            run_name    TEXT,
            source      TEXT,
            domain_labels TEXT,
            threshold   REAL,
            created_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS tagging_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT,
            table_name  TEXT,
            domain      TEXT,
            auto_domain TEXT,
            score       REAL,
            columns     TEXT,
            text_used   TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS user_overrides (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT,
            table_name  TEXT,
            new_domain  TEXT,
            overridden_at TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS domain_configs (
            config_name TEXT PRIMARY KEY,
            labels      TEXT,
            created_at  TEXT
        );
    """)
    con.commit()
    con.close()


def db_save_run(run_name: str, source: str, domain_labels: list, threshold: float,
                results: dict, overrides: dict) -> str:
    run_id = str(uuid.uuid4())[:8]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO runs VALUES (?,?,?,?,?,?)",
        (run_id, run_name, source, json.dumps(domain_labels), threshold, now),
    )
    for tbl, info in results.items():
        cur.execute(
            "INSERT INTO tagging_results (run_id,table_name,domain,auto_domain,score,columns,text_used) VALUES (?,?,?,?,?,?,?)",
            (run_id, tbl, info["domain"], info.get("auto_domain", info["domain"]),
             info["score"], json.dumps(info["columns"]), info.get("text_used", "")),
        )
    for tbl, new_domain in overrides.items():
        cur.execute(
            "INSERT INTO user_overrides (run_id,table_name,new_domain,overridden_at) VALUES (?,?,?,?)",
            (run_id, tbl, new_domain, now),
        )
    con.commit()
    con.close()
    return run_id


def db_load_runs() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT run_id, run_name, source, threshold, created_at FROM runs ORDER BY created_at DESC", con)
    con.close()
    return df


def db_load_run(run_id: str) -> tuple[dict, dict, list, float]:
    """Returns (results, overrides, domain_labels, threshold)."""
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT table_name, domain, auto_domain, score, columns, text_used FROM tagging_results WHERE run_id=?",
        (run_id,)
    ).fetchall()
    results = {}
    for tbl, domain, auto_domain, score, cols, text_used in rows:
        results[tbl] = {
            "domain": domain, "auto_domain": auto_domain, "score": score,
            "columns": json.loads(cols), "text_used": text_used, "col_analysis": {},
        }
    override_rows = con.execute(
        "SELECT table_name, new_domain FROM user_overrides WHERE run_id=?", (run_id,)
    ).fetchall()
    overrides = {tbl: dom for tbl, dom in override_rows}
    run_row = con.execute(
        "SELECT domain_labels, threshold FROM runs WHERE run_id=?", (run_id,)
    ).fetchone()
    domain_labels = json.loads(run_row[0]) if run_row else []
    threshold = run_row[1] if run_row else DEFAULT_CONFIDENCE_THRESHOLD
    con.close()
    return results, overrides, domain_labels, threshold


def db_save_domain_config(config_name: str, labels: list):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT OR REPLACE INTO domain_configs VALUES (?,?,?)",
        (config_name, json.dumps(labels), datetime.now().strftime("%Y-%m-%d %H:%M")),
    )
    con.commit()
    con.close()


def db_load_domain_configs() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT config_name, labels, created_at FROM domain_configs ORDER BY created_at DESC", con)
    con.close()
    return df


def db_delete_domain_config(config_name: str):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM domain_configs WHERE config_name=?", (config_name,))
    con.commit()
    con.close()


def db_delete_run(run_id: str):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM tagging_results WHERE run_id=?", (run_id,))
    con.execute("DELETE FROM user_overrides WHERE run_id=?", (run_id,))
    con.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Trino connectivity helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_trino_connection(host, port, username, password, http_scheme, cluster_name):
    """Create and cache a Trino connection."""
    from trino.dbapi import connect
    from trino.auth import BasicAuthentication
    headers = {"cluster-name": cluster_name} if cluster_name.strip() else {}
    conn = connect(
        host=host,
        port=int(port),
        auth=BasicAuthentication(username, password),
        http_scheme=http_scheme,
        http_headers=headers,
    )
    return conn


def trino_fetch_schemas(conn, catalog: str) -> list[str]:
    cur = conn.cursor()
    cur.execute(f'SHOW SCHEMAS FROM "{catalog}"')
    return [r[0] for r in cur.fetchall()]


def trino_fetch_tables(conn, catalog: str, schema: str) -> list[str]:
    cur = conn.cursor()
    cur.execute(f'SHOW TABLES FROM "{catalog}"."{schema}"')
    return [r[0] for r in cur.fetchall()]


def trino_fetch_columns_bulk(conn, catalog: str, schema: str, tables: list[str]) -> dict[str, list[str]]:
    """Fetch column names for all tables in ONE query via information_schema."""
    placeholders = ", ".join(f"'{t}'" for t in tables)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT table_name, column_name
        FROM "{catalog}"."information_schema"."columns"
        WHERE table_schema = '{schema}'
          AND table_name IN ({placeholders})
        ORDER BY table_name, ordinal_position
    """)
    result: dict[str, list[str]] = {}
    for table_name, col_name in cur.fetchall():
        result.setdefault(table_name, []).append(col_name)
    return result


def trino_fetch_sample_values(conn, catalog: str, schema: str, table: str, columns: list[str], limit: int = 50) -> dict[str, list]:
    """Fetch sample values for specific columns only — used only for anonymous columns."""
    quoted_cols = ", ".join(f'"{c}"' for c in columns)
    cur = conn.cursor()
    cur.execute(f'SELECT {quoted_cols} FROM "{catalog}"."{schema}"."{table}" LIMIT {limit}')
    rows = cur.fetchall()
    result = {col: [] for col in columns}
    for row in rows:
        for col, val in zip(columns, row):
            if val is not None:
                result[col].append(val)
    return result


def trino_build_tables_smart(conn, catalog: str, schema: str, tables: list[str], sample_limit: int = 50) -> dict[str, pd.DataFrame]:
    """
    Two-phase load:
      Phase 1: bulk-fetch all column names via information_schema (1 query total).
      Phase 2: for tables with anonymous columns only, fetch a small value sample.
    Returns {table_name: DataFrame} where meaningful tables have 0 rows (column names enough).
    """
    col_map = trino_fetch_columns_bulk(conn, catalog, schema, tables)

    result: dict[str, pd.DataFrame] = {}
    tables_needing_samples: list = []

    for tbl in tables:
        columns = col_map.get(tbl, [])
        if not columns:
            continue
        quality = detect_column_quality_per_table(columns)
        if quality == "meaningful":
            result[tbl] = pd.DataFrame(columns=columns)
        else:
            anon_cols = [c for c in columns if GENERIC_COL_PATTERN.match(c.strip())]
            tables_needing_samples.append((tbl, columns, anon_cols))

    for tbl, columns, anon_cols in tables_needing_samples:
        try:
            sample_vals = trino_fetch_sample_values(
                conn, catalog, schema, tbl, anon_cols or columns, sample_limit
            )
            df = pd.DataFrame({c: sample_vals.get(c, []) for c in columns})
            result[tbl] = df
        except Exception:
            result[tbl] = pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
UNTAGGED_LABEL = "⚠️ Untagged"

# ---------------------------------------------------------------------------
# Lazy-load the embedding model (cached so it loads only once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Step 1: Parse uploaded file into {table_name: DataFrame}
# ---------------------------------------------------------------------------

def parse_uploaded_data(uploaded_files) -> dict[str, pd.DataFrame]:
    all_tables = {}
    for uploaded_file in uploaded_files:
        name = uploaded_file.name.lower()
        if name.endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name, nrows=20)
                if not df.empty:
                    all_tables[sheet_name] = df
        elif name.endswith(".csv"):
            table_name = uploaded_file.name.rsplit(".", 1)[0]
            df = pd.read_csv(uploaded_file, nrows=20)
            if not df.empty:
                all_tables[table_name] = df
    return all_tables


# ---------------------------------------------------------------------------
# Step 2: Per-table column quality detection
# ---------------------------------------------------------------------------

GENERIC_COL_PATTERN = re.compile(
    r"^(col|column|field|c|f|var|v|attr|x|unnamed|feature)[\s_\-]?\d*$",
    re.IGNORECASE,
)

def detect_column_quality_per_table(columns: list[str]) -> str:
    if not columns:
        return "meaningful"
    generic_count = sum(1 for c in columns if GENERIC_COL_PATTERN.match(c.strip()))
    ratio = generic_count / len(columns)
    if ratio > 0.6:
        return "anonymous"
    elif ratio > 0.25:
        return "mixed"
    else:
        return "meaningful"


# ---------------------------------------------------------------------------
# Step 3: Value-level pattern detection — ONLY for anonymous columns
# ---------------------------------------------------------------------------

def detect_value_pattern(values: list) -> str:
    str_vals = [str(v).strip() for v in values if pd.notna(v) and str(v).strip() != ""]
    if not str_vals:
        return "unknown"

    sample = str_vals[:20]

    # Email
    email_matches = sum(1 for v in sample if re.match(r"^[\w.+-]+@[\w-]+\.[\w.]+$", v))
    if email_matches / len(sample) > 0.5:
        return "email_address"

    # URL
    url_matches = sum(1 for v in sample if re.match(r"^https?://", v))
    if url_matches / len(sample) > 0.5:
        return "web_url"

    # Phone
    phone_matches = sum(1 for v in sample if re.match(r"^[\+]?[\d\s\-\(\)\.]{7,18}$", v))
    if phone_matches / len(sample) > 0.5:
        return "phone_number"

    # Date
    date_patterns = [
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
        r"^\d{4}\d{2}\d{2}$",
    ]
    date_matches = sum(1 for v in sample if any(re.match(p, v) for p in date_patterns))
    if date_matches / len(sample) > 0.5:
        return "date"

    # Datetime
    ts_matches = sum(1 for v in sample if re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:", v))
    if ts_matches / len(sample) > 0.5:
        return "timestamp"

    # Time only
    time_matches = sum(1 for v in sample if re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", v))
    if time_matches / len(sample) > 0.5:
        return "time"

    # Currency with symbol
    curr_matches = sum(1 for v in sample if re.match(r"^[\$\u20ac\u00a3\u20b9\u00a5]\s?[\d,]+\.?\d*$", v))
    if curr_matches / len(sample) > 0.3:
        return "currency_amount"

    # IP address
    ip_matches = sum(1 for v in sample if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", v))
    if ip_matches / len(sample) > 0.5:
        return "ip_address"

    # JSON
    json_matches = sum(1 for v in sample if v.startswith("{") and v.endswith("}"))
    if json_matches / len(sample) > 0.3:
        return "json_data"

    # Numeric analysis
    numeric_vals = []
    for v in sample:
        cleaned = v.replace(",", "").replace(" ", "")
        try:
            numeric_vals.append(float(cleaned))
        except ValueError:
            pass

    numeric_ratio = len(numeric_vals) / len(sample) if sample else 0

    if numeric_ratio > 0.8:
        unique_ratio = len(set(sample)) / len(sample)
        all_int = all(n == int(n) for n in numeric_vals)

        if all_int and set(int(n) for n in numeric_vals).issubset({0, 1}):
            return "boolean_flag"
        elif all_int and unique_ratio > 0.9:
            return "identifier"
        elif all_int and unique_ratio < 0.2:
            return "category_code"
        elif unique_ratio > 0.8:
            avg_val = sum(numeric_vals) / len(numeric_vals)
            if avg_val > 1000:
                return "monetary_amount"
            elif 0 <= avg_val <= 100:
                return "percentage_or_score"
            else:
                return "numeric_measure"
        else:
            return "numeric_measure"

    # Text analysis
    unique_vals = set(v.lower() for v in sample)
    unique_ratio = len(unique_vals) / len(sample) if sample else 0

    bool_vals = {"true", "false", "yes", "no", "y", "n", "0", "1", "t", "f"}
    if unique_vals.issubset(bool_vals):
        return "boolean_flag"

    gender_vals = {"m", "f", "male", "female", "other", "prefer not to say", "non-binary"}
    if unique_vals.issubset(gender_vals):
        return "gender"

    if all(len(v) in (2, 3) and v.isalpha() and v.isupper() for v in sample):
        return "country_or_region_code"

    currency_codes = {"usd", "eur", "gbp", "inr", "jpy", "cad", "aud", "chf", "cny"}
    if unique_vals.issubset(currency_codes):
        return "currency_code"

    if unique_ratio < 0.15 and len(unique_vals) <= 10:
        return "status_category"

    if unique_ratio < 0.3:
        return "category"

    avg_words = sum(len(v.split()) for v in sample) / len(sample)
    if avg_words < 3 and unique_ratio > 0.7:
        return "person_or_entity_name"

    avg_len = sum(len(v) for v in sample) / len(sample)
    if avg_len > 80:
        return "long_text_description"

    address_keywords = {"st", "ave", "blvd", "dr", "ln", "rd", "street", "avenue", "drive", "lane", "road", "pkwy", "way"}
    addr_matches = sum(
        1 for v in sample
        if any(kw in v.lower().split() for kw in address_keywords) and any(c.isdigit() for c in v)
    )
    if addr_matches / len(sample) > 0.3:
        return "street_address"

    city_matches = sum(
        1 for v in sample
        if 1 <= len(v.split()) <= 3 and not any(c.isdigit() for c in v) and len(v) > 0 and v[0].isupper()
    )
    if city_matches / len(sample) > 0.7 and unique_ratio > 0.1:
        return "city_or_location"

    if unique_ratio > 0.8:
        return "unique_text"
    return "text"


def detect_sample_values(values: list) -> list[str]:
    seen = set()
    samples = []
    for v in values:
        if pd.notna(v):
            s = str(v).strip()
            if s and s not in seen:
                seen.add(s)
                samples.append(s)
                if len(samples) >= 5:
                    break
    return samples


# ---------------------------------------------------------------------------
# Step 4: Build table text representation
#   - Meaningful columns → use column names directly (NO regex on values)
#   - Anonymous columns → sample values → detect patterns → use patterns
#   - Mixed → column-by-column decision
# ---------------------------------------------------------------------------

def build_table_text(table_name: str, df: pd.DataFrame) -> tuple[str, str, dict]:
    columns = list(df.columns)
    quality = detect_column_quality_per_table(columns)

    col_analysis = {}
    text_parts = [table_name]

    for col in columns:
        is_generic = bool(GENERIC_COL_PATTERN.match(col.strip()))

        if quality == "meaningful" or (quality == "mixed" and not is_generic):
            # ---- MEANINGFUL: use column name, skip value pattern detection ----
            col_analysis[col] = {
                "quality": "meaningful",
                "pattern": "—",           # not computed, not needed
                "samples": detect_sample_values(df[col].dropna().head(20).tolist()),
            }
            text_parts.append(col)

        else:
            # ---- ANONYMOUS: sample values, detect pattern, use pattern ----
            sample_vals = df[col].dropna().head(20).tolist()
            pattern = detect_value_pattern(sample_vals)
            col_analysis[col] = {
                "quality": "anonymous",
                "pattern": pattern,
                "samples": detect_sample_values(sample_vals),
            }
            text_parts.append(pattern)

    text = " ".join(text_parts)
    return text, quality, col_analysis


# ---------------------------------------------------------------------------
# Step 5: Domain label extraction
# ---------------------------------------------------------------------------

def extract_auto_domains(
    table_texts: dict[str, str],
    table_col_info: dict[str, dict],
    top_n: int = 15,
) -> list[str]:
    stopwords = {
        "id", "key", "code", "type", "name", "date", "flag", "status",
        "num", "number", "desc", "description", "value", "amount", "count",
        "is", "has", "created", "updated", "modified", "deleted", "at", "by",
        "col", "column", "field", "var", "attr", "feature", "unnamed",
        "dim", "fact", "stg", "staging", "raw", "src", "tmp", "temp",
        "start", "end", "total", "avg", "min", "max", "sum",
        "first", "last", "new", "old", "primary", "the", "and", "for",
        "text", "unknown", "category", "identifier", "numeric", "measure",
        "boolean", "unique", "person", "entity", "street", "address",
        "email", "phone", "date", "time", "timestamp", "url", "web",
        "json", "data", "percentage", "score", "monetary", "currency",
        "city", "location", "country", "region", "gender", "long",
        "line", "unit", "per", "net", "pct", "qty", "amt", "prc",
        "archive", "hdr", "dtl", "mstr", "recv", "aging", "ref",
    }

    root_counter: Counter = Counter()

    for tbl_name in table_texts:
        parts = re.split(r"[_\-.\s]+", tbl_name.lower())
        for p in parts:
            if p and p not in stopwords and len(p) > 2 and not p.isdigit():
                root_counter[p] += 1

        if tbl_name in table_col_info:
            for col_name, info in table_col_info[tbl_name].items():
                if info["quality"] == "meaningful":
                    parts = re.split(r"[_\-.\s]+", col_name.lower())
                    for p in parts:
                        if p and p not in stopwords and len(p) > 2 and not p.isdigit():
                            root_counter[p] += 1

                if info["pattern"] in ("category", "status_category"):
                    for sv in info.get("samples", []):
                        parts = re.split(r"[_\-.\s]+", sv.lower())
                        for p in parts:
                            if p and p not in stopwords and len(p) > 2 and not p.isdigit():
                                root_counter[p] += 0.5

    domains = [word.title() for word, _ in root_counter.most_common(top_n)]
    return domains


# ---------------------------------------------------------------------------
# Step 6: Embed → UMAP → HDBSCAN → Gemini label per cluster
# ---------------------------------------------------------------------------

def embed_tables(table_texts: dict[str, str]) -> tuple[list[str], "np.ndarray"]:
    """Embed all table texts. Returns (table_names, embeddings)."""
    import numpy as np
    model = load_model()
    table_names = list(table_texts.keys())
    texts = [table_texts[t] for t in table_names]
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)
    return table_names, embeddings


def cluster_embeddings(embeddings: "np.ndarray", min_cluster_size: int = 5) -> "np.ndarray":
    """UMAP (384→10d) then HDBSCAN. Returns cluster label array (-1 = noise)."""
    import numpy as np
    import umap
    import hdbscan

    n = len(embeddings)
    # UMAP: reduce to 10d for clustering quality, cap neighbors sensibly
    n_neighbors = max(5, min(15, n // 10))
    reducer = umap.UMAP(
        n_components=min(10, n - 2),
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    clusterer.fit(reduced)
    return clusterer.labels_


def label_cluster_gemini(
    cluster_tables: list[str],
    table_texts: dict[str, str],
    api_key: str,
) -> str:
    """Send up to 8 representative table texts to Gemini Flash, get a domain label back."""
    import urllib.request

    sample = cluster_tables[:8]
    lines = [f"- {t}: {table_texts[t][:120]}" for t in sample]
    prompt = (
        "You are a data architect. These database tables belong to the same business domain:\n"
        + "\n".join(lines)
        + "\n\nRespond with a single concise business domain label (1-3 words, title-cased). "
        "Examples: Customer, Order Management, Finance, Telemetry, HR, Product Catalog. "
        "Return ONLY the label, nothing else."
    )

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 16},
    }).encode()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def tag_tables(
    table_texts: dict[str, str],
    table_col_analysis: dict[str, dict],
    table_columns: dict[str, list[str]],
    min_cluster_size: int = 5,
    gemini_api_key: str = "",
) -> dict[str, dict]:
    """
    Full pipeline:
      1. Embed (sentence-transformer, local, free)
      2. UMAP + HDBSCAN cluster
      3. Label each cluster via Gemini Flash (1 call/cluster)
      4. Build results dict in same schema as before
    """
    import numpy as np

    table_names, embeddings = embed_tables(table_texts)
    cluster_labels = cluster_embeddings(embeddings, min_cluster_size)

    # Store vectors in session for future EDA reuse
    st.session_state["table_embeddings"] = {
        "names": table_names,
        "vectors": embeddings,
        "cluster_labels": cluster_labels.tolist(),
    }

    # Group tables by cluster
    cluster_to_tables: dict[int, list[str]] = {}
    for tbl, cl in zip(table_names, cluster_labels):
        cluster_to_tables.setdefault(int(cl), []).append(tbl)

    # Label each cluster
    cluster_to_domain: dict[int, str] = {}
    for cl_id, tables in cluster_to_tables.items():
        if cl_id == -1:
            cluster_to_domain[-1] = UNTAGGED_LABEL
            continue
        if gemini_api_key.strip():
            try:
                label = label_cluster_gemini(tables, table_texts, gemini_api_key)
                cluster_to_domain[cl_id] = label
            except Exception as e:
                # Fallback: use most common word from table names in cluster
                words = []
                for t in tables:
                    words += [p for p in re.split(r"[_\-\.\s]+", t.lower()) if len(p) > 2]
                cluster_to_domain[cl_id] = Counter(words).most_common(1)[0][0].title() if words else f"Cluster {cl_id}"
        else:
            # No API key: fallback label from table name tokens
            words = []
            for t in tables:
                words += [p for p in re.split(r"[_\-\.\s]+", t.lower()) if len(p) > 2]
            cluster_to_domain[cl_id] = Counter(words).most_common(1)[0][0].title() if words else f"Cluster {cl_id}"

    # Compute per-table score = HDBSCAN soft membership (use cosine to cluster centroid as proxy)
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    cluster_centroids = {}
    for cl_id, tables in cluster_to_tables.items():
        idxs = [table_names.index(t) for t in tables]
        cluster_centroids[cl_id] = embeddings[idxs].mean(axis=0)

    results = {}
    for i, tbl in enumerate(table_names):
        cl_id = int(cluster_labels[i])
        centroid = cluster_centroids.get(cl_id, embeddings[i])
        score = float(cos_sim([embeddings[i]], [centroid])[0][0])
        domain = cluster_to_domain.get(cl_id, UNTAGGED_LABEL)
        results[tbl] = {
            "domain": domain,
            "auto_domain": domain,
            "score": round(score, 3),
            "columns": table_columns.get(tbl, []),
            "col_analysis": table_col_analysis.get(tbl, {}),
            "text_used": table_texts[tbl],
            "cluster_id": cl_id,
        }

    return results


def tag_tables_predefined(
    table_texts: dict[str, str],
    table_col_analysis: dict[str, dict],
    table_columns: dict[str, list[str]],
    domain_labels: list[str],
    confidence_threshold: float = 0.25,
) -> dict[str, dict]:
    """
    Predefined-domain mode:
      1. Embed all table texts (sentence-transformer, local)
      2. Embed the domain labels
      3. Cosine similarity: each table → best-matching domain
      4. Below threshold → Untagged
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    model = load_model()

    # Embed tables
    table_names = list(table_texts.keys())
    table_vecs = model.encode(
        [table_texts[t] for t in table_names],
        show_progress_bar=False, batch_size=64,
    )

    # Embed domain labels — prefix with context for better matching
    domain_prompts = [f"{d} department data tables columns" for d in domain_labels]
    domain_vecs = model.encode(domain_prompts, show_progress_bar=False)

    # Cosine similarity matrix: (n_tables, n_domains)
    sim_matrix = cos_sim(table_vecs, domain_vecs)

    # Store vectors in session for future reuse
    st.session_state["table_embeddings"] = {
        "names": table_names,
        "vectors": table_vecs,
        "cluster_labels": [],
    }

    results = {}
    for i, tbl in enumerate(table_names):
        best_idx = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i][best_idx])

        if best_score >= confidence_threshold:
            domain = domain_labels[best_idx]
        else:
            domain = UNTAGGED_LABEL

        # Also include runner-up for context
        sorted_idxs = np.argsort(sim_matrix[i])[::-1]
        runner_up_idx = int(sorted_idxs[1]) if len(sorted_idxs) > 1 else best_idx
        runner_up = domain_labels[runner_up_idx]
        runner_up_score = float(sim_matrix[i][runner_up_idx])

        results[tbl] = {
            "domain": domain,
            "auto_domain": domain,
            "score": round(best_score, 3),
            "columns": table_columns.get(tbl, []),
            "col_analysis": table_col_analysis.get(tbl, {}),
            "text_used": table_texts[tbl],
            "cluster_id": best_idx if best_score >= confidence_threshold else -1,
            "runner_up": f"{runner_up} ({runner_up_score:.2f})",
        }

    return results


def group_by_domain(results: dict[str, dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for tbl, info in results.items():
        domain = info["domain"]
        if domain not in grouped:
            grouped[domain] = []
        grouped[domain].append({
            "table": tbl,
            "score": info["score"],
            "columns": info["columns"],
            "col_analysis": info.get("col_analysis", {}),
            "text_used": info.get("text_used", ""),
            "auto_domain": info.get("auto_domain", domain),
            "runner_up": info.get("runner_up", ""),
        })
    # Sort: Untagged last, rest by count descending
    def sort_key(item):
        domain, tbls = item
        if domain == UNTAGGED_LABEL:
            return (1, 0)
        return (0, -len(tbls))
    grouped = dict(sorted(grouped.items(), key=sort_key))
    return grouped


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
DOMAIN_COLORS = [
    ("#ede9fe", "#5b21b6"),   # violet
    ("#d1fae5", "#065f46"),   # emerald
    ("#ffedd5", "#9a3412"),   # orange
    ("#dbeafe", "#1e40af"),   # blue
    ("#fef3c7", "#92400e"),   # amber
    ("#fce7f3", "#9d174d"),   # pink
    ("#dcfce7", "#166534"),   # green
    ("#fee2e2", "#991b1b"),   # red
    ("#f1f5f9", "#475569"),   # slate
    ("#e0e7ff", "#3730a3"),   # indigo
]
UNTAGGED_COLOR = ("#fef9c3", "#854d0e")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    db_init()

    st.set_page_config(
        page_title="Data Domain Tagger",
        page_icon="🏷️",
        layout="wide",
    )

    # ── Sidebar: run history ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Run history")
        history_df = db_load_runs()
        if history_df.empty:
            st.caption("No saved runs yet.")
        else:
            for _, row in history_df.iterrows():
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    if st.button(
                        f"**{row['run_name']}**\n{row['created_at']}",
                        key=f"load_run_{row['run_id']}",
                        use_container_width=True,
                    ):
                        r, o, dl, thr = db_load_run(row["run_id"])
                        st.session_state["tag_results"] = r
                        st.session_state["user_overrides"] = o
                        st.session_state["domain_labels_used"] = dl
                        st.session_state["loaded_run_id"] = row["run_id"]
                        st.rerun()
                with col_b:
                    if st.button("🗑", key=f"del_run_{row['run_id']}", help="Delete run"):
                        db_delete_run(row["run_id"])
                        st.rerun()

        st.divider()
        st.markdown("### Domain configs")
        configs_df = db_load_domain_configs()
        if configs_df.empty:
            st.caption("No saved configs yet.")
        else:
            for _, row in configs_df.iterrows():
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    if st.button(row["config_name"], key=f"load_cfg_{row['config_name']}", use_container_width=True):
                        st.session_state["load_domain_config"] = json.loads(row["labels"])
                        st.rerun()
                with col_b:
                    if st.button("🗑", key=f"del_cfg_{row['config_name']}", help="Delete config"):
                        db_delete_domain_config(row["config_name"])
                        st.rerun()

    st.markdown("""
    <style>
        /* ── Layout ─────────────────────────────────────────── */
        .block-container { padding-top: 1rem !important; max-width: 980px; }
        header[data-testid="stHeader"] { display: none !important; }
        #MainMenu { visibility: hidden; }

        /* ── Page header ────────────────────────────────────── */
        .app-header {
            display: flex; align-items: baseline; gap: 8px;
            margin: 0 0 2px 0; padding: 0;
        }
        .app-header-title {
            font-size: 24px; font-weight: 700; letter-spacing: -0.3px;
            margin: 0; padding: 0; line-height: 1.2;
        }
        .app-header-version {
            font-size: 12px; font-weight: 600; color: #a1a1aa;
            background: #f4f4f5; padding: 2px 8px; border-radius: 100px;
        }
        .app-subtitle {
            font-size: 13px; color: #71717a; margin: 2px 0 0 0; line-height: 1.5;
        }

        /* ── Section labels ─────────────────────────────────── */
        .section-label {
            font-size: 11px; font-weight: 700; color: #a1a1aa;
            text-transform: uppercase; letter-spacing: 1.2px;
            margin: 0 0 12px 0; padding: 0;
        }
        .section-label .step-num {
            display: inline-flex; align-items: center; justify-content: center;
            width: 20px; height: 20px; border-radius: 50%;
            background: #f4f4f5; color: #71717a;
            font-size: 11px; font-weight: 700; margin-right: 6px;
        }

        /* ── Domain badges ──────────────────────────────────── */
        .domain-badge {
            display: inline-block; padding: 5px 16px; border-radius: 8px;
            font-size: 14px; font-weight: 600; margin-right: 8px;
            letter-spacing: 0.2px;
        }

        /* ── Table chips ────────────────────────────────────── */
        .table-chip {
            display: inline-flex; align-items: center; gap: 6px;
            background: #fafafa; border: 1px solid #e4e4e7;
            padding: 5px 12px; border-radius: 6px;
            font-size: 13px; margin: 3px 4px;
            font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
            cursor: default; transition: background 0.15s;
        }
        .table-chip:hover { background: #f4f4f5; }

        .score-text {
            font-size: 11px; color: #a1a1aa; font-weight: 500;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* ── Quality pills ──────────────────────────────────── */
        .quality-meaningful {
            color: #065f46; background: #d1fae5;
            padding: 3px 10px; border-radius: 100px; font-size: 12px;
            font-weight: 600; display: inline-block;
        }
        .quality-anonymous {
            color: #9a3412; background: #ffedd5;
            padding: 3px 10px; border-radius: 100px; font-size: 12px;
            font-weight: 600; display: inline-block;
        }
        .quality-mixed {
            color: #854d0e; background: #fef9c3;
            padding: 3px 10px; border-radius: 100px; font-size: 12px;
            font-weight: 600; display: inline-block;
        }

        /* ── Reassigned badge ───────────────────────────────── */
        .reassigned-badge {
            display: inline-block; background: #dbeafe; color: #1e40af;
            padding: 2px 8px; border-radius: 100px; font-size: 10px;
            font-weight: 600; margin-left: 4px; letter-spacing: 0.3px;
            text-transform: uppercase;
        }

        /* ── Metric cards ───────────────────────────────────── */
        [data-testid="stMetric"] {
            background: #fafafa; border: 1px solid #e4e4e7;
            border-radius: 10px; padding: 16px 18px 14px;
        }
        [data-testid="stMetricLabel"] p {
            font-size: 12px !important; font-weight: 600 !important;
            color: #71717a !important; text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        [data-testid="stMetricValue"] {
            font-size: 28px !important; font-weight: 700 !important;
            color: #18181b !important;
        }

        /* ── Sidebar polish ─────────────────────────────────── */
        section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
            font-size: 13px; font-weight: 700; color: #71717a;
            text-transform: uppercase; letter-spacing: 0.8px;
        }

        /* ── Expander cleanup ───────────────────────────────── */
        [data-testid="stExpander"] {
            border: 1px solid #e4e4e7 !important;
            border-radius: 10px !important;
            background: #fff !important;
        }
        [data-testid="stExpander"] summary {
            font-size: 14px; font-weight: 600;
        }

        /* ── Tabs ───────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0; font-size: 14px; font-weight: 600;
        }

        /* ── Buttons ────────────────────────────────────────── */
        .stDownloadButton button, [data-testid="stBaseButton-primary"] {
            border-radius: 8px !important; font-weight: 600 !important;
        }
        .stDownloadButton button {
            border: 1px solid #e4e4e7 !important;
        }

        /* ── Summary bar ────────────────────────────────────── */
        .summary-bar {
            display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
            font-size: 13px; color: #71717a; margin-top: 4px;
        }
        .summary-bar span { display: inline-flex; align-items: center; gap: 4px; }
        .summary-bar .dot {
            width: 4px; height: 4px; border-radius: 50%;
            background: #d4d4d8; display: inline-block;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Page header ──
    st.markdown("""
        <div class="app-header">
            <span class="app-header-title">Data Domain Tagger</span>
        </div>
        <p class="app-subtitle">
            Auto-classify database tables into business domains using semantic similarity.<br>
            Upload data or connect to Trino — column names and value patterns drive the tagging.
        </p>
    """, unsafe_allow_html=True)

    st.divider()

    # ---------- Section 1: Data Source ----------
    st.markdown('<p class="section-label"><span class="step-num">1</span> Data source</p>', unsafe_allow_html=True)

    tab_file, tab_trino = st.tabs(["Upload files", "Trino / DataOS"])

    all_tables: dict = {}

    # ── Tab A: File upload ──────────────────────────────────────────────────
    with tab_file:
        col_upload, col_info = st.columns([2, 1])
        with col_upload:
            uploaded_files = st.file_uploader(
                "Upload data files",
                type=["xlsx", "xls", "csv"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                help="Excel file: each sheet = one table. Or upload multiple CSVs.",
            )
        with col_info:
            st.caption("""
            **Excel** (.xlsx) — each sheet = one table  
            **CSV** — each file = one table  
            Upload the actual data, not just column names.
            """)

        if uploaded_files:
            with st.spinner("Reading tables..."):
                all_tables = parse_uploaded_data(uploaded_files)
            if not all_tables:
                st.error("No tables found in the uploaded files.")

    # ── Tab B: Trino / DataOS ───────────────────────────────────────────────
    with tab_trino:
        st.caption("**Connection**")

        c1, c2 = st.columns([3, 1])
        with c1:
            trino_host = st.text_input(
                "Host", value="tcp.known-racer.mydataos.com",
                placeholder="tcp.your-instance.mydataos.com",
            )
        with c2:
            trino_port = st.number_input("Port", value=7432, min_value=1, max_value=65535, step=1)

        c3, c4 = st.columns(2)
        with c3:
            trino_user = st.text_input("Username", placeholder="your-username")
        with c4:
            trino_pass = st.text_input("Password / Token", type="password", placeholder="••••••••")

        c5, c6 = st.columns(2)
        with c5:
            trino_scheme = st.selectbox("HTTP scheme", options=["https", "http"], index=0)
        with c6:
            trino_cluster = st.text_input(
                "Cluster name (cluster-name header)",
                value="minervac",
                placeholder="minervac",
                help="Passed as the 'cluster-name' HTTP header — leave blank if not needed.",
            )

        st.caption("**Schema**")
        c7, c8 = st.columns(2)
        with c7:
            trino_catalog = st.text_input("Catalog", placeholder="icebase")
        with c8:
            trino_schema_input = st.text_input("Schema", placeholder="telemetry")

        # ── Connect & list tables ──
        fetch_col, _ = st.columns([1, 3])
        with fetch_col:
            fetch_tables_btn = st.button("Fetch tables", use_container_width=True)

        if fetch_tables_btn:
            if not all([trino_host, trino_user, trino_pass, trino_catalog, trino_schema_input]):
                st.error("Fill in host, username, password, catalog and schema first.")
            else:
                try:
                    with st.spinner("Connecting to Trino..."):
                        conn = get_trino_connection(
                            trino_host, int(trino_port), trino_user, trino_pass,
                            trino_scheme, trino_cluster,
                        )
                        table_list = trino_fetch_tables(conn, trino_catalog, trino_schema_input)
                    st.session_state["trino_table_list"] = table_list
                    st.session_state["trino_conn_params"] = {
                        "host": trino_host, "port": int(trino_port),
                        "user": trino_user, "pass": trino_pass,
                        "scheme": trino_scheme, "cluster": trino_cluster,
                        "catalog": trino_catalog, "schema": trino_schema_input,
                    }
                    st.success(f"Connected — {len(table_list)} tables found in `{trino_catalog}.{trino_schema_input}`")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
                    st.session_state.pop("trino_table_list", None)

        # ── Table multiselect ──
        if "trino_table_list" in st.session_state:
            table_list = st.session_state["trino_table_list"]
            selected_tables = st.multiselect(
                f"Select tables from `{st.session_state['trino_conn_params']['catalog']}.{st.session_state['trino_conn_params']['schema']}`",
                options=table_list,
                default=table_list[:min(5, len(table_list))],
                help="Choose one or more tables to tag.",
            )

            st.caption(
                "ℹ️ **Smart loading:** column names are fetched in a single bulk query. "
                "Row samples are only fetched for tables with anonymous column names."
            )

            sample_limit = st.slider(
                "Sample rows (anonymous tables only)", min_value=20, max_value=200,
                value=50, step=10,
                help="Only applies to tables whose column names are generic (col1, c2, etc). Meaningful tables need zero rows.",
            )

            load_col, _ = st.columns([1, 3])
            with load_col:
                load_btn = st.button("Load selected tables", use_container_width=True, type="primary")

            if load_btn:
                if not selected_tables:
                    st.warning("Select at least one table.")
                else:
                    p = st.session_state["trino_conn_params"]
                    conn = get_trino_connection(
                        p["host"], p["port"], p["user"], p["pass"],
                        p["scheme"], p["cluster"],
                    )
                    try:
                        with st.spinner(f"Phase 1/2 — fetching column names for {len(selected_tables)} tables…"):
                            loaded = trino_build_tables_smart(
                                conn, p["catalog"], p["schema"], selected_tables, sample_limit
                            )
                        if loaded:
                            meaningful_count = sum(1 for df in loaded.values() if len(df) == 0)
                            anon_count = len(loaded) - meaningful_count
                            st.session_state["trino_loaded_tables"] = loaded
                            st.success(
                                f"Loaded {len(loaded)} tables — "
                                f"{meaningful_count} used column names only, "
                                f"{anon_count} fetched row samples."
                            )
                        else:
                            st.error("No tables loaded — check permissions or table names.")
                    except Exception as e:
                        st.error(f"Load failed: {e}")

        # Merge Trino tables into all_tables (takes precedence if loaded)
        if "trino_loaded_tables" in st.session_state:
            all_tables = st.session_state["trino_loaded_tables"]

    # ── Guard: need at least one source ────────────────────────────────────
    if not all_tables:
        st.info("Upload files or connect to Trino to get started.", icon="👆")
        st.stop()

    st.divider()

    # ---------- Section 2: Analysis ----------
    st.markdown('<p class="section-label"><span class="step-num">2</span> Table analysis</p>', unsafe_allow_html=True)

    total_tables = len(all_tables)
    total_cols = sum(len(df.columns) for df in all_tables.values())
    avg_cols = total_cols / total_tables if total_tables else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Tables found", total_tables)
    m2.metric("Total columns", total_cols)
    m3.metric("Avg columns / table", f"{avg_cols:.0f}")

    # Build text representations
    with st.spinner("Analyzing column names and sampling values..."):
        table_texts = {}
        table_qualities = {}
        table_col_analysis = {}
        table_columns = {}

        for tbl_name, df in all_tables.items():
            text, quality, col_analysis = build_table_text(tbl_name, df)
            table_texts[tbl_name] = text
            table_qualities[tbl_name] = quality
            table_col_analysis[tbl_name] = col_analysis
            table_columns[tbl_name] = list(df.columns)

    q_counts = Counter(table_qualities.values())

    st.markdown("")
    st.caption("**Column name quality per table**")

    quality_summary = []
    if q_counts.get("meaningful", 0):
        quality_summary.append(f'<span class="quality-meaningful">meaningful: {q_counts["meaningful"]}</span>')
    if q_counts.get("mixed", 0):
        quality_summary.append(f'<span class="quality-mixed">mixed: {q_counts["mixed"]}</span>')
    if q_counts.get("anonymous", 0):
        quality_summary.append(f'<span class="quality-anonymous">anonymous: {q_counts["anonymous"]}</span>')
    st.markdown("  ".join(quality_summary), unsafe_allow_html=True)

    # Preview
    with st.expander(f"Preview all {total_tables} tables with detected patterns"):
        for tbl_name in sorted(all_tables.keys()):
            df = all_tables[tbl_name]
            quality = table_qualities[tbl_name]
            col_info_dict = table_col_analysis[tbl_name]

            q_class = f"quality-{quality}"
            st.markdown(
                f"**`{tbl_name}`** ({len(df)} rows) "
                f'<span class="{q_class}">{quality}</span>',
                unsafe_allow_html=True,
            )

            col_details = []
            for col, info in col_info_dict.items():
                samples_str = ", ".join(info["samples"][:3]) if info["samples"] else "—"
                col_details.append({
                    "Column": col,
                    "Quality": info["quality"],
                    "Used for embedding": col if info["quality"] == "meaningful" else info["pattern"],
                    "Detected pattern": info["pattern"],
                    "Sample values": samples_str,
                })
            if col_details:
                st.dataframe(
                    pd.DataFrame(col_details),
                    use_container_width=True,
                    hide_index=True,
                    height=min(35 + len(col_details) * 35, 250),
                )
            st.markdown("---")

    st.divider()

    # ---------- Section 3: Configure ----------
    st.markdown('<p class="section-label"><span class="step-num">3</span> Configure</p>', unsafe_allow_html=True)

    # ── Domain mode selection ──
    # Check if domains were loaded from sidebar config
    loaded_domains = st.session_state.pop("load_domain_config", None)
    if loaded_domains:
        st.session_state["predefined_domains_text"] = ", ".join(loaded_domains)

    # Determine default mode: if domains are loaded, default to predefined
    has_predefined = bool(st.session_state.get("predefined_domains_text", "").strip())
    default_mode_idx = 1 if has_predefined else 0

    tagging_mode = st.radio(
        "Tagging mode",
        options=["Auto-discover domains", "Match to predefined domains"],
        index=default_mode_idx,
        horizontal=True,
        help="Auto-discover uses UMAP + HDBSCAN to find natural clusters. Predefined matches every table to one of your specified domains.",
    )

    use_predefined = tagging_mode == "Match to predefined domains"

    if use_predefined:
        # ── Predefined domain config ──
        predefined_text = st.text_area(
            "Domain labels (comma-separated)",
            value=st.session_state.get("predefined_domains_text", ""),
            placeholder="e.g. Customer, Sales, Product, Finance, Logistics, HR, Telemetry",
            height=68,
            key="predefined_domains_input",
        )
        # Keep in session so it persists
        st.session_state["predefined_domains_text"] = predefined_text

        predefined_labels = [d.strip().title() for d in predefined_text.split(",") if d.strip()]

        if predefined_labels:
            pills_html = " ".join(
                f'<span class="quality-meaningful">{d}</span>' for d in predefined_labels
            )
            st.markdown(f"{len(predefined_labels)} domains: {pills_html}", unsafe_allow_html=True)
        else:
            st.warning("Enter at least one domain label, or load a saved config from the sidebar.")

        cfg_c1, cfg_c2 = st.columns(2)
        with cfg_c1:
            confidence_threshold = st.slider(
                "Confidence threshold",
                min_value=0.05, max_value=0.60, value=DEFAULT_CONFIDENCE_THRESHOLD, step=0.05,
                help="Tables scoring below this against all domains go to Untagged.",
            )
        with cfg_c2:
            # Save config shortcut
            with st.expander("Save this domain set"):
                cfg_name = st.text_input("Config name", placeholder="e.g. Retail domains", label_visibility="collapsed", key="save_cfg_name")
                if st.button("Save config", use_container_width=True):
                    if not cfg_name.strip():
                        st.warning("Enter a name.")
                    elif not predefined_labels:
                        st.warning("No labels to save.")
                    else:
                        db_save_domain_config(cfg_name.strip(), predefined_labels)
                        st.success(f"Saved '{cfg_name}'")
                        st.rerun()

    else:
        # ── Auto-discover config ──
        cfg_c1, cfg_c2 = st.columns(2)
        with cfg_c1:
            min_cluster_size = st.slider(
                "Min cluster size",
                min_value=2, max_value=30, value=5, step=1,
                help="Minimum tables per cluster. Smaller = more clusters. Tables that don't fit any cluster go to Untagged.",
            )
        with cfg_c2:
            gemini_api_key = st.text_input(
                "Gemini API key (for cluster labelling)",
                type="password",
                placeholder="AIza...",
                help="Optional — without a key, labels fall back to table name tokens. 1 API call per cluster, very cheap.",
            )
        if not gemini_api_key.strip():
            st.caption("No API key — cluster labels will be inferred from table name tokens.")

        confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD

        # Save domain config
        with st.expander("Save a manual domain label set"):
            manual_labels_input = st.text_area(
                "Domain labels",
                placeholder="e.g. Customer, Sales, Product, Finance, Logistics, HR",
                height=68, label_visibility="collapsed",
            )
            cfg_col1, cfg_col2 = st.columns([3, 1])
            with cfg_col1:
                cfg_name = st.text_input("Config name", placeholder="e.g. Retail domains", label_visibility="collapsed")
            with cfg_col2:
                if st.button("Save config", use_container_width=True):
                    if not cfg_name.strip():
                        st.warning("Enter a name.")
                    elif not manual_labels_input.strip():
                        st.warning("No labels to save.")
                    else:
                        labels = [d.strip().title() for d in manual_labels_input.split(",") if d.strip()]
                        db_save_domain_config(cfg_name.strip(), labels)
                        st.success(f"Saved '{cfg_name}'")
                        st.rerun()

    # ── Run button ──
    st.markdown("")
    button_label = "Match tables to domains" if use_predefined else "Cluster & label tables"
    if st.button(button_label, type="primary", use_container_width=True):

        if use_predefined:
            if not predefined_labels:
                st.error("Enter at least one domain label first.")
                st.stop()

            with st.spinner(f"Embedding {len(table_texts)} tables + {len(predefined_labels)} domains…"):
                results = tag_tables_predefined(
                    table_texts, table_col_analysis,
                    table_columns, predefined_labels,
                    confidence_threshold,
                )
        else:
            with st.spinner(f"Embedding {len(table_texts)} tables, clustering, and labelling…"):
                results = tag_tables(
                    table_texts, table_col_analysis,
                    table_columns, min_cluster_size,
                    gemini_api_key,
                )

        # In predefined mode, include ALL input labels so reassignment dropdown has every option
        if use_predefined:
            domain_labels = sorted(set(predefined_labels + [r["domain"] for r in results.values()]))
        else:
            domain_labels = sorted(set(r["domain"] for r in results.values()))

        # Store in session state
        st.session_state["tag_results"] = results
        st.session_state["domain_labels_used"] = domain_labels
        st.session_state["last_threshold"] = confidence_threshold
        st.session_state["last_mode"] = tagging_mode
        st.session_state["user_overrides"] = {}

    # ---------- Section 4: Results with editable domains ----------
    if "tag_results" not in st.session_state:
        st.stop()

    results = st.session_state["tag_results"]
    domain_labels_used = st.session_state["domain_labels_used"]
    user_overrides = st.session_state.get("user_overrides", {})
    confidence_threshold = st.session_state.get("last_threshold", DEFAULT_CONFIDENCE_THRESHOLD)

    # Apply any user overrides to results
    for tbl, new_domain in user_overrides.items():
        if tbl in results:
            results[tbl]["domain"] = new_domain

    grouped = group_by_domain(results)

    st.divider()
    st.markdown('<p class="section-label"><span class="step-num">4</span> Results</p>', unsafe_allow_html=True)

    # Summary
    num_tables = len(results)
    num_untagged = sum(1 for r in results.values() if r["domain"] == UNTAGGED_LABEL)
    num_tagged = num_tables - num_untagged
    num_reassigned = len(user_overrides)
    num_domains = len([d for d in grouped if d != UNTAGGED_LABEL])
    avg_score = sum(r["score"] for r in results.values()) / num_tables if num_tables else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tagged", num_tagged)
    m2.metric("Untagged", num_untagged)
    m3.metric("Domains", num_domains)
    m4.metric("Reassigned by you", num_reassigned)

    last_mode = st.session_state.get("last_mode", "Auto-discover domains")
    mode_label = "Predefined" if "Predefined" in last_mode or "predefined" in last_mode or "Match" in last_mode else "Auto-discover"

    st.markdown(
        f'<div class="summary-bar">'
        f'<span>Mode: <b>{mode_label}</b></span>'
        f'<span class="dot"></span>'
        f'<span>Domains: <b>{", ".join(domain_labels_used)}</b></span>'
        f'<span class="dot"></span>'
        f'<span>Threshold: <b>{confidence_threshold}</b></span>'
        f'<span class="dot"></span>'
        f'<span>Avg score: <b>{avg_score:.2f}</b></span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # Dropdown options for reassignment — include all predefined domains even if no tables matched
    reassign_options = sorted(set(domain_labels_used))
    if UNTAGGED_LABEL in reassign_options:
        reassign_options.remove(UNTAGGED_LABEL)
    reassign_options.append(UNTAGGED_LABEL)

    # ---------- Render each domain group ----------
    for idx, (domain, tbl_list) in enumerate(grouped.items()):
        if domain == UNTAGGED_LABEL:
            bg_color, text_color = UNTAGGED_COLOR
        else:
            color_idx = idx if domain != UNTAGGED_LABEL else 0
            bg_color, text_color = DOMAIN_COLORS[color_idx % len(DOMAIN_COLORS)]

        tbl_count = len(tbl_list)
        avg_domain_score = sum(t["score"] for t in tbl_list) / tbl_count if tbl_count else 0

        # Domain card header
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:10px; margin-top:16px; margin-bottom:6px;">'
            f'<span class="domain-badge" style="background:{bg_color}; color:{text_color};">{domain}</span>'
            f'<span class="score-text" style="font-size:13px;">'
            f'{tbl_count} table{"s" if tbl_count != 1 else ""}'
            f' · avg {avg_domain_score:.2f}'
            f'</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Table chips (quick overview)
        chips_html = '<div style="display:flex; flex-wrap:wrap; gap:4px; margin-bottom:8px;">'
        for t in sorted(tbl_list, key=lambda x: -x["score"]):
            score_display = f"{t['score']:.2f}"
            was_reassigned = t["table"] in user_overrides
            reassigned_tag = '<span class="reassigned-badge">moved</span>' if was_reassigned else ""
            chips_html += (
                f'<span class="table-chip">{t["table"]}'
                f'<span class="score-text">{score_display}</span>'
                f'{reassigned_tag}</span>'
            )
        chips_html += '</div>'
        st.markdown(chips_html, unsafe_allow_html=True)

        # Expandable: details + reassignment dropdowns
        with st.expander(f"Details & reassign"):
            for t in sorted(tbl_list, key=lambda x: -x["score"]):
                tbl_name = t["table"]
                auto_domain = t.get("auto_domain", domain)
                current_domain = domain

                col_detail, col_reassign = st.columns([3, 1])

                with col_detail:
                    auto_note = ""
                    if tbl_name in user_overrides:
                        auto_note = f"  ← was *{auto_domain}*"
                    runner_up = t.get("runner_up", "")
                    runner_note = f"  · 2nd: {runner_up}" if runner_up else ""
                    st.markdown(f"**`{tbl_name}`** · {t['score']:.3f}{auto_note}")
                    st.caption(f"{t['text_used'][:120]}{'…' if len(t['text_used']) > 120 else ''}{runner_note}")

                with col_reassign:
                    current_idx = reassign_options.index(current_domain) if current_domain in reassign_options else 0
                    new_domain = st.selectbox(
                        "Move to",
                        options=reassign_options,
                        index=current_idx,
                        key=f"reassign_{tbl_name}",
                        label_visibility="collapsed",
                    )

                    if new_domain != current_domain:
                        st.session_state["user_overrides"][tbl_name] = new_domain
                        st.rerun()

                # Column analysis table
                if t.get("col_analysis"):
                    rows_display = []
                    for col, info in t["col_analysis"].items():
                        rows_display.append({
                            "Column": col,
                            "Quality": info["quality"],
                            "Embedding input": col if info["quality"] == "meaningful" else info["pattern"],
                            "Samples": ", ".join(info["samples"][:3]),
                        })
                    if rows_display:
                        st.dataframe(
                            pd.DataFrame(rows_display),
                            use_container_width=True,
                            hide_index=True,
                            height=min(35 + len(rows_display) * 35, 180),
                        )
                st.markdown("---")

        st.markdown("")

    # ---------- Export ----------
    st.divider()
    st.markdown('<p class="section-label">Export & save</p>', unsafe_allow_html=True)

    export_rows = []
    for tbl, info in results.items():
        was_reassigned = tbl in user_overrides
        anon_cols = sum(1 for c in info.get("col_analysis", {}).values() if c["quality"] == "anonymous")
        total_c = len(info.get("col_analysis", {}))
        export_rows.append({
            "table_name": tbl,
            "domain": info["domain"],
            "auto_domain": info.get("auto_domain", info["domain"]),
            "manually_reassigned": "Yes" if was_reassigned else "No",
            "confidence_score": info["score"],
            "columns": "; ".join(info["columns"]),
            "anonymous_columns": f"{anon_cols}/{total_c}",
            "text_used_for_matching": info["text_used"],
        })
    export_df = pd.DataFrame(export_rows).sort_values(["domain", "table_name"])

    # Save run to SQLite
    with st.expander("Save this run to history"):
        run_col1, run_col2 = st.columns([3, 1])
        with run_col1:
            source_label = st.session_state.get("trino_conn_params", {})
            if source_label:
                default_run_name = f"{source_label.get('catalog','')}.{source_label.get('schema','')} — {datetime.now().strftime('%d %b %H:%M')}"
            else:
                default_run_name = f"Run — {datetime.now().strftime('%d %b %H:%M')}"
            run_name = st.text_input("Run name", value=default_run_name, label_visibility="collapsed")
        with run_col2:
            if st.button("Save", use_container_width=True, type="primary"):
                source_str = json.dumps(st.session_state.get("trino_conn_params", {"source": "file_upload"}))
                run_id = db_save_run(
                    run_name, source_str,
                    domain_labels_used, confidence_threshold,
                    results, user_overrides,
                )
                st.success(f"Saved as `{run_id}`")
                st.rerun()

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            "⬇ Download CSV",
            data=csv_data,
            file_name="domain_tagged_tables.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_dl2:
        json_export = {}
        for domain_name, tbls in grouped.items():
            json_export[domain_name] = [
                {
                    "table": t["table"],
                    "score": t["score"],
                    "reassigned": t["table"] in user_overrides,
                }
                for t in tbls
            ]
        st.download_button(
            "⬇ Download JSON",
            data=json.dumps(json_export, indent=2),
            file_name="domain_tagged_tables.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()