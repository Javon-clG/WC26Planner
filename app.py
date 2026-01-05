import os
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Optional
import uuid

# ---- Optional Google Sheets backend (Phase 2) ----
# Uses a Service Account JSON key.
# Set env vars:
#   WC26_SHEETS_ID = "<google sheet id>"
#   GOOGLE_APPLICATION_CREDENTIALS = "/path/to/service_account.json"
# Tabs expected in the Google Sheet (names are configurable below):
#   match_catalog, itineraries, itinerary_matches
#
# If Sheets config isn't present, the app falls back to local CSVs in ./data.

SHEETS_ID = os.getenv("WC26_SHEETS_ID", "").strip()
DATA_DIR = "data"

SHEET_TAB_CATALOG = os.getenv("WC26_TAB_CATALOG", "match_catalog")
SHEET_TAB_ITINERARIES = os.getenv("WC26_TAB_ITINERARIES", "itineraries")
SHEET_TAB_ITIN_MATCHES = os.getenv("WC26_TAB_ITIN_MATCHES", "itinerary_matches")

st.set_page_config(page_title="WC26 Planner POC", layout="wide")

# -------- Local CSV helpers --------
@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(f"{DATA_DIR}/{name}")

def save_csv(name: str, df: pd.DataFrame) -> None:
    df.to_csv(f"{DATA_DIR}/{name}", index=False)

# -------- Google Sheets helpers --------
def sheets_enabled() -> bool:
    return bool(SHEETS_ID)

@st.cache_resource
def _gs_client():
    import gspread
    # IMPORTANT: gspread.service_account() defaults to ~/.config/gspread/service_account.json unless a filename is provided.
    # We explicitly honor GOOGLE_APPLICATION_CREDENTIALS so local dev works predictably.
    keyfile = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if keyfile:
        return gspread.service_account(filename=keyfile)
    return gspread.service_account()

def _ws(tab_name: str):
    sh = _gs_client().open_by_key(SHEETS_ID)
    return sh.worksheet(tab_name)

@st.cache_data(ttl=600)  # 10 minutes
def read_sheet_df(tab_name: str) -> pd.DataFrame:
    ws = _ws(tab_name)
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    headers = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=headers)


def write_sheet_df(tab_name: str, df: pd.DataFrame) -> None:
    # Overwrites the entire tab (simple + reliable for Phase 2)
    ws = _ws(tab_name)
    df2 = df.copy()
    df2 = df2.replace({np.nan: ""})
    payload = [df2.columns.tolist()] + df2.astype(str).values.tolist()
    ws.clear()
    ws.update(payload, value_input_option="RAW")

def append_rows_aligned(tab_name: str, df_rows: pd.DataFrame) -> None:
    ws = _ws(tab_name)

    # Read header row from the sheet
    headers = ws.row_values(1)
    if not headers:
        raise ValueError(f"Sheet tab '{tab_name}' has no header row")

    df2 = df_rows.copy().replace({np.nan: ""})

    # Ensure all header columns exist in df (fill missing with "")
    for h in headers:
        if h not in df2.columns:
            df2[h] = ""

    # Reorder df columns to match sheet headers exactly
    df2 = df2[headers]

    ws.append_rows(df2.astype(str).values.tolist(), value_input_option="RAW")


# -------- Shared utilities --------
def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def normalize_ticket_category(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = s.replace("Category ", "Cat").replace("CATEGORY ", "Cat").replace("category ", "Cat")
    s = s.replace("CAT", "Cat")
    return s

def pick_unit_price(row) -> float:
    cat = str(row.get("ticket_category","")).strip().upper()
    if cat in ("CAT1","CATEGORY 1"):
        return row.get("cat1", np.nan)
    if cat in ("CAT2","CATEGORY 2"):
        return row.get("cat2", np.nan)
    if cat in ("CAT3","CATEGORY 3"):
        return row.get("cat3", np.nan)
    if cat in ("CAT4","CATEGORY 4"):
        return row.get("cat4", np.nan)
    return np.nan

def build_team_display(df: pd.DataFrame):
    home = df.get("home_team_name")
    away = df.get("away_team_name")
    if home is None: home = pd.Series([""] * len(df))
    if away is None: away = pd.Series([""] * len(df))

    home2 = home.fillna("").astype(str).str.strip()
    away2 = away.fillna("").astype(str).str.strip()

    hc = df.get("home_team_code")
    ac = df.get("away_team_code")
    if hc is None: hc = pd.Series([""] * len(df))
    if ac is None: ac = pd.Series([""] * len(df))

    hc2 = hc.fillna("").astype(str).str.strip()
    ac2 = ac.fillna("").astype(str).str.strip()

    home_disp = home2.where(home2 != "", hc2)
    away_disp = away2.where(away2 != "", ac2)
    home_disp = home_disp.where(home_disp != "", "TBD")
    away_disp = away_disp.where(away_disp != "", "TBD")
    return home_disp, away_disp

def validate_no_same_day(sel: pd.DataFrame):
    if sel.empty:
        return True, sel
    grp = sel.groupby("date_local")["match_id"].nunique().reset_index(name="n")
    bad_days = grp[grp["n"] > 1]["date_local"].tolist()
    offenders = sel[sel["date_local"].isin(bad_days)][["date_local","match_id","matchup","city_name","stage","time_local"]]\
        .sort_values(["date_local","time_local","match_id"])
    return len(bad_days) == 0, offenders

def compute_total(sel: pd.DataFrame):
    if sel.empty:
        return 0.0, sel
    sel = sel.copy()
    sel["quantity"] = pd.to_numeric(sel["quantity"], errors="coerce").fillna(1).astype(int)
    sel["ticket_category"] = sel["ticket_category"].astype(str).apply(normalize_ticket_category)
    sel["unit_price"] = sel.apply(pick_unit_price, axis=1)
    sel["row_cost"] = pd.to_numeric(sel["unit_price"], errors="coerce") * sel["quantity"]
    total = float(np.nansum(sel["row_cost"].values))
    return total, sel

def export_itinerary_csv(sel: pd.DataFrame, itinerary_id: str) -> bytes:
    out = sel.copy()
    out.insert(0, "itinerary_id", itinerary_id)
    return out.to_csv(index=False).encode("utf-8")

def load_catalog() -> pd.DataFrame:
    if sheets_enabled():
        df = read_sheet_df(SHEET_TAB_CATALOG)
    else:
        df = load_csv("match_catalog.csv")

    df = ensure_columns(df, [
        "match_id","stage","date_local","time_local","timezone","city_name",
        "home_team_name","away_team_name","home_team_code","away_team_code",
        "cat1","cat2","cat3","cat4","price_band"
    ])

    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").astype("Int64")
    for col in ["cat1","cat2","cat3","cat4"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date_local"] = df["date_local"].astype(str)
    df["time_local"] = df["time_local"].astype(str)
    df["home_display"], df["away_display"] = build_team_display(df)
    df["matchup"] = df["home_display"] + " vs " + df["away_display"]
    return df

def load_itineraries() -> pd.DataFrame:
    if sheets_enabled():
        df = read_sheet_df(SHEET_TAB_ITINERARIES)
    else:
        df = load_csv("itineraries.csv")
    df = ensure_columns(df, ["itinerary_id","owner_name","origin_iata","budget_usd"])
    df["budget_usd"] = pd.to_numeric(df["budget_usd"], errors="coerce")
    return df

def load_itinerary_matches() -> pd.DataFrame:
    if sheets_enabled():
        df = read_sheet_df(SHEET_TAB_ITIN_MATCHES)
    else:
        df = load_csv("itinerary_matches.csv")

    # Ensure expected columns exist
    df = ensure_columns(df, ["itinerary_id","match_id","ticket_category","quantity","priority","notes","saved_at","op","save_id"])

    # Normalize ids early (Sheets often returns blanks/NaN/whitespace)
    df["itinerary_id"] = df["itinerary_id"].fillna("").astype(str).str.strip()

    # Parse match_id numerically early so dedupe is consistent
    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").astype("Int64")

    # Parse saved_at (blank -> NaT)
    df["saved_at_dt"] = pd.to_datetime(df["saved_at"].fillna(""), errors="coerce")

    # Drop rows that can't be keyed
    df = df[(df["itinerary_id"] != "") & (df["match_id"].notna())].copy()

    # Sort so "latest wins"
    df = df.sort_values("saved_at_dt")

    # Keep latest record per itinerary + match
    df = df.drop_duplicates(subset=["itinerary_id","match_id"], keep="last")

    # Apply deletes if/when you start writing them
    if "op" in df.columns:
        df["op"] = df["op"].fillna("UPSERT").astype(str).str.upper()
        df = df[df["op"] != "DELETE"]

    # Final type normalization
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1).astype(int)
    df["priority"] = pd.to_numeric(df["priority"], errors="coerce").fillna(999).astype(int)

    return df.drop(columns=["saved_at_dt"], errors="ignore")


def save_itinerary_matches(all_rows: pd.DataFrame) -> None:
    cols = ["itinerary_id","match_id","ticket_category","quantity","priority","notes"]
    out = ensure_columns(all_rows.copy(), cols)[cols]
    if sheets_enabled():
        write_sheet_df(SHEET_TAB_ITIN_MATCHES, out)
    else:
        save_csv("itinerary_matches.csv", out)

def handle_itinerary_switch(itinerary_id: str):
    prev = st.session_state.get("active_itinerary_id")
    if prev != itinerary_id:
        st.session_state["active_itinerary_id"] = itinerary_id
        st.session_state["editor_nonce"] = st.session_state.get("editor_nonce", 0) + 1


# ---------------- UI ----------------
st.title("WC26 Planner")

catalog = load_catalog()
itineraries = load_itineraries()
all_itin_matches = load_itinerary_matches()
tmp_ids = itineraries.get("itinerary_id", pd.Series([], dtype="object")).dropna().astype(str).str.strip()
test_itinerary_id = (tmp_ids.iloc[0] if len(tmp_ids) > 0 else "it_001")

import datetime

if st.sidebar.button("Refresh from Sheets"):
    st.cache_data.clear()
    st.rerun()

if itineraries.empty:
    st.error("No itineraries found. Add rows in the itineraries tab (Google Sheet) or data/itineraries.csv.")
    st.stop()

#Create User
with st.sidebar.expander("Add a new user", expanded=False):
    new_name = st.text_input("Name (display)", placeholder="e.g., Mike")
    new_origin = st.text_input("Origin airport (IATA)", value="BOS", max_chars=3).upper().strip()
    new_budget = st.number_input("Budget (USD)", min_value=0, value=2000, step=100)

    if st.button("Create user", key="create_user"):
        if not new_name.strip():
            st.error("Name is required.")
            st.stop()

        tmp = itineraries.copy()
        tmp["itinerary_id"] = tmp["itinerary_id"].fillna("").astype(str).str.strip()
        nums = tmp["itinerary_id"].str.extract(r"it_(\d+)", expand=False)
        nums = pd.to_numeric(nums, errors="coerce")
        next_n = int(nums.max()) + 1 if nums.notna().any() else 1
        new_it_id = f"it_{next_n:03d}"

        from datetime import datetime, timezone
        row = pd.DataFrame([{
            "itinerary_id": new_it_id,
            "owner_name": new_name.strip(),
            "origin_iata": new_origin if new_origin else "BOS",
            "budget_usd": str(int(new_budget)),
            "created_at": datetime.now(timezone.utc).date().isoformat(),
        }])

        if len(new_origin) != 3 or not new_origin.isalpha():
            st.error("Origin airport must be a 3-letter IATA code (e.g., BOS).")
            st.stop()

        if (itineraries["owner_name"].fillna("").str.strip().str.lower() == new_name.strip().lower()).any():
            st.warning("That name already exists. Consider adding a last initial.")
            st.stop()


        append_rows_aligned(SHEET_TAB_ITINERARIES, row)
        st.success(f"Created {new_name} ({new_it_id}) ✅")
        st.cache_data.clear()
        st.rerun()


#Where you select the owner in a dropdown as exists in the Itinerary Matches tab
owner = st.sidebar.selectbox("Select itinerary (person)", itineraries["owner_name"].tolist())
row = itineraries[itineraries["owner_name"] == owner].iloc[0]
itinerary_id = str(row["itinerary_id"]).strip()
prev = st.session_state.get("active_itinerary_id")
if prev != itinerary_id:
    st.session_state["active_itinerary_id"] = itinerary_id
    st.session_state["editor_nonce"] = st.session_state.get("editor_nonce", 0) + 1

origin = row.get("origin_iata","")
budget = float(row.get("budget_usd") or 0)

# In-session selection buffer (we still use session_state for snappy UI)
if "selections" not in st.session_state:
    st.session_state["selections"] = {}

if itinerary_id not in st.session_state["selections"]:
    saved = all_itin_matches[all_itin_matches["itinerary_id"] == itinerary_id].copy()
    if not saved.empty:
        sel = saved.merge(catalog, on="match_id", how="left")
        sel = ensure_columns(sel, ["matchup"])
        st.session_state["selections"][itinerary_id] = sel
    else:
        st.session_state["selections"][itinerary_id] = pd.DataFrame()

tabs = st.tabs(["Browse matches", "My itinerary", "Compare (CSV)"])

# --- Browse ---
with tabs[0]:
    st.subheader("Browse matches")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        stage_filter = st.multiselect("Stage", sorted([s for s in catalog["stage"].dropna().unique() if str(s).strip()]), default=[])
    with c2:
        city_filter = st.multiselect("City", sorted(catalog["city_name"].dropna().unique().tolist()), default=[])
    with c3:
        price_band_filter = st.multiselect("Price band", ["LOW","MED","HIGH"], default=[])
    with c4:
        max_cat4 = st.number_input("Max Cat4 ($)", min_value=0.0, value=0.0, step=10.0, help="0 = no cap (filters using Cat4 if present)")
    with c5:
        all_teams = sorted(pd.unique(pd.concat([catalog["home_display"], catalog["away_display"]], ignore_index=True)).tolist())
        all_teams = [t for t in all_teams if isinstance(t, str) and t.strip() and t != "TBD"]
        team_filter = st.multiselect("Team", all_teams, default=[])

    df = catalog.copy()
    if stage_filter:
        df = df[df["stage"].isin(stage_filter)]
    if city_filter:
        df = df[df["city_name"].isin(city_filter)]
    if price_band_filter:
        df = df[df["price_band"].isin(price_band_filter)]
    if max_cat4 and max_cat4 > 0:
        df = df[pd.to_numeric(df["cat4"], errors="coerce") <= max_cat4]
    if team_filter:
        df = df[(df["home_display"].isin(team_filter)) | (df["away_display"].isin(team_filter))]

    df = df.sort_values(["match_id"])

    st.dataframe(df[[
        "match_id","matchup","stage","date_local","time_local","timezone","city_name",
        "cat1","cat2","cat3","cat4","price_band"
    ]], hide_index=True, use_container_width=True, height=420)

    add_cols = st.columns([2,2,2,2,3])
    with add_cols[0]:
        add_match_id = st.number_input("Add match_id", min_value=1, max_value=int(catalog["match_id"].max()), value=1, step=1)
    with add_cols[1]:
        add_cat = st.selectbox("Ticket category", ["Cat4","Cat3","Cat2","Cat1"])
    with add_cols[2]:
        add_qty = st.number_input("Qty", min_value=1, max_value=10, value=2, step=1)
    with add_cols[3]:
        add_priority = st.number_input("Priority", min_value=1, max_value=999, value=1, step=1)
    with add_cols[4]:
        add_notes = st.text_input("Notes (optional)", value="")

    if st.button("Add to itinerary"):
        sel = st.session_state["selections"][itinerary_id]
        match_row = catalog[catalog["match_id"] == int(add_match_id)]
        if match_row.empty:
            st.error("match_id not found.")
        else:
            new = match_row.copy()
            new["ticket_category"] = add_cat
            new["quantity"] = int(add_qty)
            new["priority"] = int(add_priority)
            new["notes"] = add_notes

            if not sel.empty and (sel["match_id"] == int(add_match_id)).any():
                sel = sel[sel["match_id"] != int(add_match_id)]
            sel = pd.concat([sel, new], ignore_index=True)

            ok, offenders = validate_no_same_day(sel)
            if not ok:
                st.error("Rule violation: can't select two matches on the same day. Add cancelled.")
                st.dataframe(offenders, use_container_width=True)
                sel = sel[sel["match_id"] != int(add_match_id)]
            else:
                st.success("Added.")
            st.session_state["selections"][itinerary_id] = sel

# --- My itinerary ---
with tabs[1]:
    st.subheader(f"My itinerary: {owner}  (Origin: {origin}, Budget: ${budget:,.0f})")

    sel = st.session_state["selections"][itinerary_id].copy()
    if st.button("Sync from Google Sheet"):
        st.cache_data.clear()          # force pull latest from Sheets (use sparingly)
        all_itin_matches = load_itinerary_matches()

        all_itin_matches["itinerary_id"] = all_itin_matches["itinerary_id"].fillna("").astype(str).str.strip()
        saved = all_itin_matches[all_itin_matches["itinerary_id"] == str(itinerary_id).strip()].copy()

        saved["match_id"] = pd.to_numeric(saved["match_id"], errors="coerce").astype("Int64")
        saved = saved.dropna(subset=["match_id"])

        sel = saved.merge(catalog, on="match_id", how="left")
        st.session_state["selections"][itinerary_id] = sel
        st.success(f"Synced {len(sel)} rows from Sheets.")
        st.rerun()


    if sel.empty:
        st.info("No matches selected yet. Add some from the Browse tab.")
    else:
        sel = ensure_columns(sel, ["ticket_category","quantity","priority","notes","matchup"])
        sel["quantity"] = pd.to_numeric(sel["quantity"], errors="coerce").fillna(1).astype(int)
        sel["priority"] = pd.to_numeric(sel["priority"], errors="coerce").fillna(999).astype(int)
        sel = sel.sort_values(["date_local","time_local","priority","match_id"])

        ok, offenders = validate_no_same_day(sel)
        total, cost_df = compute_total(sel)

        top = st.columns([2,2,2,4])
        top[0].metric("Matches", int(sel["match_id"].nunique()))
        top[1].metric("Total cost", f"${total:,.0f}")
        top[2].metric("Budget left", f"${(budget-total):,.0f}" if budget else "—")
        if not ok:
            top[3].error("Invalid: multiple matches on same day")
        elif budget and total > budget:
            top[3].warning("Over budget (not blocked)")

        editor_key = f"it_editor_{itinerary_id}_{st.session_state.get('editor_nonce', 0)}"

        edited = st.data_editor(
            cost_df[[
                "match_id","matchup","stage","date_local","time_local","city_name",
                "ticket_category","quantity","unit_price","row_cost","priority","notes"
            ]],
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key=editor_key,
            column_config={
                "ticket_category": st.column_config.SelectboxColumn(options=["Cat4","Cat3","Cat2","Cat1"]),
                "quantity": st.column_config.NumberColumn(min_value=1, step=1),
                "priority": st.column_config.NumberColumn(min_value=1, step=1),
            }
        )

        edited["match_id"] = pd.to_numeric(edited["match_id"], errors="coerce").astype("Int64")
        merged = edited.merge(catalog, on="match_id", how="left", suffixes=("","_x"))
        merged = ensure_columns(merged, ["ticket_category","quantity","priority","notes","matchup"])

        ok2, offenders2 = validate_no_same_day(merged)
        total2, cost_df2 = compute_total(merged)

        if not ok2:
            st.error("Same-day conflict after edits. Fix before exporting/saving.")
            st.dataframe(offenders2, use_container_width=True)

        st.session_state["selections"][itinerary_id] = merged


        if st.button("Save selections (shared)", key="fsave_"):
            try:
                source = st.session_state["selections"][itinerary_id].copy()
                source = ensure_columns(source, ["match_id","ticket_category","quantity","priority","notes"])
                current = source[["match_id","ticket_category","quantity","priority","notes"]].copy()
                current.insert(0, "itinerary_id", str(itinerary_id).strip())

                # normalize types for Sheets
                current["match_id"] = pd.to_numeric(current["match_id"], errors="coerce").astype("Int64")
                current = current.dropna(subset=["match_id"])
                current["match_id"] = current["match_id"].astype(int).astype(str)

                current["quantity"] = pd.to_numeric(current["quantity"], errors="coerce").fillna(1).astype(int).astype(str)
                current["priority"] = pd.to_numeric(current["priority"], errors="coerce").fillna(999).astype(int).astype(str)

                current["ticket_category"] = current["ticket_category"].astype(str)
                current["notes"] = current["notes"].fillna("").astype(str)

                from datetime import datetime, timezone
                current["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")


                #Doing Deletes
                # Pull latest backend state for THIS itinerary (deduped)
                backend = load_itinerary_matches().copy()
                backend["itinerary_id"] = backend["itinerary_id"].fillna("").astype(str).str.strip()
                backend = backend[backend["itinerary_id"] == str(itinerary_id).strip()]

                backend_ids = set(backend["match_id"].dropna().astype(int).tolist())

                # Current UI state
                ui_ids = set(pd.to_numeric(current["match_id"], errors="coerce").dropna().astype(int).tolist())

                removed_ids = backend_ids - ui_ids

                if removed_ids:
                    delete_df = pd.DataFrame([{
                        "itinerary_id": str(itinerary_id).strip(),
                        "match_id": str(mid),
                        "ticket_category": "",
                        "quantity": "",
                        "priority": "",
                        "notes": "deleted in app",
                        "op": "DELETE",
                        "save_id": str(uuid.uuid4()),
                        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    } for mid in sorted(removed_ids)])

                    append_rows_aligned(SHEET_TAB_ITIN_MATCHES, delete_df)
                # Checking for versioning 
                
                current["op"] = "UPSERT"
                current["save_id"] = str(uuid.uuid4())

                append_rows_aligned(SHEET_TAB_ITIN_MATCHES, current)
                st.success(f"Saved {len(current)} rows to Google Sheets ✅")

            except Exception as e:
                st.error("Save failed ❌")
                st.exception(e)

        st.download_button(
            label="Export this itinerary as CSV",
            data=export_itinerary_csv(cost_df2, itinerary_id),
            file_name=f"{owner}_itinerary_{itinerary_id}.csv",
            mime="text/csv",
            disabled=not ok2
        )
                # Persist to backend (Google Sheets or local CSV)



    if st.button("Reload from Sheets"):
        latest = load_itinerary_matches()  # this uses cached read; see note below
        latest["itinerary_id"] = latest["itinerary_id"].fillna("").astype(str).str.strip()

        saved = latest[latest["itinerary_id"] == str(itinerary_id).strip()].copy()
        #saved["match_id"] = pd.to_numeric(saved["match_id"], errors="coerce").astype("Int64")
        st.write("Raw saved rows from Sheets (before merge):")
        st.dataframe(saved, use_container_width=True)


        sel = saved.merge(catalog, on="match_id", how="left")
        st.session_state["selections"][itinerary_id] = sel
        st.success("Reloaded.")
        st.rerun()

# --- Compare ---
with tabs[2]:
    st.subheader("Compare itineraries")

    mode = st.radio("Comparison source", ["Shared (Google Sheet / backend)", "Upload CSV exports"], horizontal=True)

    if mode.startswith("Shared"):
    # Compare directly from backend itinerary_matches tab
        comb = all_itin_matches.merge(catalog, on="match_id", how="left")
        comb = ensure_columns(comb, ["ticket_category","quantity","matchup","stage","date_local","time_local","city_name","home_display","away_display"])

        # cost math
        comb["quantity"] = pd.to_numeric(comb["quantity"], errors="coerce").fillna(1).astype(int)
        comb["ticket_category"] = comb["ticket_category"].astype(str).apply(normalize_ticket_category)
        comb["unit_price"] = comb.apply(pick_unit_price, axis=1)
        comb["row_cost"] = pd.to_numeric(comb["unit_price"], errors="coerce") * comb["quantity"]

        # standardize IDs + join owner labels
        comb["itinerary_id"] = comb["itinerary_id"].fillna("").astype(str).str.strip()
        itineraries["itinerary_id"] = itineraries["itinerary_id"].fillna("").astype(str).str.strip()
        comb = comb.merge(itineraries[["itinerary_id","owner_name"]], on="itinerary_id", how="left")
        comb["who"] = comb["owner_name"].fillna(comb["itinerary_id"]).fillna("").astype(str)

        # normalize date_local early so groupby/min/max won't crash
        comb["date_local"] = comb["date_local"].fillna("").astype(str).str.strip()
        comb = comb[comb["date_local"] != ""]
        
        
        with st.expander("Filters", expanded=False):
            
            # ---------- Filters ----------
            # People filter (do this BEFORE summary/conflicts)
            people = sorted([p for p in comb["who"].dropna().astype(str).unique().tolist() if p.strip()])
            selected_people = st.multiselect("People", people, default=people)
            if selected_people:
                comb = comb[comb["who"].astype(str).isin(selected_people)]

            # City filter
            city_pool = sorted([c for c in comb["city_name"].dropna().astype(str).unique().tolist() if c.strip()])
            selected_cities = st.multiselect("Cities", city_pool, default=[])
            if selected_cities:
                comb = comb[comb["city_name"].astype(str).isin(selected_cities)]

            # Team filter (prefer home_display/away_display; fallback to matchup)
            if "home_display" in comb.columns and "away_display" in comb.columns and comb[["home_display","away_display"]].notna().any().any():
                team_pool = sorted(pd.unique(pd.concat([
                    comb["home_display"].fillna("").astype(str),
                    comb["away_display"].fillna("").astype(str),
                ])).tolist())
                team_pool = [t for t in team_pool if t.strip() and t.strip().upper() != "TBD"]
                selected_teams = st.multiselect("Teams", team_pool, default=[])
                if selected_teams:
                    comb = comb[
                        comb["home_display"].astype(str).isin(selected_teams) |
                        comb["away_display"].astype(str).isin(selected_teams)
                    ]
            else:
                mu = comb["matchup"].fillna("").astype(str)
                parts = mu.str.split(" vs ", n=1, expand=True)
                t1 = parts[0].fillna("")
                t2 = parts[1].fillna("") if parts.shape[1] > 1 else pd.Series([""] * len(mu))
                team_pool = sorted(pd.unique(pd.concat([t1, t2])).tolist())
                team_pool = [t for t in team_pool if t.strip() and t.strip().upper() != "TBD"]
                selected_teams = st.multiselect("Teams", team_pool, default=[])
                if selected_teams:
                    comb = comb[mu.apply(lambda x: any(t in x for t in selected_teams))]

            # (Optional) Stage filter
            stage_pool = sorted([s for s in comb["stage"].dropna().astype(str).unique().tolist() if s.strip()])
            selected_stages = st.multiselect("Stages", stage_pool, default=[])
            if selected_stages:
                comb = comb[comb["stage"].astype(str).isin(selected_stages)]

            # Date range filter (string YYYY-MM-DD; safe)
            # Date range filter (string YYYY-MM-DD) — use select_slider for strings
            date_vals = sorted([d for d in comb["date_local"].unique().tolist() if d])
            if date_vals:
                # Optional sanity: only keep YYYY-MM-DD like values
                date_vals = [d for d in date_vals if len(d) >= 10]

                date_range = st.select_slider(
                    "Date range (local)",
                    options=date_vals,
                    value=(date_vals[0], date_vals[-1]),
                )
                comb = comb[(comb["date_local"] >= date_range[0]) & (comb["date_local"] <= date_range[1])]
        st.markdown("Expand here to filter by person, cities, teams, or stage!")

        # If filters remove everything, stop cleanly
        if comb.empty:
            st.info("No rows match the current filters.")
        else:
            # ---------- Overlaps ----------
            over = comb.copy()

            st.markdown("#### Overlaps (same match)")
            overlaps_same_match = (
                over.groupby(["match_id", "matchup", "date_local", "city_name"], as_index=False)
                .agg(
                    people=("who", lambda s: ", ".join(sorted(set([x for x in s.dropna().astype(str) if x.strip()])))),
                    num_people=("who", lambda s: len(set([x for x in s.dropna().astype(str) if x.strip()]))),
                )
            )
            overlaps_same_match = overlaps_same_match[overlaps_same_match["num_people"] >= 2]\
                .sort_values(["date_local","city_name","match_id"])

            if overlaps_same_match.empty:
                st.caption("No shared matches found with current filters.")
            else:
                st.dataframe(overlaps_same_match, hide_index=True,use_container_width=True, height=300)

            # ---------- Summary (after filters) ----------
            st.markdown("#### Summary of friends")
            summary = comb.groupby("who", as_index=False).agg(
                matches=("match_id","nunique"),
                total_cost=("row_cost","sum"),
                first_date=("date_local","min"),
                last_date=("date_local","max"),
            ).sort_values("who")

            st.dataframe(summary, hide_index=True,use_container_width=True)

            # ---------- Same-day conflict check ----------
            st.markdown("#### Same-day conflict check")
            for who in summary["who"].tolist():
                sub = comb[comb["who"] == who].copy()
                ok, offenders = validate_no_same_day(sub)
                if ok:
                    st.success(f"{who}: no same-day conflicts ✅")
                else:
                    st.error(f"{who}: conflicts ❌")
                    st.dataframe(offenders, use_container_width=True)

            # ---------- Detailed rows ----------
            st.markdown("#### Detailed rows")
            st.dataframe(comb[[
                "who","match_id","matchup","stage","date_local","time_local","city_name",
                "ticket_category","quantity","unit_price","row_cost"
            ]].sort_values(["who","date_local","time_local","match_id"]), hide_index=True,use_container_width=True, height=420)
