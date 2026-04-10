from __future__ import annotations

from pathlib import Path
from typing import Optional
import random

import pandas as pd
import streamlit as st
from supabase import create_client

DEFAULT_DATA_DIR = "."
DEFAULT_VOTES_CSV = "blind_votes.csv"

BLIND_FIELDS = [
    "PA", "WAR", "Scaled WAR", "WAR/PA", "Scaled WAR/PA",
    "BA", "OBP", "SLG", "OPS",
    "HR", "3B", "2B", "1B", "BB", "OUTS",
    "Out Rate", "BB Rate", "HR Rate", "XBH", "XBH Rate",
    "Best Top 2 Overall Score",
    "Best 2 Film Run Score",
    "Best 3 Film Run Score",
    "Best 5 Film Run Score",
]

REVEAL_FIELDS = [
    "Director", "Eligible", "Rank", "Strength", "Expected Wins", "Expected Win %",
    "Career_Value", "Rate_Quality", "Peak", "Profile",
    "PA", "WAR", "Scaled WAR", "WAR/PA", "Scaled WAR/PA",
    "BA", "OBP", "SLG", "OPS",
    "HR", "3B", "2B", "1B", "BB", "OUTS",
    "Out Rate", "BB Rate", "HR Rate", "XBH", "XBH Rate",
    "Best Top 2 Overall Score",
    "Best 2 Film Run Score", "Best 2 Film Run Titles",
    "Best 3 Film Run Score", "Best 3 Film Run Titles",
    "Best 5 Film Run Score", "Best 5 Film Run Titles",
]


def normalize_name(name: str) -> str:
    return str(name).strip().lower()


@st.cache_resource
def get_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"],
    )


@st.cache_data(show_spinner=False)
def load_data(data_dir: str, all_directors: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir)
    cards_name = "phase2_optionA_resume_cards_all_directors.csv" if all_directors else "phase2_optionA_resume_cards.csv"
    cards_path = data_path / cards_name
    pairwise_path = data_path / "phase2_optionA_pairwise_matrix.csv"

    if not cards_path.exists():
        raise FileNotFoundError(f"Could not find {cards_path}")
    if not pairwise_path.exists():
        raise FileNotFoundError(f"Could not find {pairwise_path}")

    cards = pd.read_csv(cards_path)
    pairwise = pd.read_csv(pairwise_path)
    cards["_norm"] = cards["Director"].map(normalize_name)
    return cards, pairwise


@st.cache_data(show_spinner=False)
def load_votes(votes_csv: str) -> pd.DataFrame:
    columns = [
        "session", "resume_a_director", "resume_b_director", "choice",
        "notes", "timestamp_utc"
    ]
    try:
        supabase = get_supabase()
        response = (
            supabase.table("blind_votes")
            .select("session_id,director_a,director_b,choice,notes,created_at")
            .order("created_at")
            .execute()
        )
        data = response.data or []
        if not data:
            return pd.DataFrame(columns=columns)
        df = pd.DataFrame(data).rename(columns={
            "session_id": "session",
            "director_a": "resume_a_director",
            "director_b": "resume_b_director",
            "created_at": "timestamp_utc",
        })
        for col in columns:
            if col not in df.columns:
                df[col] = ""
        return df[columns]
    except Exception as e:
        st.error(f"Could not load votes from Supabase: {e}")
        return pd.DataFrame(columns=columns)


def save_vote(votes_csv: str, a_name: str, b_name: str, choice: str, notes: str = "", session: str = "default") -> None:
    choice = choice.strip().upper()
    if choice not in {"A", "B", "T"}:
        raise ValueError("choice must be A, B, or T")

    payload = {
        "session_id": session,
        "director_a": a_name,
        "director_b": b_name,
        "choice": choice,
        "notes": notes,
    }

    supabase = get_supabase()
    supabase.table("blind_votes").insert(payload).execute()
    load_votes.clear()



def row_by_name(cards: pd.DataFrame, name: str) -> pd.Series:
    norm = normalize_name(name)
    matches = cards[cards["_norm"] == norm]
    if matches.empty:
        matches = cards[cards["Director"].str.lower().str.contains(norm, na=False)]
    if matches.empty:
        raise ValueError(f"Director not found: {name}")
    if len(matches) > 1:
        raise ValueError(f"Multiple directors matched: {name}. Be more specific.")
    return matches.iloc[0]



def format_value(v) -> str:
    if pd.isna(v):
        return "—"
    if isinstance(v, (int,)) or (hasattr(v, "is_integer") and callable(v.is_integer) and v.is_integer()):
        return str(int(v))
    if isinstance(v, float):
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) >= 10:
            return f"{v:.2f}"
        return f"{v:.3f}"
    return str(v)



def model_view(cards: pd.DataFrame, pairwise: pd.DataFrame, a_name: str, b_name: str) -> dict:
    a = row_by_name(cards, a_name)
    b = row_by_name(cards, b_name)

    p = pairwise
    mask_ab = (p["Director A"] == a["Director"]) & (p["Director B"] == b["Director"])
    mask_ba = (p["Director A"] == b["Director"]) & (p["Director B"] == a["Director"])

    if mask_ab.any():
        row = p.loc[mask_ab].iloc[0]
        a_prob = float(row["P(A beats B)"])
        b_prob = float(row["P(B beats A)"])
        bucket_map = {
            "Career": float(row.get("Career Diff", 0)),
            "Rate": float(row.get("Rate Diff", 0)),
            "Peak": float(row.get("Peak Diff", 0)),
            "Profile": float(row.get("Profile Diff", 0)),
        }
    elif mask_ba.any():
        row = p.loc[mask_ba].iloc[0]
        a_prob = float(row["P(B beats A)"])
        b_prob = float(row["P(A beats B)"])
        bucket_map = {
            "Career": -float(row.get("Career Diff", 0)),
            "Rate": -float(row.get("Rate Diff", 0)),
            "Peak": -float(row.get("Peak Diff", 0)),
            "Profile": -float(row.get("Profile Diff", 0)),
        }
    else:
        raise ValueError("Pairwise row not found for matchup.")

    return {
        "a": a,
        "b": b,
        "a_prob": a_prob,
        "b_prob": b_prob,
        "favored": a["Director"] if a_prob >= b_prob else b["Director"],
        "bucket_map": bucket_map,
    }



def pick_matchup(
    cards: pd.DataFrame,
    votes: pd.DataFrame,
    eligible_only: bool,
    min_pa: int,
    max_pa_gap: Optional[int],
    strength_gap_max: Optional[float],
    rng: random.Random,
) -> tuple[str, str]:
    pool = cards.copy()
    if eligible_only and "Eligible" in pool.columns:
        pool = pool[pool["Eligible"] == True].copy()
    if "PA" in pool.columns:
        pool = pool[pool["PA"] >= min_pa].copy()

    if len(pool) < 2:
        raise ValueError("Not enough directors after filtering.")

    used = set()
    if len(votes) > 0:
        for _, r in votes.iterrows():
            used.add(tuple(sorted([str(r["resume_a_director"]), str(r["resume_b_director"])])))

    idxs = list(pool.index)
    for _ in range(10000):
        i, j = rng.sample(idxs, 2)
        a = pool.loc[i]
        b = pool.loc[j]
        key = tuple(sorted([a["Director"], b["Director"]]))
        if key in used:
            continue
        if max_pa_gap is not None and abs(float(a["PA"]) - float(b["PA"])) > max_pa_gap:
            continue
        if strength_gap_max is not None and pd.notna(a.get("Strength")) and pd.notna(b.get("Strength")):
            if abs(float(a["Strength"]) - float(b["Strength"])) > strength_gap_max:
                continue
        if rng.random() < 0.5:
            return a["Director"], b["Director"]
        return b["Director"], a["Director"]
    raise ValueError("Could not find a fresh matchup that satisfies the filters.")



def blind_table(cards: pd.DataFrame, a_name: str, b_name: str) -> pd.DataFrame:
    a = row_by_name(cards, a_name)
    b = row_by_name(cards, b_name)
    rows = []
    for field in BLIND_FIELDS:
        if field in cards.columns:
            rows.append({
                "Metric": field,
                "Resume A": format_value(a.get(field)),
                "Resume B": format_value(b.get(field)),
            })
    return pd.DataFrame(rows)



def reveal_table(cards: pd.DataFrame, a_name: str, b_name: str) -> pd.DataFrame:
    a = row_by_name(cards, a_name)
    b = row_by_name(cards, b_name)
    rows = []
    for field in REVEAL_FIELDS:
        if field in cards.columns:
            rows.append({
                "Field": field,
                "Resume A": format_value(a.get(field)),
                "Resume B": format_value(b.get(field)),
            })
    return pd.DataFrame(rows)



def vote_agreement_summary(pairwise: pd.DataFrame, votes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    p = pairwise
    for _, r in votes.iterrows():
        a = str(r["resume_a_director"])
        b = str(r["resume_b_director"])
        choice = str(r["choice"]).upper()
        mask_ab = (p["Director A"] == a) & (p["Director B"] == b)
        mask_ba = (p["Director A"] == b) & (p["Director B"] == a)
        if mask_ab.any():
            row = p.loc[mask_ab].iloc[0]
            pa = float(row["P(A beats B)"])
            pb = float(row["P(B beats A)"])
        elif mask_ba.any():
            row = p.loc[mask_ba].iloc[0]
            pa = float(row["P(B beats A)"])
            pb = float(row["P(A beats B)"])
        else:
            continue
        model_choice = "A" if pa > pb else "B" if pb > pa else "T"
        rows.append({
            "resume_a_director": a,
            "resume_b_director": b,
            "choice": choice,
            "model_choice": model_choice,
            "agrees_with_model": choice == model_choice,
            "model_prob_for_user_choice": pa if choice == "A" else pb if choice == "B" else 0.5,
        })
    return pd.DataFrame(rows)



def bucket_edge_label(v: float) -> str:
    if abs(v) < 0.05:
        return "close"
    if abs(v) < 0.20:
        return "slight A edge" if v > 0 else "slight B edge"
    return "A edge" if v > 0 else "B edge"


def main() -> None:
    st.set_page_config(page_title="Blind Director Resume Battles", layout="wide")
    st.title("Blind Director Résumé Battles")
    st.caption("Blind mode hides names and movie titles, but keeps the run scores visible.")

    with st.sidebar:
        st.header("Settings")
        data_dir = st.text_input("Data folder", value=DEFAULT_DATA_DIR)
        votes_csv = st.text_input("Votes CSV (unused with Supabase)", value=DEFAULT_VOTES_CSV)
        all_directors = st.checkbox("Use all-directors cards", value=True)
        eligible_only = st.checkbox("Eligible directors only", value=True)
        min_pa = st.number_input("Minimum films (PA)", min_value=1, max_value=200, value=1, step=1)
        max_pa_gap_val = st.number_input("Maximum filmography gap", min_value=0, max_value=200, value=8, step=1)
        strength_gap_val = st.slider("Maximum strength gap", min_value=0.1, max_value=3.0, value=0.75, step=0.05)
        session_name = st.text_input("Session name", value="default")
        random_seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
        st.caption("Use 0 PA gap or 0 strength gap style extremes by widening the filters instead of narrowing them too much.")

    try:
        cards, pairwise = load_data(data_dir, all_directors=all_directors)
    except Exception as e:
        st.error(str(e))
        st.stop()

    votes = load_votes(votes_csv)

    if "battle_meta" not in st.session_state:
        st.session_state.battle_meta = None
    if "revealed" not in st.session_state:
        st.session_state.revealed = False
    if "vote_saved" not in st.session_state:
        st.session_state.vote_saved = False
    if "choice" not in st.session_state:
        st.session_state.choice = "A"

    cols_top = st.columns([1, 1, 1, 1])
    with cols_top[0]:
        if st.button("Next battle", type="primary"):
            try:
                rng = random.Random(int(random_seed) + len(votes))
                a_name, b_name = pick_matchup(
                    cards,
                    votes,
                    eligible_only=eligible_only,
                    min_pa=int(min_pa),
                    max_pa_gap=int(max_pa_gap_val) if max_pa_gap_val > 0 else None,
                    strength_gap_max=float(strength_gap_val) if strength_gap_val > 0 else None,
                    rng=rng,
                )
                st.session_state.battle_meta = {"A": a_name, "B": b_name}
                st.session_state.revealed = False
                st.session_state.vote_saved = False
                st.session_state.choice = "A"
            except Exception as e:
                st.error(str(e))

    if st.session_state.battle_meta is None:
        st.info("Click **Next battle** to generate a blind matchup.")
        st.stop()

    a_name = st.session_state.battle_meta["A"]
    b_name = st.session_state.battle_meta["B"]

    st.subheader("Blind matchup")
    blind_df = blind_table(cards, a_name, b_name)
    st.dataframe(blind_df, use_container_width=True, hide_index=True)

    pick_col, note_col = st.columns([1, 2])
    with pick_col:
        choice = st.radio("Your pick", options=["A", "B", "T"], format_func=lambda x: {"A": "Resume A", "B": "Resume B", "T": "Toss-up"}[x], key="choice")
    with note_col:
        notes = st.text_input("Notes", value="", placeholder="Why did you choose that side?")

    act1, act2, act3 = st.columns([1, 1, 1])
    with act1:
        if st.button("Save vote"):
            if not st.session_state.vote_saved:
                save_vote(votes_csv, a_name, b_name, st.session_state.choice, notes=notes, session=session_name)
                st.session_state.vote_saved = True
                st.success("Vote saved.")
            else:
                st.info("Vote already saved for this battle.")
    with act2:
        if st.button("Reveal"):
            st.session_state.revealed = True
    with act3:
        if st.button("Skip without voting"):
            st.session_state.battle_meta = None
            st.session_state.revealed = False
            st.session_state.vote_saved = False
            st.rerun()

    if st.session_state.revealed:
        mv = model_view(cards, pairwise, a_name, b_name)
        st.subheader("Reveal")
        st.markdown(f"**Resume A** = {a_name}  ")
        st.markdown(f"**Resume B** = {b_name}")
        st.markdown(f"**Favored by model:** {mv['favored']}")
        st.markdown(f"**Resume A win probability:** {mv['a_prob']:.1%} | **Resume B win probability:** {mv['b_prob']:.1%}")

        bucket_rows = [{
            "Bucket": k,
            "Diff (A-B)": round(v, 3),
            "Edge": bucket_edge_label(v),
        } for k, v in mv["bucket_map"].items()]
        st.dataframe(pd.DataFrame(bucket_rows), use_container_width=True, hide_index=True)

        reveal_df = reveal_table(cards, a_name, b_name)
        st.dataframe(reveal_df, use_container_width=True, hide_index=True)

        with st.expander("Show run titles"):
            a = mv["a"]
            b = mv["b"]
            titles = pd.DataFrame([
                {"Run": "Best 2 Film Run Titles", "Resume A": a.get("Best 2 Film Run Titles", "—"), "Resume B": b.get("Best 2 Film Run Titles", "—")},
                {"Run": "Best 3 Film Run Titles", "Resume A": a.get("Best 3 Film Run Titles", "—"), "Resume B": b.get("Best 3 Film Run Titles", "—")},
                {"Run": "Best 5 Film Run Titles", "Resume A": a.get("Best 5 Film Run Titles", "—"), "Resume B": b.get("Best 5 Film Run Titles", "—")},
            ])
            st.dataframe(titles, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Vote stats")
    stat1, stat2, stat3 = st.columns(3)
    stat1.metric("Votes logged", len(votes))
    if len(votes) > 0:
        agreement = vote_agreement_summary(pairwise, votes)
        if len(agreement) > 0:
            stat2.metric("Agreement with model", f"{agreement['agrees_with_model'].mean():.1%}")
            stat3.metric("Avg model confidence for your pick", f"{agreement['model_prob_for_user_choice'].mean():.1%}")
            with st.expander("Recent votes"):
                st.dataframe(votes.tail(20), use_container_width=True, hide_index=True)
        else:
            stat2.metric("Agreement with model", "—")
            stat3.metric("Avg model confidence for your pick", "—")
    else:
        stat2.metric("Agreement with model", "—")
        stat3.metric("Avg model confidence for your pick", "—")

    st.caption(
        "This build reads and writes votes through Supabase for durable shared storage."
    )


if __name__ == "__main__":
    main()
