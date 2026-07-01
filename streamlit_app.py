import os
import tempfile
from pathlib import Path

os.environ.setdefault("STREAMLIT_SERVER_MAX_UPLOAD_SIZE", "500")

import pandas as pd
import streamlit as st

from rank import TOP_K, rank_candidates, write_submission


try:
    st.set_option("server.maxUploadSize", 500)
except Exception:
    pass


st.set_page_config(page_title="Redrob Top-100 Ranker", layout="wide")
st.title("Redrob Candidate Ranker")

uploaded = st.file_uploader(
    "Upload candidates.jsonl, candidates.jsonl.gz, or sample_candidates.json",
    type=["jsonl", "json", "gz"],
    accept_multiple_files=False,
)

st.caption("Upload limit: 500 MB. The app shows and downloads only the top 100 ranks.")

if uploaded is not None:
    suffix = "".join(Path(uploaded.name).suffixes) or ".jsonl"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    try:
        with st.spinner("Ranking candidates offline..."):
            rows = rank_candidates(tmp_path, limit=TOP_K)

        visible_rows = [
            {
                "candidate_id": row["candidate_id"],
                "rank": row["rank"],
                "score": row["score"],
                "reasoning": row["reasoning"],
            }
            for row in rows[:TOP_K]
        ]
        df = pd.DataFrame(visible_rows, columns=["candidate_id", "rank", "score", "reasoning"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as out:
            out_path = out.name
        write_submission(rows, out_path)
        csv_bytes = Path(out_path).read_bytes()
        st.download_button(
            "Download top_100_submission.csv",
            data=csv_bytes,
            file_name="top_100_submission.csv",
            mime="text/csv",
        )
        Path(out_path).unlink(missing_ok=True)
    except Exception as exc:
        st.error(f"Ranking failed: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
else:
    st.info("Upload a candidate file to generate the top 100 ranked CSV.")
