# Redrob Top-100 Candidate Ranker

This project contains the complete offline ranking solution for the Redrob Senior AI Engineer JD.

## Files

- `rank.py` - dependency-free CLI ranker that emits the required CSV format.
- `streamlit_app.py` - sandbox/demo app using the same ranking code, with a 500 MB upload limit.
- `README.md` - this documentation file.

## Reproduce the submission CSV

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

The output uses exactly:

```text
candidate_id,rank,score,reasoning
```

## Run the Streamlit demo
```bash
streamlit run streamlit_app.py --server.maxUploadSize 500
```
The app accepts `.jsonl`, `.jsonl.gz`, or `.json` candidate files and displays only the top 100 candidates. It also downloads only the top-100 CSV.

## Methodology

The ranker is deterministic, CPU-only, and does not use network calls, hosted LLMs, GPUs, or external model files. It streams JSONL/GZIP inputs and keeps a bounded candidate heap, so the full 100k candidate file can be ranked within the challenge reproduction constraints.

The score is built from these weighted components:

- 22% title and career-title relevance for Senior AI, ML, search, recommendation, NLP, and applied ML roles.
- 25% career-history text evidence for retrieval, ranking, recommendation systems, vector search, embeddings, evaluation, production deployment, and recruiter/candidate matching.
- 17% trusted skills, where core JD skills are weighted by proficiency, duration, and endorsements instead of raw keyword count.
- 10% experience fit, centered on the JD's 5-9 year target while allowing strong adjacent candidates.
- 9% product-company/context evidence, favoring product, SaaS, AI/ML, marketplace, and user-facing systems over pure services profiles.
- 8% location/logistics fit for Pune/Noida, major Indian metros, relocation willingness, and hybrid/onsite compatibility.
- 4% education fit for CS, ML, data science, statistics, IT, and institution tier.
- 5% direct Redrob behavioral signal contribution.

Behavioral signals also act as an availability multiplier. Recent activity, recruiter response rate, response speed, notice period, interview completion, offer acceptance, GitHub activity, profile completeness, recruiter saves, profile views, verification, and open-to-work status down-weight candidates who look good on paper but are unlikely to engage.

## Honeypot and trap handling

The solution avoids the explicit trap in the JD: it does not reward AI keywords alone. Candidates with many AI skills but non-AI titles and weak career-history evidence are penalized. Additional anomaly checks catch likely honeypots:

- expert skills with zero months of usage;
- too many advanced/expert claims for low total experience;
- AI keyword stuffing without matching production retrieval/ranking work;
- internally inconsistent salary ranges;
- profile experience that diverges materially from career durations;
- mostly services/consulting careers without product/retrieval evidence;
- research, computer vision, speech, or robotics profiles without NLP/IR/search evidence;
- very short-tenure title-chasing patterns.

Reasoning strings are generated from facts in each candidate profile: title, experience, the strongest matching career-history role, company, matched retrieval/ranking evidence, skill proficiency/duration, location fit, response rate, notice period, interview/recruiter activity, and the top concern when present. The wording varies by candidate so the output reads like actual ranking judgment rather than a fixed template.
