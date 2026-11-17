# AutoClean Agents

--> Can a multi-agent system replicate the data cleaning decisions a data scientist makes manually?

A research project exploring agentic AI for automated data quality assessment and correction, applied to real-world railway punctuality data (SNCF TGV dataset, 15,424 rows × 26 columns).

---

## Motivation

During an internship working with railway operational data, a significant portion of analysis time was spent on manual data cleaning — detecting missing values, fixing type inconsistencies, handling outliers. This project asks: can a pipeline of collaborative AI agents automate these decisions, and how do we measure whether they got it right?

---

## Architecture

Three agents collaborate in sequence:

[ Profiler Agent ]  →  [ Planner Agent ]  →  [ Validator Agent ]
  reads raw CSV         writes pandas          executes code,
  flags issues          correction code        measures improvement
  builds report         from report            vs ground truth


- **Profiler**: detects missing values, type corruption, outliers, duplicates, and cause-percentage drift
- **Planner**: receives the structured quality report and generates Python correction code autonomously
- **Validator**: runs the code in a sandbox, computes recall and precision against a ground truth file

Each agent runs a **ReAct loop** (Reason → Act → Observe → repeat) powered by `llama-3.1-70b-versatile` via Groq API.

---

## Evaluation Methodology

Before running the agents, five types of noise are deliberately injected into the dataset with a fixed seed (`seed=42`):

| Noise Type | Columns affected | Rate |
|---|---|---|
| Missing values | retard_moyen_depart, retard_moyen_arrivee, nb_train_retard_sup_15 | 6% |
| Type corruption | nb_train_prevu, nb_annulation | 4% |
| Outliers (5× p99) | retard_moyen_depart, nb_train_retard_sup_60 | 2% |
| Duplicate rows | — | 1.5% |
| Cause % drift | 6 prct_cause columns | 5% |

The exact location of every injected error is saved to `results/noise_ground_truth.json`. This allows the Validator to compute **recall** (errors found / errors injected) and **precision** (correct fixes / total fixes attempted) against a known answer key.

---

## Results

| Metric | Value |
|---|---|
| Missing values recall | 0.84 |
| Type corruption recall | 0.91 |
| Outlier recall | 0.78 |
| Duplicate recall | 1.00 |
| Overall precision | 0.87 |

*Full metrics available in `results/metrics.json`*

---

## Stack

- **Agents**: CrewAI + LangChain
- **LLM**: llama-3.1-70b-versatile via Groq API (free tier)
- **Data**: SNCF TGV monthly regularity dataset (open data)
- **Evaluation**: custom recall/precision vs injected ground truth
- **Tests**: pytest (4 passing, 1 intentional failure — IQR sensitivity on small dataframes)

---

## How to Run

```bash
# 1. Clone and install
git clone https://github.com/yourname/autoclean-agents
cd autoclean-agents
pip install -r requirements.txt

# 2. Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Inject noise into the dataset
python tools/noise_injection.py

# 4. Run the full pipeline
python main.py

# 5. View results
cat results/metrics.json
```

---

## Project Status

Work in progress — pipeline complete, evaluation methodology implemented, README and results finalized.

---

## Author

Zyad Larabi — ENSEIRB-MATMECA, Bordeaux INP