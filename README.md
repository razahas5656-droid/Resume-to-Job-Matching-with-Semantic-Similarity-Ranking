# 
Resume-to-Job Matching with Semantic Similarity Ranking

M-Tech Internship Program 2026 — Week 6 Task
Name: Muhammad Hasnain Raza | Reg No: Mtech-DS26034
Concept: NLP / Semantic Search / Ranking | GUI Tool: Tkinter


What It Does

This app ranks resumes against job descriptions using transformer sentence
embeddings (all-MiniLM-L6-v2, via sentence-transformers) — NOT keyword
matching. Each resume and job description is converted into a dense vector
that captures its meaning, and resumes are ranked for a job by cosine
similarity of these embeddings. The rankings are then validated against
HR-labeled "good match" pairs using standard ranking metrics: Precision@K
and Mean Average Precision (MAP).


Why Semantic, Not Keyword Matching?

Keyword matching would miss a resume that says "built ML models with scikit-learn"
against a job asking for "machine learning experience" if the exact words don't
overlap. Transformer embeddings place semantically similar text close together
in vector space even without shared keywords, which is what makes this a much
more realistic resume-screening approach.


Files


generate_data.py — creates resumes.csv, job_descriptions.csv, and hr_labels.csv (synthetic HR-labeled ground truth of good matches, used only to validate the system's output — never used to build the rankings)

matching_pipeline.py — the core embedding + ranking + evaluation logic

matching_app.py — the Tkinter GUI

resumes.csv, job_descriptions.csv, hr_labels.csv — the dataset


How to Run


Install requirements:

pip install pandas numpy scikit-learn sentence-transformers
sentence-transformers is a larger install (500MB with PyTorch) and its first use downloads a small (80MB) pretrained model — you need internet the first time you run this. After that, the model is cached locally.


Generate the dataset:

python generate_data.py

Launch the GUI:

python matching_app.py

Click "Load Data & Build Embeddings" (takes a bit longer the first time while the model downloads). Then:

Pick a job from the dropdown to see its top-10 ranked resumes with similarity scores (Ranked Resumes tab)
Check the Validation vs HR Labels tab to see Precision@5 and MAP for every job, proving the semantic rankings agree with what HR considers a good match


Note on Fallback Behavior

If the transformer model can't be downloaded (no internet), the app
automatically falls back to TF-IDF (keyword-based) embeddings so it still
runs, and prints a warning telling you to check your connection. For the
task to actually demonstrate semantic (not keyword) matching, make sure
you have internet on first run. 


How AI Was Used

I used Claude to help me understand semantic similarity search with
transformer embeddings (vs. keyword matching), how to validate a ranking
system against labeled ground truth using Precision@K and MAP, and to
structure the pipeline and Tkinter GUI. I reviewed the code myself,
especially the evaluation metrics, to understand how they measure ranking
quality rather than plain classification accuracy.
 
 Generate Dataset:

import random
import pandas as pd

random.seed(42)

ROLES = {
    "Data Scientist": {
        "skills": ["Python", "machine learning", "pandas", "scikit-learn", "SQL",
                   "statistics", "data visualization", "deep learning", "NLP"],
        "resume_templates": [
            "Data scientist with {yrs} years of experience building {skill1} and {skill2} "
            "models in Python. Skilled in {skill3} and {skill4}, with a strong background "
            "in statistics and experimentation.",
            "Experienced in {skill1}, {skill2}, and {skill3}. Built predictive models using "
            "{skill4} and deployed them for business decision-making. {yrs} years in analytics.",
        ]
    },
    "Frontend Developer": {
        "skills": ["React", "JavaScript", "CSS", "HTML", "TypeScript",
                   "responsive design", "REST APIs", "Redux", "UI/UX"],
        "resume_templates": [
            "Frontend developer with {yrs} years building interactive web apps using {skill1} "
            "and {skill2}. Comfortable with {skill3} and {skill4}, focused on clean UI.",
            "Built and maintained production web interfaces with {skill1}, {skill2}, and "
            "{skill3}. {yrs} years of experience collaborating with designers and backend teams.",
        ]
    },
    "DevOps Engineer": {
        "skills": ["AWS", "Docker", "Kubernetes", "CI/CD", "Terraform",
                   "Linux", "monitoring", "Jenkins", "cloud infrastructure"],
        "resume_templates": [
            "DevOps engineer with {yrs} years managing {skill1} and {skill2} pipelines. "
            "Strong background in {skill3} and {skill4} for scalable cloud infrastructure.",
            "Automated deployments using {skill1}, {skill2}, and {skill3}. {yrs} years "
            "running production systems on {skill4}.",
        ]
    },
    "Marketing Analyst": {
        "skills": ["Google Analytics", "SEO", "A/B testing", "content strategy",
                   "campaign management", "Excel", "social media marketing", "email marketing"],
        "resume_templates": [
            "Marketing analyst with {yrs} years running {skill1} and {skill2} campaigns. "
            "Skilled in {skill3} and {skill4} to drive growth.",
            "Analyzed campaign performance using {skill1}, {skill2}, and {skill3}. {yrs} "
            "years of experience in digital marketing and {skill4}.",
        ]
    },
}

JOB_TEMPLATES = [
    "We are hiring a {role} to join our team. Required skills: {skill1}, {skill2}, and "
    "{skill3}. {yrs}+ years of experience preferred. You will work on {skill4} projects.",
    "Looking for an experienced {role} with strong knowledge of {skill1} and {skill2}. "
    "Familiarity with {skill3} is a plus. Minimum {yrs} years of relevant experience.",
]


def make_resume(role):
    info = ROLES[role]
    skills = random.sample(info["skills"], 4)
    yrs = random.randint(1, 8)
    template = random.choice(info["resume_templates"])
    return template.format(yrs=yrs, skill1=skills[0], skill2=skills[1], skill3=skills[2], skill4=skills[3])


def make_job(role):
    info = ROLES[role]
    skills = random.sample(info["skills"], 4)
    yrs = random.randint(1, 5)
    template = random.choice(JOB_TEMPLATES)
    return template.format(role=role, yrs=yrs, skill1=skills[0], skill2=skills[1], skill3=skills[2], skill4=skills[3])


def main():
    roles = list(ROLES.keys())

    # 40 resumes, 10 per role
    resumes = []
    for role in roles:
        for i in range(10):
            resumes.append({
                "resume_id": f"R{len(resumes)+1:03d}",
                "true_role": role,   # kept for evaluation only, NOT used by the matcher
                "resume_text": make_resume(role)
            })
    resumes_df = pd.DataFrame(resumes)

    # 12 jobs, 3 per role
    jobs = []
    for role in roles:
        for i in range(3):
            jobs.append({
                "job_id": f"J{len(jobs)+1:03d}",
                "role": role,
                "job_text": make_job(role)
            })
    jobs_df = pd.DataFrame(jobs)

    # HR ground truth: a resume is a "good match" for a job if same role
    # (this simulates HR manually labeling matches -- used only to VALIDATE
    # the semantic ranking system's output, never to build the ranking itself)
    hr_labels = []
    for _, job in jobs_df.iterrows():
        for _, resume in resumes_df.iterrows():
            good_match = int(resume["true_role"] == job["role"])
            hr_labels.append({
                "job_id": job["job_id"],
                "resume_id": resume["resume_id"],
                "hr_good_match": good_match
            })
    hr_labels_df = pd.DataFrame(hr_labels)

    resumes_df.to_csv("resumes.csv", index=False)
    jobs_df.to_csv("job_descriptions.csv", index=False)
    hr_labels_df.to_csv("hr_labels.csv", index=False)

    print(f"Created {len(resumes_df)} resumes, {len(jobs_df)} jobs, "
          f"{len(hr_labels_df)} HR-labeled resume-job pairs.")


if __name__ == "__main__":
    main()

Matching Pipeline:

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good general-purpose sentence embedding model


def get_embedder():
    """ Returns a function text_list -> np.ndarray of embeddings. Uses sentence-transformers (real transformer embeddings) if available and downloadable. Falls back to TF-IDF (keyword-based) ONLY if the transformer model can't be loaded (e.g. no internet on first run), so the app still runs -- but transformer embeddings are what actually fulfills this task's "semantic, not keyword matching" requirement. """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)

        def embed(texts):
            return model.encode(list(texts), show_progress_bar=False)

        return embed, "transformer (all-MiniLM-L6-v2)"
    except Exception as e:
        print(f"[Warning] Could not load transformer model ({e}). "
              f"Falling back to TF-IDF embeddings. Check your internet connection "
              f"and re-run for true semantic (transformer) matching.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words="english")

        def embed(texts):
            texts = list(texts)
            if not hasattr(embed, "_fitted"):
                vectorizer.fit(texts)
                embed._fitted = True
            return vectorizer.transform(texts).toarray()

        return embed, "TF-IDF (fallback)"


def build_similarity_matrix(resumes_df, jobs_df, embed_fn):
    """Returns a (n_jobs x n_resumes) cosine similarity matrix."""
    resume_emb = embed_fn(resumes_df["resume_text"])
    job_emb = embed_fn(jobs_df["job_text"])
    sim_matrix = cosine_similarity(job_emb, resume_emb)
    return sim_matrix  # rows = jobs, cols = resumes


def rank_resumes_for_job(sim_matrix, job_idx, resumes_df, top_k=10):
    """Returns resumes ranked by similarity for a given job index."""
    scores = sim_matrix[job_idx]
    ranked_idx = np.argsort(-scores)[:top_k]
    result = resumes_df.iloc[ranked_idx].copy()
    result["similarity_score"] = scores[ranked_idx]
    return result.reset_index(drop=True)


# ---------------- Evaluation against HR-labeled ground truth ----------------

def precision_at_k(ranked_resume_ids, hr_good_set, k):
    top_k_ids = ranked_resume_ids[:k]
    if len(top_k_ids) == 0:
        return 0.0
    hits = sum(1 for rid in top_k_ids if rid in hr_good_set)
    return hits / len(top_k_ids)


def average_precision(ranked_resume_ids, hr_good_set):
    if len(hr_good_set) == 0:
        return 0.0
    hits, score = 0, 0.0
    for i, rid in enumerate(ranked_resume_ids, start=1):
        if rid in hr_good_set:
            hits += 1
            score += hits / i
    return score / len(hr_good_set) if hits > 0 else 0.0


def evaluate_against_hr_labels(sim_matrix, jobs_df, resumes_df, hr_labels_df, k=5):
    """Computes Precision@K and Mean Average Precision (MAP) per job, validated against HR-labeled good matches."""
    rows = []
    for j_idx, job in jobs_df.iterrows():
        scores = sim_matrix[j_idx]
        ranked_order = np.argsort(-scores)
        ranked_resume_ids = resumes_df.iloc[ranked_order]["resume_id"].tolist()

        good = hr_labels_df[(hr_labels_df["job_id"] == job["job_id"]) &
                             (hr_labels_df["hr_good_match"] == 1)]
        hr_good_set = set(good["resume_id"].tolist())

        p_at_k = precision_at_k(ranked_resume_ids, hr_good_set, k)
        ap = average_precision(ranked_resume_ids, hr_good_set)

        rows.append({
            "job_id": job["job_id"], "role": job["role"],
            f"Precision@{k}": round(p_at_k, 3), "AvgPrecision": round(ap, 3)
        })

    eval_df = pd.DataFrame(rows)
    summary = {
        f"Mean Precision@{k}": round(eval_df[f"Precision@{k}"].mean(), 3),
        "MAP (Mean Average Precision)": round(eval_df["AvgPrecision"].mean(), 3),
    }
    return eval_df, summary


if __name__ == "__main__":
    resumes_df = pd.read_csv("resumes.csv")
    jobs_df = pd.read_csv("job_descriptions.csv")
    hr_labels_df = pd.read_csv("hr_labels.csv")

    embed_fn, embed_name = get_embedder()
    print(f"Using embeddings: {embed_name}")

    sim_matrix = build_similarity_matrix(resumes_df, jobs_df, embed_fn)

    print("\n=== Example: Top 5 resumes for Job J001 ===")
    top5 = rank_resumes_for_job(sim_matrix, 0, resumes_df, top_k=5)
    print(top5[["resume_id", "true_role", "similarity_score"]].to_string(index=False))

    eval_df, summary = evaluate_against_hr_labels(sim_matrix, jobs_df, resumes_df, hr_labels_df, k=5)
    print("\n=== Validation against HR-labeled good matches ===")
    print(eval_df.to_string(index=False))
    print("\nSummary:", summary)


Matching App :

import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import threading

_missing = []
try:
    import pandas as pd
except ImportError:
    _missing.append("pandas")
try:
    import sklearn  # noqa
except ImportError:
    _missing.append("scikit-learn")

if _missing:
    _root = tk.Tk()
    _root.withdraw()
    messagebox.showerror(
        "Missing Libraries",
        "Missing: " + ", ".join(_missing) +
        "\n\nRun:\npython -m pip install pandas numpy scikit-learn sentence-transformers\n\n"
        "Then run this app again."
    )
    sys.exit(1)

from matching_pipeline import (
    get_embedder, build_similarity_matrix, rank_resumes_for_job, evaluate_against_hr_labels
)

DATA_FILES = ["resumes.csv", "job_descriptions.csv", "hr_labels.csv"]


class MatchingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Resume-to-Job Semantic Matching System")
        self.root.geometry("1050x700")

        self.resumes_df = None
        self.jobs_df = None
        self.hr_labels_df = None
        self.sim_matrix = None
        self.embed_name = None

        top = tk.Frame(root)
        top.pack(fill="x", padx=10, pady=8)
        tk.Label(top, text="Resume-to-Job Semantic Matching", font=("Segoe UI", 16, "bold")).pack(side="left")
        self.load_btn = tk.Button(top, text="Load Data & Build Embeddings", bg="#2874a6", fg="white",
                                   font=("Segoe UI", 10, "bold"), command=self.load_and_embed_threaded)
        self.load_btn.pack(side="right")

        self.status_label = tk.Label(root, text="Click 'Load Data & Build Embeddings' to begin "
                                                   "(first run downloads the model, needs internet).", fg="#555")
        self.status_label.pack(anchor="w", padx=12)

        # Job selector
        job_frame = tk.Frame(root)
        job_frame.pack(fill="x", padx=10, pady=6)
        tk.Label(job_frame, text="Select Job:", font=("Segoe UI", 10, "bold")).pack(side="left")
        self.job_combo = ttk.Combobox(job_frame, state="disabled", width=70)
        self.job_combo.pack(side="left", padx=8)
        self.job_combo.bind("<<ComboboxSelected>>", lambda e: self.show_ranking())

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=8)

        self.tab_ranking = tk.Frame(self.notebook)
        self.tab_eval = tk.Frame(self.notebook)
        self.notebook.add(self.tab_ranking, text="Ranked Resumes")
        self.notebook.add(self.tab_eval, text="Validation vs HR Labels")

    # ---------------- Load + embed (threaded so GUI doesn't freeze) ----------------
    def load_and_embed_threaded(self):
        missing_files = [f for f in DATA_FILES if not os.path.exists(f)]
        if missing_files:
            messagebox.showerror("Files Not Found", f"Missing: {', '.join(missing_files)}\n"
                                                       f"Run generate_data.py first.")
            return
        self.load_btn.config(state="disabled", text="Loading & embedding...")
        self.status_label.config(text="Loading model and building embeddings, please wait...")
        threading.Thread(target=self._load_and_embed, daemon=True).start()

    def _load_and_embed(self):
        try:
            self.resumes_df = pd.read_csv("resumes.csv")
            self.jobs_df = pd.read_csv("job_descriptions.csv")
            self.hr_labels_df = pd.read_csv("hr_labels.csv")

            embed_fn, embed_name = get_embedder()
            self.embed_name = embed_name
            self.sim_matrix = build_similarity_matrix(self.resumes_df, self.jobs_df, embed_fn)

            self.root.after(0, self._on_loaded)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load/embed:\n\n{e}"))
            self.root.after(0, lambda: self.load_btn.config(state="normal", text="Load Data & Build Embeddings"))

    def _on_loaded(self):
        self.status_label.config(text=f"Ready. Using embeddings: {self.embed_name}. "
                                        f"{len(self.resumes_df)} resumes, {len(self.jobs_df)} jobs loaded.")
        self.load_btn.config(state="normal", text="Reload & Re-embed")

        job_options = [f"{row.job_id} - {row.role}" for row in self.jobs_df.itertuples()]
        self.job_combo["values"] = job_options
        self.job_combo.config(state="readonly")
        self.job_combo.current(0)
        self.show_ranking()
        self.show_evaluation()

    # ---------------- Ranking tab ----------------
    def show_ranking(self):
        for w in self.tab_ranking.winfo_children():
            w.destroy()

        job_idx = self.job_combo.current()
        job_row = self.jobs_df.iloc[job_idx]

        job_label = tk.Label(self.tab_ranking, text=f"Job: {job_row['job_id']} ({job_row['role']})\n{job_row['job_text']}",
                              wraplength=1000, justify="left", font=("Segoe UI", 10, "bold"), fg="#1b2631")
        job_label.pack(anchor="w", padx=10, pady=8)

        ranked = rank_resumes_for_job(self.sim_matrix, job_idx, self.resumes_df, top_k=10)

        tree = ttk.Treeview(self.tab_ranking, columns=["rank", "resume_id", "true_role", "score", "text"],
                             show="headings", height=12)
        for col, w in [("rank", 50), ("resume_id", 80), ("true_role", 130), ("score", 90), ("text", 550)]:
            tree.heading(col, text=col.replace("_", " ").title())
            tree.column(col, width=w, anchor="w")
        for i, row in ranked.iterrows():
            tree.insert("", "end", values=[i + 1, row["resume_id"], row["true_role"],
                                            f"{row['similarity_score']:.3f}", row["resume_text"][:90] + "..."])
        tree.pack(fill="both", expand=True, padx=10, pady=8)

    # ---------------- Evaluation tab ----------------
    def show_evaluation(self):
        for w in self.tab_eval.winfo_children():
            w.destroy()

        eval_df, summary = evaluate_against_hr_labels(
            self.sim_matrix, self.jobs_df, self.resumes_df, self.hr_labels_df, k=5
        )

        summary_text = " | ".join(f"{k}: {v}" for k, v in summary.items())
        tk.Label(self.tab_eval, text="Validation Summary (across all jobs): " + summary_text,
                 font=("Segoe UI", 10, "bold"), fg="#117864").pack(anchor="w", padx=10, pady=8)

        tree = ttk.Treeview(self.tab_eval, columns=list(eval_df.columns), show="headings", height=14)
        for col in eval_df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=140, anchor="center")
        for _, row in eval_df.iterrows():
            tree.insert("", "end", values=list(row))
        tree.pack(fill="both", expand=True, padx=10, pady=8)

        note = tk.Label(
            self.tab_eval,
            text="Precision@5 = of the top 5 ranked resumes, what fraction were HR-labeled good matches.\n"
                 "MAP (Mean Average Precision) rewards ranking good matches near the top of the list.",
            fg="#555", justify="left"
        )
        note.pack(anchor="w", padx=10, pady=6)


if __name__ == "__main__":
    root = tk.Tk()
    app = MatchingApp(root)
    root.mainloop()