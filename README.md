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

