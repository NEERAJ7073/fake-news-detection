"""
Fake News Detection Pipeline
==============================
Dataset   : Synthetic dataset modelled after the Liar / Kaggle Fake News datasets
Pipeline  : Preprocessing → Feature Engineering (TF-IDF + Sentiment) → ML Models → Evaluation
"""

import warnings, random, re, string, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, csr_matrix

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ──────────────────────────────────────────────────────────────────────────────
# 1.  SYNTHETIC DATASET  (mirrors Liar / Kaggle Fake-News structure)
# ──────────────────────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

REAL_TEMPLATES = [
    "Scientists at {uni} published findings showing {effect} in a peer-reviewed study.",
    "The {govt} announced new policies aimed at {topic} following extensive consultation.",
    "According to official data, {metric} increased by {pct}% this quarter.",
    "Researchers confirmed {finding} after analyzing data from over {n} participants.",
    "The {org} released a report indicating {result} based on verified sources.",
    "Health authorities stated that {treatment} shows promise in clinical trials.",
    "Economic experts forecast steady growth as {indicator} remains stable.",
    "Local officials confirmed that {event} will take place following standard procedures.",
    "A new study from {uni} demonstrates the effectiveness of {method} in controlled settings.",
    "Government records show that {spending} has been allocated for {program} this fiscal year.",
]

FAKE_TEMPLATES = [
    "BREAKING: Secret documents EXPOSE the TRUTH about {conspiracy} that the media hides!",
    "SHOCKING: {celebrity} ADMITS to {scandalous} in leaked audio — share before deleted!",
    "The DEEP STATE is using {tech} to control your {thing} — wake up sheeple!!!",
    "Scientists SUPPRESSED the {cure} because Big Pharma doesn't want you to know!",
    "CONFIRMED: {politician} caught on camera accepting bribes from {org} — mainstream media silent!",
    "Anonymous insider REVEALS terrifying plot to {plot} by the end of {year}.",
    "You won't believe what {public_figure} said about {group} when he thought cameras were off!",
    "MIRACLE CURE discovered by {doctor} that {disease} patients MUST know about NOW!",
    "Government ADMITS they've been hiding {secret} from the public for decades!",
    "This ONE WEIRD TRICK {claim} and doctors HATE him for revealing it!",
]

UNI = ["Harvard", "MIT", "Oxford", "Stanford", "Cambridge", "Johns Hopkins"]
GOVT = ["the federal government", "state officials", "the ministry", "local council"]
TOPIC = ["climate resilience", "public health", "economic recovery", "education reform"]
ORG = ["WHO", "the UN", "UNICEF", "the CDC", "national health board"]
CELEB = ["a prominent senator", "a tech billionaire", "a renowned scientist"]
CONSPIRACY = ["vaccine chips", "climate control", "5G mind waves", "chemtrails"]
POLITICIAN = ["top official", "senior minister", "prominent lawmaker"]
PLOT = ["seize control of all banks", "eliminate free speech", "ban cash globally"]


def fill_template(t, fake=False):
    subs = {
        "uni": random.choice(UNI), "govt": random.choice(GOVT),
        "effect": random.choice(["reduced inflammation", "improved cognition", "lower emissions"]),
        "topic": random.choice(TOPIC), "metric": random.choice(["GDP", "employment", "output"]),
        "pct": random.randint(2, 18), "finding": "a statistically significant correlation",
        "n": random.randint(500, 10000), "org": random.choice(ORG),
        "result": "positive outcomes for the studied population",
        "treatment": "the new protocol", "indicator": "consumer confidence",
        "event": "the scheduled summit", "method": "the revised methodology",
        "spending": f"${random.randint(10,500)}M", "program": "infrastructure",
        "celebrity": random.choice(CELEB), "scandalous": "embezzling funds",
        "conspiracy": random.choice(CONSPIRACY), "tech": random.choice(["5G", "AI", "nanobots"]),
        "thing": random.choice(["thoughts", "behaviour", "health"]),
        "cure": random.choice(["cancer cure", "aging reversal", "mind control antidote"]),
        "politician": random.choice(POLITICIAN),
        "public_figure": random.choice(["a world leader", "a tech mogul"]),
        "group": random.choice(["taxpayers", "minorities", "the middle class"]),
        "doctor": "Dr. Smith", "disease": random.choice(["diabetes", "cancer", "dementia"]),
        "secret": random.choice(["alien contact", "water fluoridation plans", "suppressed tech"]),
        "claim": random.choice(["burns belly fat", "boosts IQ by 50 points", "reverses aging"]),
        "year": random.randint(2024, 2027),
        "plot": random.choice(PLOT),
    }
    return t.format(**subs)


def generate_dataset(n=3000):
    rows = []
    for _ in range(n // 2):
        t = random.choice(REAL_TEMPLATES)
        text = fill_template(t, fake=False)
        # Pad real news with realistic context
        context = (f" The findings were corroborated by multiple independent researchers. "
                   f"Data collection followed strict ethical guidelines approved by relevant authorities.")
        rows.append({"text": text + context, "label": 0, "label_name": "REAL"})
    for _ in range(n // 2):
        t = random.choice(FAKE_TEMPLATES)
        text = fill_template(t, fake=True)
        extra = (" SHARE THIS NOW before it's taken down!!! "
                 "They don't want YOU to know the truth!!!")
        rows.append({"text": text + extra, "label": 1, "label_name": "FAKE"})
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2.  TEXT PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)           # URLs
    text = re.sub(r"[^a-z\s]", "", text)                  # punctuation / digits
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────
sia = SentimentIntensityAnalyzer()

def sentiment_features(texts):
    feats = []
    for t in texts:
        scores = sia.polarity_scores(t)
        excl  = t.count("!")
        caps  = sum(1 for c in t if c.isupper())
        feats.append([scores["pos"], scores["neg"], scores["neu"],
                      scores["compound"], excl, caps])
    return np.array(feats)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("       FAKE NEWS DETECTION PIPELINE")
print("=" * 65)

# 4a. Generate & inspect dataset
print("\n[1/6] Generating dataset …")
df = pd.read_csv("FakeNewsNet.csv")
df = df.rename(columns={"title": "text", "real": "label"})
df = df[["text", "label"]].dropna()
df["label_name"] = df["label"].map({1: "REAL", 0: "FAKE"})
print(f"      Total samples : {len(df)}")
print(f"      Real news     : {(df.label == 0).sum()}")
print(f"      Fake news     : {(df.label == 1).sum()}")

# 4b. Preprocess
print("\n[2/6] Preprocessing text …")
t0 = time.time()
df["clean_text"] = df["text"].apply(preprocess)
print(f"      Done in {time.time()-t0:.1f}s  |  Example:")
print(f"      Original : {df.text.iloc[0][:80]}…")
print(f"      Cleaned  : {df.clean_text.iloc[0][:80]}…")

# 4c. Train / test split
X_train_raw, X_test_raw, X_train_clean, X_test_clean, y_train, y_test = \
    train_test_split(df["text"], df["clean_text"], df["label"],
                     test_size=0.2, random_state=42, stratify=df["label"])

# 4d. Feature engineering
print("\n[3/6] Engineering features …")
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train_clean)
X_test_tfidf  = tfidf.transform(X_test_clean)

sent_train = csr_matrix(sentiment_features(X_train_raw.tolist()))
sent_test  = csr_matrix(sentiment_features(X_test_raw.tolist()))

X_train_final = hstack([X_train_tfidf, sent_train])
X_test_final  = hstack([X_test_tfidf,  sent_test])
print(f"      Feature matrix shape : {X_train_final.shape}")

# 4e. Model zoo
print("\n[4/6] Training models …")

models = {
    "Logistic Regression":        (LogisticRegression(max_iter=1000, C=1.0, random_state=42), "full"),
    "Passive Aggressive":         (PassiveAggressiveClassifier(max_iter=1000, random_state=42), "full"),
    "Linear SVM":                 (LinearSVC(max_iter=2000, random_state=42), "full"),
    "Naive Bayes":                (MultinomialNB(alpha=0.1), "tfidf"),   # no negative values
    "Random Forest":              (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), "full"),
    "Gradient Boosting":          (GradientBoostingClassifier(n_estimators=100, random_state=42), "full"),
}

results = {}
for name, (model, feat_set) in models.items():
    Xtr = X_train_tfidf if feat_set == "tfidf" else X_train_final
    Xte = X_test_tfidf  if feat_set == "tfidf" else X_test_final
    t0 = time.time()
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    elapsed = time.time() - t0

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    # ROC-AUC (use decision_function or predict_proba)
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(Xte)[:, 1]
    else:
        y_scores = model.decision_function(Xte)
    auc = roc_auc_score(y_test, y_scores)

    results[name] = dict(model=model, y_pred=y_pred, y_scores=y_scores,
                         acc=acc, prec=prec, rec=rec, f1=f1, auc=auc, time=elapsed)
    print(f"  ✓  {name:<26}  Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}  ({elapsed:.1f}s)")

best_name = max(results, key=lambda k: results[k]["f1"])
best = results[best_name]
print(f"\n  ★  Best model: {best_name}  (F1={best['f1']:.4f})")

# ──────────────────────────────────────────────────────────────────────────────
# 5.  DETAILED EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[5/6] Detailed evaluation for → {best_name}")
print(classification_report(y_test, best["y_pred"],
                             target_names=["REAL", "FAKE"]))

# ──────────────────────────────────────────────────────────────────────────────
# 6.  VISUALISATIONS
# ──────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Generating visualisations …")

PALETTE = {"REAL": "#2ecc71", "FAKE": "#e74c3c"}
BG      = "#f8f9fa"
DARK    = "#2c3e50"

fig = plt.figure(figsize=(22, 18), facecolor=BG)
fig.suptitle("Fake News Detection – Complete Analysis Dashboard",
             fontsize=20, fontweight="bold", color=DARK, y=0.98)

gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.38,
                       left=0.06, right=0.97, top=0.93, bottom=0.05)

# ── Plot 1: Label Distribution
ax1 = fig.add_subplot(gs[0, 0])
counts = df["label_name"].value_counts()
bars = ax1.bar(counts.index, counts.values,
               color=[PALETTE[k] for k in counts.index],
               width=0.5, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
             f"{val}", ha="center", va="bottom", fontweight="bold", fontsize=12)
ax1.set_title("Dataset Distribution", fontsize=13, fontweight="bold", color=DARK)
ax1.set_ylabel("Count"); ax1.set_facecolor(BG)
ax1.spines[["top","right"]].set_visible(False)

# ── Plot 2: Text Length Distribution
ax2 = fig.add_subplot(gs[0, 1])
for lbl, col in PALETTE.items():
    lengths = df[df.label_name == lbl]["text"].str.len()
    ax2.hist(lengths, bins=30, alpha=0.65, color=col, label=lbl, edgecolor="white")
ax2.set_title("Article Length Distribution", fontsize=13, fontweight="bold", color=DARK)
ax2.set_xlabel("Characters"); ax2.set_ylabel("Frequency")
ax2.legend(); ax2.set_facecolor(BG)
ax2.spines[["top","right"]].set_visible(False)

# ── Plot 3: Sentiment Compound Score
ax3 = fig.add_subplot(gs[0, 2])
for lbl, col in PALETTE.items():
    subset = df[df.label_name == lbl]["text"].tolist()
    scores = [sia.polarity_scores(t)["compound"] for t in subset]
    ax3.hist(scores, bins=30, alpha=0.65, color=col, label=lbl, edgecolor="white")
ax3.set_title("Sentiment Compound Score", fontsize=13, fontweight="bold", color=DARK)
ax3.set_xlabel("Compound Score"); ax3.set_ylabel("Frequency")
ax3.axvline(0, color="grey", linestyle="--", linewidth=1)
ax3.legend(); ax3.set_facecolor(BG)
ax3.spines[["top","right"]].set_visible(False)

# ── Plot 4: Model Comparison Bar Chart
ax4 = fig.add_subplot(gs[1, :2])
metrics = ["acc", "prec", "rec", "f1", "auc"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
x = np.arange(len(metrics))
width = 0.13
colors = plt.cm.tab10(np.linspace(0, 0.7, len(models)))

for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics]
    offset = (i - len(models)/2 + 0.5) * width
    bars = ax4.bar(x + offset, vals, width, label=name, color=colors[i],
                   edgecolor="white", linewidth=0.8)

ax4.set_xticks(x); ax4.set_xticklabels(metric_labels, fontsize=10)
ax4.set_ylim(0.5, 1.05)
ax4.set_title("Model Performance Comparison", fontsize=13, fontweight="bold", color=DARK)
ax4.set_ylabel("Score")
ax4.legend(loc="lower right", fontsize=8, ncol=2)
ax4.set_facecolor(BG)
ax4.spines[["top","right"]].set_visible(False)
ax4.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)

# ── Plot 5: ROC Curves
ax5 = fig.add_subplot(gs[1, 2])
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res["y_scores"])
    ax5.plot(fpr, tpr, color=colors[i], linewidth=1.8,
             label=f"{name.split()[0]} ({res['auc']:.2f})")
ax5.plot([0,1],[0,1],"k--", linewidth=1)
ax5.set_title("ROC Curves (All Models)", fontsize=13, fontweight="bold", color=DARK)
ax5.set_xlabel("False Positive Rate"); ax5.set_ylabel("True Positive Rate")
ax5.legend(fontsize=7.5); ax5.set_facecolor(BG)
ax5.spines[["top","right"]].set_visible(False)

# ── Plot 6: Confusion Matrix (best model)
ax6 = fig.add_subplot(gs[2, 0])
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn",
            xticklabels=["REAL","FAKE"], yticklabels=["REAL","FAKE"],
            ax=ax6, linewidths=2, linecolor="white",
            annot_kws={"size": 14, "weight": "bold"})
ax6.set_title(f"Confusion Matrix\n({best_name})", fontsize=12, fontweight="bold", color=DARK)
ax6.set_ylabel("Actual"); ax6.set_xlabel("Predicted")

# ── Plot 7: Per-class Metrics (best model)
ax7 = fig.add_subplot(gs[2, 1])
per_class = {
    "Precision": [precision_score(y_test, best["y_pred"], pos_label=i) for i in [0,1]],
    "Recall":    [recall_score(y_test, best["y_pred"], pos_label=i) for i in [0,1]],
    "F1-Score":  [f1_score(y_test, best["y_pred"], pos_label=i) for i in [0,1]],
}
x7 = np.arange(2)
w7 = 0.25
for j, (met, vals) in enumerate(per_class.items()):
    ax7.bar(x7 + j*w7 - w7, vals, w7, label=met,
            color=["#3498db","#9b59b6","#e67e22"][j], edgecolor="white")
ax7.set_xticks(x7); ax7.set_xticklabels(["REAL","FAKE"])
ax7.set_ylim(0, 1.08)
ax7.set_title(f"Per-Class Metrics\n({best_name})", fontsize=12, fontweight="bold", color=DARK)
ax7.set_ylabel("Score"); ax7.legend(fontsize=9)
ax7.set_facecolor(BG)
ax7.spines[["top","right"]].set_visible(False)

# ── Plot 8: Training time vs F1
ax8 = fig.add_subplot(gs[2, 2])
names_short = [n.replace(" ", "\n") for n in results]
f1s   = [results[n]["f1"]  for n in results]
times = [results[n]["time"] for n in results]
sc = ax8.scatter(times, f1s, s=120, c=colors, zorder=5, edgecolors="white", linewidth=1.5)
for i, n in enumerate(names_short):
    ax8.annotate(n, (times[i], f1s[i]), textcoords="offset points",
                 xytext=(6, 4), fontsize=7.5, color=DARK)
ax8.set_xlabel("Training Time (s)"); ax8.set_ylabel("F1-Score")
ax8.set_title("Speed vs Accuracy Trade-off", fontsize=12, fontweight="bold", color=DARK)
ax8.set_facecolor(BG)
ax8.spines[["top","right"]].set_visible(False)

plt.savefig("fake_news_detection_dashboard.png",
            dpi=150, bbox_inches="tight", facecolor=BG),
print("      Saved → fake_news_detection_dashboard.png")

# ──────────────────────────────────────────────────────────────────────────────
# 7.  SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL RESULTS SUMMARY")
print("=" * 65)
print(f"  {'Model':<28} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print("  " + "-"*63)
for name, res in sorted(results.items(), key=lambda x: -x[1]["f1"]):
    star = " ★" if name == best_name else ""
    print(f"  {name+star:<28} {res['acc']:>6.3f} {res['prec']:>6.3f} "
          f"{res['rec']:>6.3f} {res['f1']:>6.3f} {res['auc']:>6.3f}")
print("=" * 65)
print("\nPipeline complete ✓")
