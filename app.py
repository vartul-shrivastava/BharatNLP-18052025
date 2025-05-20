from flask import (
    Flask, request, jsonify, render_template,
    url_for, send_from_directory
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from gensim import corpora, models
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend for rendering

from wordcloud import WordCloud, STOPWORDS

import pandas as pd
import unicodedata, re, io, base64, os, uuid
from collections import Counter

# ───────── Flask basics ──────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")

# ───────── Directories & mappings ───────────────────────────────────
ROOT          = os.path.abspath(os.path.dirname(__file__))
FONT_DIR      = os.path.join(ROOT, "./static/Font Styles Corpus")
STOPWORD_DIR  = os.path.join(ROOT, "./static/Stopwords Corpus")
DOWNLOAD_DIR  = os.path.join(ROOT, "tmp")          # for labelled CSVs
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

FONT_MAP = {
    "hindi":   "Lohit-Devanagari.ttf",
    "marathi": "Lohit-Devanagari.ttf",
    "bengali": "NotoSansBengali-Regular.ttf",
    "tamil":   "NotoSansTamil-Regular.ttf",
    "telugu":  "NotoSansTelugu-Regular.ttf",
}

REGEX_MAP = {
    "hindi":   r"[\u0900-\u097F]+",
    "marathi": r"[\u0900-\u097F]+",
    "bengali": r"[\u0980-\u09FF]+",
    "tamil":   r"[\u0B80-\u0BFF]+",
    "telugu":  r"[\u0C00-\u0C7F]+",
}

def img_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def label_from_polarity(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"

@app.route("/")
def index():
    return render_template("index.html")

# ───────── 1. Word-Cloud generator ─────────────────────────────────
@app.route("/generate", methods=["POST"])
def generate_wordcloud():
    # Only TXT upload for new UI
    txt = ""
    if "text_file" in request.files and request.files["text_file"]:
        txt = request.files["text_file"].read().decode("utf-8")
    else:
        return jsonify(error="Please upload a text file."), 400

    lang = request.form.get("language")
    sw_path = os.path.join(STOPWORD_DIR, f"{lang}_stopwords.txt")
    if not os.path.isfile(sw_path):
        return jsonify(error=f"Missing stopwords for {lang}"), 400
    with open(sw_path, encoding="utf-8") as f:
        base_sw = set(f.read().splitlines())

    extra = request.files.get("custom_stopwords_file")
    if extra and extra.filename:
        base_sw |= set(extra.read().decode("utf-8").splitlines())
    pasted = request.form.get("custom_stopwords_text", "").splitlines()
    base_sw |= {w.strip() for w in pasted if w.strip()}

    font_file = FONT_MAP.get(lang)
    font_path = os.path.join(FONT_DIR, font_file)
    if not os.path.isfile(font_path):
        return jsonify(error=f"Missing font for {lang}"), 400

    wc_args = {
        "font_path":       font_path,
        "background_color": request.form.get("background_color_custom")
                             or request.form.get("background_color"),
        "width":           800,
        "height":          400,
        "stopwords":       STOPWORDS.union(base_sw),
        "regexp":          REGEX_MAP.get(lang),
        "collocations":    bool(request.form.get("collocations")),
        "max_words":       int(request.form.get("max_words") or 200),
        "min_font_size":   int(request.form.get("min_font_size") or 4),
        "colormap":        request.form.get("colormap_custom")
                             or request.form.get("colormap"),
    }
    wc = WordCloud(**wc_args).generate(txt)
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    img_b64 = img_to_b64(fig)
    return jsonify(img_data=img_b64)

@app.route("/topic_model", methods=["POST"])
def topic_model():
    # ---- Step 1: File Handling ----
    csv_file = request.files.get("csv_file")
    text_file = request.files.get("text_file")
    if csv_file and csv_file.filename:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            return jsonify(error=f"Invalid CSV: {e}"), 400
        col = request.form.get("csv_column")
        if not col or col not in df.columns:
            return jsonify(error=f"Column '{col}' not found in CSV"), 400
        raw_texts = df[col].astype(str).tolist()
    elif text_file and text_file.filename:
        raw_texts = text_file.read().decode("utf-8").splitlines()
    else:
        return jsonify(error="Please upload CSV or text file."), 400

    # ---- Step 2: Language and Stopwords ----
    lang = request.form.get("tm_language")
    sw_path = os.path.join(STOPWORD_DIR, f"{lang}_stopwords.txt")
    if not os.path.isfile(sw_path):
        return jsonify(error=f"Missing stopwords for {lang}"), 400
    with open(sw_path, encoding="utf-8") as f:
        stopset = set(f.read().splitlines())
    regex_range = REGEX_MAP.get(lang)

    # ---- Step 3: Parameters ----
    model_type = request.form.get("model_type", "lda").lower()
    num_topics = int(request.form.get("num_topics") or 5)
    passes     = int(request.form.get("passes") or 10)
    min_freq   = int(request.form.get("min_freq") or 5)
    max_ratio  = float(request.form.get("max_ratio") or 0.8)

    # For sweep mode
    range_min = request.form.get("range_min")
    range_max = request.form.get("range_max")

    # ---- Step 4: Tokenizer ----
    def custom_tokenizer(text):
        clean = re.sub(rf"http\S+|\d+|[^\s{regex_range[1:-1]}]+", " ", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        tokens = re.findall(regex_range, clean)
        tokens = [t for t in tokens if t not in stopset and len(t) > 1]
        return tokens

    token_docs = [custom_tokenizer(line) for line in raw_texts if line.strip()]
    if not token_docs:
        return jsonify(error="No valid tokens found"), 400
    dictionary = corpora.Dictionary(token_docs)
    dictionary.filter_extremes(no_below=min_freq, no_above=max_ratio)

    # ---- Step 5: If sweep mode requested ----
    if range_min and range_max:
        try:
            topic_range = range(int(range_min), int(range_max)+1)
        except Exception:
            return jsonify(error="Invalid range parameters"), 400

        coherences = []
        for n_topics in topic_range:
            print(f"Running {model_type} with {n_topics} topics")
            try:
                if model_type == "lda":
                    corpus = [dictionary.doc2bow(doc) for doc in token_docs]
                    if len(dictionary) == 0 or not any(corpus):
                        coherences.append(None)
                        continue
                    model = models.LdaModel(
                        corpus, id2word=dictionary,
                        num_topics=n_topics, passes=passes, random_state=42
                    )
                    lda_topics = [ [w for w, _ in model.show_topic(idx, topn=10)] for idx in range(n_topics) ]
                    coherence = models.CoherenceModel(
                        topics=lda_topics, texts=token_docs, dictionary=dictionary, coherence="c_v"
                    ).get_coherence()
                    coherences.append(round(coherence, 4))
                elif model_type in ["lsa", "nmf"]:
                    tfidf = TfidfVectorizer(
                        analyzer='word',
                        tokenizer=custom_tokenizer,
                        preprocessor=None,
                        token_pattern=None
                    )
                    X = tfidf.fit_transform(raw_texts)
                    if model_type == "lsa":
                        svd = TruncatedSVD(n_components=n_topics, random_state=42)
                        H = svd.fit(X).components_
                    else:
                        nmf = NMF(n_components=n_topics, random_state=42)
                        H = nmf.fit(X).components_
                    terms = tfidf.get_feature_names_out()
                    topics = [[terms[i] for i in comp.argsort()[:-11:-1]] for comp in H]
                    coherence = models.CoherenceModel(
                        topics=topics, texts=token_docs, dictionary=dictionary, coherence="c_v"
                    ).get_coherence()
                    coherences.append(round(coherence, 4))
                else:
                    coherences.append(None)
            except Exception:
                coherences.append(None)
        return jsonify(
            topic_counts = list(topic_range),
            coherences   = coherences
        )

    # ---- Step 6: Default behavior (single model run) ----
    if model_type == "lda":
        corpus = [dictionary.doc2bow(doc) for doc in token_docs]
        if len(dictionary) == 0 or not any(corpus):
            return jsonify(error="No valid tokens after filtering."), 400
        model = models.LdaModel(
            corpus, id2word=dictionary,
            num_topics=num_topics, passes=passes,
            random_state=42
        )
        lda_topics = [ [w for w, _ in model.show_topic(idx, topn=10)] for idx in range(num_topics) ]
        coherence = models.CoherenceModel(
            topics=lda_topics, texts=token_docs, dictionary=dictionary, coherence="c_v"
        ).get_coherence()
        perplexity = model.log_perplexity(corpus)
        display_topics = [{"idx": idx, "words": ", ".join(lda_topics[idx])} for idx in range(num_topics)]
        return jsonify(
            coherence   = round(coherence, 4),
            perplexity  = round(perplexity, 4),
            topics      = display_topics,
            treemap     = None,
            model_type  = "lda"
        )
    elif model_type == "lsa":
        tfidf = TfidfVectorizer(
            analyzer='word',
            tokenizer=custom_tokenizer,
            preprocessor=None,
            token_pattern=None
        )
        X = tfidf.fit_transform(raw_texts)
        svd = TruncatedSVD(n_components=num_topics, random_state=42)
        H = svd.fit(X).components_
        terms = tfidf.get_feature_names_out()
        lsa_topics = [[terms[i] for i in comp.argsort()[:-11:-1]] for comp in H]
        coherence = models.CoherenceModel(
            topics=lsa_topics, texts=token_docs, dictionary=dictionary, coherence="c_v"
        ).get_coherence()
        explained_var = svd.explained_variance_ratio_.sum()
        display_topics = [{"idx": idx, "words": ", ".join(lsa_topics[idx])} for idx in range(num_topics)]
        return jsonify(
            coherence = round(coherence, 4),
            perplexity = None,
            topics = display_topics,
            treemap = None,
            model_type = "lsa",
            explained_variance = round(explained_var, 4)
        )
    elif model_type == "nmf":
        tfidf = TfidfVectorizer(
            analyzer='word',
            tokenizer=custom_tokenizer,
            preprocessor=None,
            token_pattern=None
        )
        X = tfidf.fit_transform(raw_texts)
        nmf = NMF(n_components=num_topics, random_state=42)
        H = nmf.fit(X).components_
        terms = tfidf.get_feature_names_out()
        nmf_topics = [[terms[i] for i in comp.argsort()[:-11:-1]] for comp in H]
        coherence = models.CoherenceModel(
            topics=nmf_topics, texts=token_docs, dictionary=dictionary, coherence="c_v"
        ).get_coherence()
        reconstruction_err = nmf.reconstruction_err_
        display_topics = [{"idx": idx, "words": ", ".join(nmf_topics[idx])} for idx in range(num_topics)]
        return jsonify(
            coherence = round(coherence, 4),
            perplexity = None,
            topics = display_topics,
            treemap = None,
            model_type = "nmf",
            reconstruction_err = round(reconstruction_err, 4)
        )
    else:
        return jsonify(error="Unsupported model type"), 400

# ───────── 3. Sentiment Analysis ─────────────────────────────
@app.route("/sentiment", methods=["POST"])
def sentiment_analysis():
    from deep_translator import GoogleTranslator
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    file = request.files["sentiment_file"]
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify(error=f"Invalid CSV: {e}"), 400

    col = request.form.get("text_column")
    if col not in df.columns:
        return jsonify(error=f"Column '{col}' not found"), 400

    # Frontend: pass language in "language" key, else default Hindi
    lang = request.form.get("language") or "hindi"
    sw_path = os.path.join(STOPWORD_DIR, f"{lang}_stopwords.txt")
    if not os.path.isfile(sw_path):
        return jsonify(error=f"Missing stopwords for {lang}"), 400
    with open(sw_path, encoding="utf-8") as f:
        base_sw = set(f.read().splitlines())
    font_file = FONT_MAP.get(lang)
    font_path = os.path.join(FONT_DIR, font_file)
    if not os.path.isfile(font_path):
        return jsonify(error=f"Missing font for {lang}"), 400
    regex_range = REGEX_MAP.get(lang)

    sia        = SentimentIntensityAnalyzer()
    counts     = Counter()
    clouds_txt = {"Positive": [], "Neutral": [], "Negative": []}
    tb_rows    = []
    vader_rows = []
    new_pol    = []
    new_lbl    = []

    for text in df[col].astype(str):
        try:
            en = GoogleTranslator(source="auto", target="en").translate(text)
        except Exception:
            en = text

        tb = TextBlob(en).sentiment
        tb_rows.append({
            "original": text,
            "polarity": round(tb.polarity, 4),
            "subjectivity": round(tb.subjectivity, 4)
        })
        vs = sia.polarity_scores(en)
        vader_rows.append({
            "original": text,
            "neg": vs["neg"], "neu": vs["neu"], "pos": vs["pos"],
            "compound": vs["compound"]
        })

        lbl = label_from_polarity(tb.polarity)
        counts[lbl] += 1
        clouds_txt[lbl].append(text)
        new_pol.append(round(tb.polarity, 4))
        new_lbl.append(lbl)

    df["polarity"]  = new_pol
    df["sentiment"] = new_lbl
    out_name = f"sent_{uuid.uuid4().hex[:8]}.csv"
    out_path = os.path.join(DOWNLOAD_DIR, out_name)
    df.to_csv(out_path, index=False)

    def sentiment_wordcloud(texts, stopwords, font_path, regex_range):
        if not texts:
            return ""
        wc = WordCloud(
            width=600, height=400,
            background_color="white",
            stopwords=STOPWORDS.union(stopwords),
            font_path=font_path,
            regexp=regex_range,
            collocations=False
        ).generate(" ".join(texts))
        fig = plt.figure(figsize=(6, 4))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    wc_images = {
        lbl: sentiment_wordcloud(clouds_txt[lbl], base_sw, font_path, regex_range)
        for lbl in ["Positive", "Neutral", "Negative"]
    }

    return jsonify(
        counts        = counts,
        wc_images     = wc_images,
        download_url  = url_for("download_file", filename=out_name),
    )

# ───────── 6. File download endpoint ───────────────────────────────
@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(DOWNLOAD_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)