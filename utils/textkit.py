"""
Features:
- Column detection & safe casting
- Text cleanup (URLs, mentions, emoji, punctuation)
- Optional POS-aware noun extraction (auto-falls back if tagger isn’t available)
- Word counting & distribution plotting
- TF-IDF + NMF topic modeling (generic)
- Keyword frequency + Word Cloud utilities (for news/tweets)

"""
import re
import numpy as np
import pandas as pd

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag, word_tokenize
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False


def ensure_nltk():
    """Ensure necessary NLTK corpora and models are downloaded."""
    if not _NLTK_AVAILABLE:
        return
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        except Exception:
            nltk.download("averaged_perceptron_tagger", quiet=True)
    for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

# Regex for text cleaning
URL_RE   = re.compile(r"http\S+|www\.\S+")
USER_RE  = re.compile(r"@\w+")
EMOJI_RE = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)
PUNCT_RE = re.compile(r"[^\w\s$]")


def _get_stops_and_lemm():
    """Load stopwords and lemmatizer if available."""
    base_stops = set()
    lemm = None
    if _NLTK_AVAILABLE:
        try:
            base_stops = set(stopwords.words("english"))
        except Exception:
            base_stops = set()
        try:
            lemm = WordNetLemmatizer()
        except Exception:
            lemm = None
    extra = {
        "rt","amp","im","ive","youre","weve","hes","shes","its","dont","cant","wont",
        "yeah","ok","okay","true","haha","ha","wow","cool","good","great","thanks","thank",
        "exactly","right","love","like","really","much","many","one","thing","things",
        "today","tomorrow","yesterday","time","year","years","people","guys"
    }
    return base_stops | extra, lemm

STOPS, LEMM = _get_stops_and_lemm()


def detect_text_column(df, candidates):
    """Find the first candidate text column that exists in a DataFrame."""
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of the candidate text columns exist: {list(candidates)}")


def to_text_series(s):
    """Convert a Series to string type, fill NaN with empty string, strip whitespace."""
    return s.fillna("").astype(str).str.strip()


def basic_cleanup(t):
    """Normalize text: lowercase, remove URLs, mentions, emojis, punctuation."""
    t = t.lower()
    t = URL_RE.sub(" ", t)
    t = USER_RE.sub(" ", t)
    t = EMOJI_RE.sub(" ", t)
    t = PUNCT_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clean_simple(text):
    """Clean text by removing stopwords, short words, and lemmatizing if possible."""
    t = basic_cleanup(text)
    out = []
    for w in t.split():
        if w in STOPS or len(w) <= 2:
            continue
        if w.startswith("$") or re.fullmatch(r"\d+(m|b|%)?", w):
            out.append(w.upper()); continue
        out.append(LEMM.lemmatize(w) if LEMM else w)
    return " ".join(out)


def clean_with_pos(text):
    """Advanced cleaning: use POS tagging to keep only nouns. Falls back to simple cleaning."""
    if not _NLTK_AVAILABLE:
        return clean_simple(text)
    try:
        t = basic_cleanup(text)
        toks = word_tokenize(t)
        if not toks:
            return ""
        tags = pos_tag(toks)
        out = []
        for w, p in tags:
            if w in STOPS or len(w) <= 2:
                continue
            if w.startswith("$") or re.fullmatch(r"\d+(m|b|%)?", w):
                out.append(w.upper()); continue
            if p.startswith("NN"):
                out.append(LEMM.lemmatize(w) if LEMM else w)
        return " ".join(out)
    except Exception:
        return clean_simple(text)


def preprocess_text_df(df, text_col=None, candidates=("content","fullText","text","body"), min_tokens=0, use_pos=None):
    """Preprocess a DataFrame: choose text column, clean text, create 'clean' column, filter by min tokens."""
    col = text_col or detect_text_column(df, candidates)
    s = to_text_series(df[col])
    if use_pos is None:
        use_pos = _NLTK_AVAILABLE
        if _NLTK_AVAILABLE:
            ensure_nltk()
    clean_fn = clean_with_pos if use_pos else clean_simple
    out = df.copy()
    out["clean"] = s.apply(clean_fn)
    if min_tokens and min_tokens > 0:
        out = out[out["clean"].str.split().str.len() >= int(min_tokens)].copy()
    return out, col


def add_word_count(df, text_col, out_col="word_count"):
    """Add a new column with word counts for each text entry."""
    s = to_text_series(df[text_col])
    df[out_col] = s.apply(lambda x: len(re.findall(r"\w+", x)))
    return df


def plot_wordcount_distribution(series, title="Word Count Distribution"):
    """Plot histogram distribution of word counts (with seaborn if available)."""
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(10, 6))
        sns.histplot(series, kde=True, bins=50)
        plt.title(title, fontsize=16)
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
    except Exception:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.hist(series, bins=50, density=False)
        plt.title(title)
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

# --------------------------- Keyword & WordCloud ------------------------------

def build_stopwords(extra=None):
    """Build an English stopword set with optional extras (set/list/tuple)."""
    sw = set()
    if _NLTK_AVAILABLE:
        try:
            sw = set(stopwords.words("english"))
        except Exception:
            sw = set()
    if extra:
        sw.update(list(extra))
    return sw


def combine_text(texts):
    """Combine an iterable/Series of texts into a single space-joined string."""
    s = pd.Series(texts).fillna("").astype(str)
    return " ".join(s.tolist())


def keyword_frequency(texts, stopword_set=None, topn=20, lowercase=True, alpha_only=True):
    """Return a DataFrame of top-N tokens and frequencies from texts."""
    if _NLTK_AVAILABLE:
        ensure_nltk()
    txt = combine_text(texts)
    if lowercase:
        txt = txt.lower()
    toks = word_tokenize(txt) if _NLTK_AVAILABLE else txt.split()
    if stopword_set is None:
        stopword_set = build_stopwords()
    cleaned = []
    for w in toks:
        if alpha_only and not w.isalpha():
            continue
        if w in stopword_set:
            continue
        cleaned.append(w)
    if not cleaned:
        return pd.DataFrame({"word": [], "freq": []})
    freq = {}
    for w in cleaned:
        freq[w] = freq.get(w, 0) + 1
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:int(topn)]
    return pd.DataFrame(items, columns=["word", "freq"])


def generate_wordcloud(texts, stopword_set=None, width=1000, height=500, background_color='white', colormap='viridis', max_words=100):
    """Generate and return a WordCloud object from texts."""
    try:
        from wordcloud import WordCloud
    except Exception as e:
        raise ImportError("wordcloud package is required. Install via `pip install wordcloud`. ") from e
    if stopword_set is None:
        stopword_set = build_stopwords()
    txt = combine_text(texts)
    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        stopwords=stopword_set,
        colormap=colormap,
        max_words=max_words,
    ).generate(txt)
    return wc


def plot_wordcloud(wc, title='Most Frequent Words'):
    """Plot a WordCloud object with matplotlib."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    if title:
        plt.title(title, fontsize=16)
    plt.show()


# --------------------------- Topic modeling helpers ---------------------------

def tfidf_matrix(texts, ngram_range=(1, 3), min_df=10, max_df=0.85):
    """Build a TF-IDF matrix from input texts and return (matrix, vocab, vectorizer)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    return X, vocab, vec


def nmf_topics(X, n_topics=8, random_state=42, init="nndsvd", max_iter=400):
    """Fit an NMF topic model and return the model plus W, H matrices."""
    from sklearn.decomposition import NMF
    nmf = NMF(n_components=n_topics, random_state=random_state, init=init, max_iter=max_iter)
    W = nmf.fit_transform(X)
    H = nmf.components_
    return nmf, W, H


def topics_table(model, feat_names, topn=12):
    """Create a DataFrame of top terms for each topic."""
    rows = []
    for k, comp in enumerate(model.components_):
        idx = comp.argsort()[::-1][:topn]
        rows.extend({"topic": k, "term": feat_names[i], "weight": float(comp[i])} for i in idx)
    out = pd.DataFrame(rows)
    return out


__all__ = [
    "ensure_nltk",
    "detect_text_column",
    "to_text_series",
    "basic_cleanup",
    "clean_simple",
    "clean_with_pos",
    "preprocess_text_df",
    "add_word_count",
    "plot_wordcount_distribution",
 
    # keywords & wordcloud
    "build_stopwords",
    "combine_text",
    "keyword_frequency",
    "generate_wordcloud",
    "plot_wordcloud",
 
    # topics
    "tfidf_matrix",
    "nmf_topics",
    "topics_table",
]