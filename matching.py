import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MBTI_TYPES = ["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP",
              "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"]

CITIES = {"Bengaluru": (12.97, 77.59), "Hyderabad": (17.39, 78.49), "Pune": (18.52, 73.86),
          "Mumbai": (19.08, 72.88), "Delhi": (28.61, 77.21), "Chennai": (13.08, 80.27),
          "Kolkata": (22.57, 88.36), "Ahmedabad": (23.02, 72.57), "Jaipur": (26.91, 75.79),
          "Kochi": (9.93, 76.27), "Chandigarh": (30.73, 76.78), "Indore": (22.72, 75.86)}

# Pairings that score 100 in the MBTI rule matrix.
IDEAL = {frozenset(p) for p in [
    ("INTJ", "ENFP"), ("INTJ", "ENTP"), ("INFJ", "ENFP"), ("INFJ", "ENTP"),
    ("INTP", "ENTJ"), ("INTP", "ESTJ"), ("ENFJ", "INFP"), ("ENFJ", "ISFP"),
    ("ISTJ", "ESFP"), ("ISTJ", "ESTP"), ("ISFJ", "ESFP"), ("ISFJ", "ESTP"),
    ("ESTJ", "ISTP"), ("ESFJ", "ISFP"), ("ENTJ", "INFP"), ("ISTP", "ESFJ")]}

DEFAULT_W = np.array([0.5, 0.3, 0.2])   # w1 text, w2 MBTI, w3 location
FEATURE_NAMES = ["Text", "MBTI", "Location"]


# ---------------------------------------------------------------- scoring layer
def mbti_rule(a, b):
    """MBTI compatibility on a 0-100 scale (ideal pairs = 100, otherwise a rule-based score)."""
    if frozenset((a, b)) in IDEAL:
        return 100
    same = lambda k: a[k] == b[k]
    return 30 + 25 * same(1) + 15 * (not same(0)) + 10 * same(3) + 10 * (not same(2))


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------- learning layer
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def learn_weights(X, y, prior=DEFAULT_W, lr=1.0, epochs=400, lam=0.02, scale=4.0):
    """Per-user weights from Accept(1)/Reject(0) feedback.

    Logistic regression fitted by gradient descent, pulled gently towards the default
    weights (L2 prior) and constrained to be non-negative, then normalised to sum to 1.
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    t0 = prior * scale
    t, b = t0.copy(), -float((X @ t0).mean())
    for _ in range(epochs):
        p = sigmoid(X @ t + b)
        t = np.maximum(t - lr * (X.T @ (p - y) / len(y) + lam * (t - t0)), 0.0)
        b -= lr * float((p - y).mean())
    return t / t.sum() if t.sum() > 1e-9 else prior.copy()


# ---------------------------------------------------------------- NLP layer + engine
class MatchEngine:
    def __init__(self, users_csv="users.csv", feedback_csv="feedback.csv"):  # feedback_csv=None -> no training
        for pkg in ["stopwords", "wordnet", "omw-1.4"]:
            nltk.download(pkg, quiet=True)
        self.stop = set(stopwords.words("english"))
        self.lem = WordNetLemmatizer()
        self.users = pd.read_csv(users_csv)
        self.feedback = (pd.read_csv(feedback_csv) if feedback_csv else
                         pd.DataFrame(columns=["user_id", "matched_user_id", "action", "timestamp"]))
        u = self.users
        self.idx = {uid: i for i, uid in enumerate(u.user_id)}

        self.vec_p = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.vec_a = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.Xp = self.vec_p.fit_transform(u.professional_summary.apply(self.clean))
        self.Xa = self.vec_a.fit_transform(u.about_me.apply(self.clean))
        raw = 0.6 * cosine_similarity(self.Xp) + 0.4 * cosine_similarity(self.Xa)
        # Rescale so a very strong text match (99th percentile) reads as ~100%.
        self.text_scale = np.percentile(raw[~np.eye(len(u), dtype=bool)], 99)
        self.mbti_matrix = pd.DataFrame(
            [[mbti_rule(a, b) for b in MBTI_TYPES] for a in MBTI_TYPES],
            index=MBTI_TYPES, columns=MBTI_TYPES)
        self.saved_weights = self.train_all_users()

    def clean(self, text):
        text = re.sub(r"[^a-z\s]", " ", str(text).lower())
        return " ".join(self.lem.lemmatize(self.lem.lemmatize(w), "v")
                        for w in text.split() if w not in self.stop and len(w) > 2)

    def features(self, profile):
        """n_users x 3 matrix: [text similarity, MBTI match, location score], each 0-1."""
        u = self.users
        r = 0.6 * cosine_similarity(self.vec_p.transform([self.clean(profile["professional_summary"])]), self.Xp)[0] \
            + 0.4 * cosine_similarity(self.vec_a.transform([self.clean(profile["about_me"])]), self.Xa)[0]
        dist = haversine(profile["latitude"], profile["longitude"], u.latitude.values, u.longitude.values)
        loc = np.exp(-dist / 400)
        mbti = self.mbti_matrix.loc[profile["mbti"], u.mbti].values / 100
        return np.column_stack([np.clip(r / self.text_scale, 0, 1), mbti, loc])

    def train_all_users(self, fb=None):
        fb = self.feedback if fb is None else fb
        weights = {}
        for uid, g in fb.groupby("user_id"):
            F = self.features(self.users.iloc[self.idx[uid]])
            weights[uid] = learn_weights(np.array([F[self.idx[m]] for m in g.matched_user_id]), g.action.values)
        return weights

    def match_score(self, user_a, user_b, personalised=True):
        """Compatibility (0-100%) between two user IDs, e.g. match_score("U001", "U025")."""
        w = self.saved_weights.get(user_a, DEFAULT_W) if personalised else DEFAULT_W
        F = self.features(self.users.iloc[self.idx[user_a]])
        return float(100 * (F[self.idx[user_b]] @ w))

    def top_matches(self, user_id, k=5, personalised=True):
        w = self.saved_weights.get(user_id, DEFAULT_W) if personalised else DEFAULT_W
        F = self.features(self.users.iloc[self.idx[user_id]])
        s = F @ w
        s[self.idx[user_id]] = -1
        top = np.argsort(-s)[:k]
        return pd.DataFrame({"user_id": self.users.user_id.values[top], "name": self.users.name.values[top],
                             "score_%": np.round(100 * s[top], 1)})


if __name__ == "__main__":
    eng = MatchEngine()
    print("U001 vs U025:", round(eng.match_score("U001", "U025"), 1), "%")
    print(eng.top_matches("U001"))
