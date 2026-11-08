import pandas as pd
from matching import MBTI_TYPES, CITIES, mbti_rule
USERS_REQUIRED = ["user_id", "name", "age", "gender", "city", "latitude", "longitude", "education", "job_title",
                  "industry", "years_experience", "skills", "mbti", "work_mode", "looking_for", "career_goal",
                  "interests", "values", "professional_summary", "about_me", "join_date"]
def clean_users(path="users.csv"):
    df = pd.read_csv(path)
    missing = [c for c in USERS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"users.csv is missing columns: {missing}")
    n0 = len(df)
    df = df.drop_duplicates("user_id")
    for col in ["name", "city", "job_title", "professional_summary", "about_me", "skills", "interests", "values"]:
        df[col] = df[col].astype(str).str.strip()
    df["mbti"] = df["mbti"].astype(str).str.strip().str.upper()
    df = df[df.mbti.isin(MBTI_TYPES)]
    df = df.dropna(subset=["latitude", "longitude", "professional_summary", "about_me"])
    df = df[(df.professional_summary.str.len() > 0) & (df.about_me.str.len() > 0)]
    df["join_date"] = pd.to_datetime(df["join_date"], errors="coerce").dt.date
    print(f"users: {n0} rows in, {len(df)} rows kept")
    return df.reset_index(drop=True)


def clean_feedback(users, path="feedback.csv"):
    fb = pd.read_csv(path)
    n0 = len(fb)
    fb = fb.drop_duplicates(["user_id", "matched_user_id"])
    ids = set(users.user_id)
    fb = fb[fb.user_id.isin(ids) & fb.matched_user_id.isin(ids) & (fb.user_id != fb.matched_user_id)]
    fb = fb[fb.action.isin([0, 1])]
    fb["timestamp"] = pd.to_datetime(fb["timestamp"], errors="coerce")
    fb = fb.dropna(subset=["timestamp"]).sort_values(["user_id", "timestamp"])
    print(f"feedback: {n0} rows in, {len(fb)} rows kept "
          f"(accept rate {fb.action.mean():.1%}, {fb.user_id.nunique()} users)")
    return fb.reset_index(drop=True)


if __name__ == "__main__":
    users = clean_users()
    fb = clean_feedback(users)
    users.to_csv("users_clean.csv", index=False)
    fb.to_csv("feedback_clean.csv", index=False)
    pd.DataFrame([[mbti_rule(a, b) for b in MBTI_TYPES] for a in MBTI_TYPES],
                 index=MBTI_TYPES, columns=MBTI_TYPES).to_csv("mbti_matrix.csv")
    print("written: users_clean.csv, feedback_clean.csv, mbti_matrix.csv")
    unknown = set(users.city) - set(CITIES)
    if unknown:
        print("note: cities not in the app's city list:", sorted(unknown))
