# Profile-Based Matching Algorithm
### An Adaptive Hybrid Recommendation System Using NLP, Personality Compatibility, Geographical Proximity, and Feedback-Based Learning

## 1. Project Overview

The **Profile-Based Matching Algorithm** is an intelligent, adaptive recommendation system developed using Python to identify and rank the most suitable profiles based on multiple dimensions of compatibility.

Unlike conventional recommendation systems that primarily rely on keyword matching or textual similarity, this project adopts a **hybrid matching approach** that integrates Natural Language Processing (NLP), personality compatibility, and geographical proximity to generate more comprehensive and personalized recommendations.

The system evaluates profiles across three primary dimensions:

- **Professional Similarity:** Measures the similarity between users' professional summaries and descriptions using TF-IDF vectorization and cosine similarity.
- **Personality Compatibility:** Evaluates compatibility between MBTI personality types using a predefined compatibility matrix.
- **Geographical Proximity:** Calculates the geographical distance between profiles using the Haversine formula and converts it into a compatibility score.

A key feature of the system is its ability to learn from user interactions. Through explicit **Accept/Reject feedback**, a personalized logistic regression model learns the relative importance of different matching components and dynamically adjusts their weights.

The updated weights are subsequently used to recalculate compatibility scores and re-rank recommendations.

The complete solution is implemented as an interactive **Streamlit web application**, providing profile-based recommendations, match explanations, feedback collection, performance analysis, and downloadable results.

---

## 2. Problem Statement

Traditional profile-matching and recommendation systems often depend heavily on textual similarity or predefined filtering criteria. Although these approaches can identify profiles containing similar keywords, they may fail to account for other factors that influence compatibility.

For example, two individuals may have similar professional backgrounds but differ significantly in personality preferences or geographical accessibility.

Additionally, conventional static recommendation systems generally do not adapt their matching criteria based on individual user preferences.

This project addresses these limitations by developing a multi-dimensional recommendation system that:

1. Evaluates profiles using professional information, personality characteristics, and geographical location.
2. Combines multiple compatibility measures into a unified matching score.
3. Collects explicit user feedback through Accept and Reject interactions.
4. Learns individual matching preferences using logistic regression.
5. Dynamically updates feature weights and re-ranks recommendations.

**Proposed Solution:** A personalized hybrid recommendation framework that integrates text-based similarity, personality compatibility, location-based scoring, and feedback-driven learning.

---

## 3. Project Objectives

The primary objectives of this project are:

- Develop a multi-factor profile-matching algorithm.
- Implement NLP-based text similarity using TF-IDF.
- Incorporate MBTI-based personality compatibility.
- Calculate geographical proximity using the Haversine distance formula.
- Develop a weighted scoring mechanism to combine different compatibility measures.
- Implement feedback-based personalization using logistic regression and gradient descent.
- Dynamically update matching weights based on user interactions.
- Generate and display the top five recommended profiles.
- Evaluate recommendation performance using ROC-AUC and top-5 acceptance rate.
- Build an interactive application using Streamlit.

---

## 4. Key Features

| Feature | Description |
|---|---|
| Hybrid Recommendation Engine | Combines text, personality, and geographical compatibility. |
| NLP-Based Matching | Uses TF-IDF and cosine similarity for profile comparison. |
| Personality Compatibility | Evaluates MBTI compatibility using a predefined matrix. |
| Location-Based Matching | Calculates geographical distance using the Haversine formula. |
| Weighted Scoring | Combines individual compatibility scores into a unified score. |
| Personalized Learning | Learns individual preferences from Accept/Reject feedback. |
| Dynamic Weight Adjustment | Updates the importance of matching components for each user. |
| Automatic Re-ranking | Recalculates and reorders recommendations after feedback. |
| Top-5 Recommendations | Displays the five highest-ranked candidate profiles. |
| Explainable Matching | Provides an individual score breakdown for each recommendation. |
| Performance Dashboard | Displays feedback trends, weight changes, and model evaluation results. |
| CSV Export | Allows users to download their recommended profiles. |

---

## 5. System Architecture

The proposed system consists of three primary matching components, followed by a feedback-based personalization layer.

### Overall Architecture

```text
                       USER PROFILE
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
        TEXT MATCHING   MBTI MATCHING   LOCATION MATCHING
             |              |              |
             v              v              v
        TF-IDF AND       COMPATIBILITY   HAVERSINE
        COSINE SIMILARITY   MATRIX       DISTANCE
             |              |              |
             +--------------+--------------+
                            |
                            v
                   INITIAL WEIGHTED SCORE
                            |
                            v
                    FINAL MATCH SCORES
                            |
                            v
                   TOP 5 RECOMMENDATIONS
                            |
                            v
                    USER INTERACTIONS
                      ACCEPT / REJECT
                            |
                            v
                   LOGISTIC REGRESSION
                      AND GRADIENT
                       DESCENT
                            |
                            v
                  PERSONALIZED WEIGHTS
                            |
                            v
                    SCORE RECALCULATION
                            |
                            v
                      RE-RANKING
                            |
                            v
                  PERSONALIZED RESULTS
```

The architecture separates the matching process into independent scoring components, allowing each compatibility dimension to be calculated individually before being integrated into the final recommendation score.

---

## 6. Dataset Description

The project uses structured profile and feedback datasets stored in CSV format.

### 6.1 User Dataset – `users.csv`

Contains **120 simulated user profiles with 21 columns**.

The dataset includes profile attributes required for calculating text similarity, personality compatibility, and geographical proximity.

These attributes provide the input features for the recommendation engine.

### 6.2 Feedback Dataset – `feedback.csv`

Contains **1,083 simulated Accept/Reject feedback records**.

The feedback is represented numerically:

- Accept = 1
- Reject = 0

These interactions are used to train the personalized feedback model and estimate the importance of individual matching components.

The dataset also supports performance evaluation and analysis of recommendation behaviour.

**Note:** The project uses synthetic data with simulated user preferences. The evaluation results therefore demonstrate the model's ability to learn the simulated preferences rather than establish performance on real-world users.

---

## 7. Methodology and Implementation

### 7.1 Text Preprocessing

Before calculating textual similarity, profile descriptions are processed to remove unnecessary information and standardize the text.

The project uses the NLTK library for Natural Language Processing tasks.

The preprocessing pipeline includes:

1. **Lowercasing:** Converts all text into lowercase to maintain consistency.
2. **Punctuation Removal:** Removes punctuation marks and unnecessary special characters.
3. **Stopword Removal:** Eliminates commonly occurring words that contribute little to meaningful comparison.
4. **Lemmatization:** Converts words into their base or dictionary forms.

**Example:**

Input:

```text
"I am developing Machine Learning applications!"
```

After preprocessing:

```text
"develop machine learn application"
```

This standardized representation improves the consistency of subsequent text comparisons.

### 7.2 TF-IDF-Based Text Matching

The system uses **Term Frequency–Inverse Document Frequency (TF-IDF)** to convert textual profile information into numerical feature vectors.

TF-IDF determines the importance of words by considering their frequency within a document and their distribution across the entire collection of documents.

The implementation incorporates:

- TF-IDF vectorization.
- Unigram and bigram features.
- Sublinear term frequency scaling.
- Cosine similarity for comparing profile vectors.

Two independent textual similarity scores are calculated.

**A. Professional Summary Similarity – 60%**

Measures the similarity between the professional summaries of two profiles. It captures common professional interests, skills, and career-related information.

**B. About Me Similarity – 40%**

Measures similarity between personal descriptions, interests, and other self-reported information.

The combined text score is calculated as:

```text
Text Score =
(0.6 × Professional Summary Similarity)
+
(0.4 × About Me Similarity)
```

Cosine similarity measures the angular similarity between two TF-IDF vectors, producing a score that represents how closely the textual descriptions are related.

### 7.3 MBTI Personality Compatibility

In addition to textual similarity, the system incorporates personality compatibility using the Myers-Briggs Type Indicator (MBTI).

MBTI classifies personality preferences into 16 different personality types based on four dimensions:

- Introversion (I) / Extraversion (E)
- Intuition (N) / Sensing (S)
- Thinking (T) / Feeling (F)
- Judging (J) / Perceiving (P)

The system uses a predefined **MBTI compatibility matrix** to assign compatibility scores to different personality-type combinations.

Rather than requiring exact personality matches, the compatibility mechanism evaluates predefined relationships between different personality types.

This produces a numerical personality compatibility score that contributes to the overall matching process.

### 7.4 Geographical Location Matching

The system considers geographical proximity as another important compatibility factor.

It uses the **Haversine formula** to calculate the great-circle distance between two geographical coordinates on the Earth's surface.

The Haversine formula is:

```text
a = sin²(Δφ / 2)
    + cos(φ1) × cos(φ2) × sin²(Δλ / 2)

c = 2 × atan2(√a, √(1-a))

Distance = R × c
```

Where:

- φ1 and φ2 represent the latitudes of two locations.
- Δφ represents the difference in latitude.
- Δλ represents the difference in longitude.
- R represents the Earth's mean radius.

The calculated distance is converted into a location compatibility score using a location-decay function.

The scoring mechanism follows the principle:

- Smaller geographical distance results in higher location compatibility.
- Greater geographical distance results in lower location compatibility.

This allows geographical proximity to be incorporated into the final recommendation score.

### 7.5 Initial Weighted Matching Score

After calculating the three individual compatibility scores, the system combines them using a weighted scoring mechanism.

The initial weights are:

| Matching Component | Initial Weight |
|---|---:|
| Text Similarity | 50% |
| MBTI Compatibility | 30% |
| Location Compatibility | 20% |

The overall matching score is calculated as:

```text
Final Match Score =

(0.5 × Text Score)
+
(0.3 × MBTI Score)
+
(0.2 × Location Score)
```

Each component contributes according to its assigned weight.

The resulting score is normalized to a percentage scale between 0 and 100, making recommendations easier to interpret.

These initial weights provide a common starting point before personalized feedback is introduced.

---

## 8. Personalized Feedback Learning

One of the major contributions of this project is its ability to adapt recommendations based on individual user preferences.

Instead of maintaining fixed weights throughout the recommendation process, the system learns from explicit user interactions.

### 8.1 Accept/Reject Feedback

Users can evaluate recommended profiles using two actions:

- **Accept (1):** Indicates that the recommended profile is suitable or preferred.
- **Reject (0):** Indicates that the recommended profile is not suitable or preferred.

These interactions are collected and used as training observations for the feedback-learning model.

### 8.2 Logistic Regression Model

The system implements a personalized logistic regression model trained using gradient descent.

The three matching components serve as input features:

```text
Input Features:

1. Text Similarity
2. MBTI Compatibility
3. Location Compatibility
```

The model learns the relationship between these compatibility scores and the user's Accept/Reject decisions.

The logistic regression model estimates the probability of acceptance using:

```text
P(Accept) = σ(z)

z = w1(Text Score)
  + w2(MBTI Score)
  + w3(Location Score)
  + b
```

Where:

- w1 represents the learned text weight.
- w2 represents the learned MBTI weight.
- w3 represents the learned location weight.
- b represents the model bias.
- σ represents the sigmoid activation function.

Gradient descent iteratively updates the model parameters to reduce prediction error.

The learned coefficients are subsequently normalized to obtain personalized matching weights.

### 8.3 Personalized Weight Adjustment

The normalized weights represent the relative importance of different compatibility dimensions for an individual user.

For example:

**User A – Professionally Focused**

```text
Text Similarity       → High Importance
MBTI Compatibility    → Lower Importance
Location Compatibility → Moderate Importance
```

**User B – Personality Focused**

```text
Text Similarity       → Moderate Importance
MBTI Compatibility    → High Importance
Location Compatibility → Lower Importance
```

The system can therefore adapt the recommendation criteria according to the feedback collected from individual users.

### 8.4 Dynamic Re-ranking

After receiving feedback, the model updates personalized weights and recalculates the matching scores.

The updated scores are used to re-rank the candidate profiles.

```text
Initial Recommendations
          |
          v
     User Feedback
          |
          v
   Feedback Collection
          |
          v
  Logistic Regression
          |
          v
  Updated Feature Weights
          |
          v
  Recalculate Match Scores
          |
          v
     Re-rank Profiles
          |
          v
 Personalized Recommendations
```

This creates an adaptive recommendation process in which subsequent recommendations can reflect the user's previously expressed preferences.

---

## 9. Streamlit Web Application

The complete recommendation system is integrated into an interactive web application developed using **Streamlit**.

The application provides an accessible interface for entering profile information, exploring recommendations, providing feedback, and monitoring personalization.

### 9.1 Profile Input

Users can enter the profile information required by the recommendation engine.

The entered information is processed by the matching engine to calculate compatibility scores against available candidate profiles.

### 9.2 Top Matches Dashboard

Displays the five highest-ranked profiles based on their overall compatibility scores.

The dashboard includes:

- Top-five recommended profiles.
- Overall match percentage for each profile.
- Individual score breakdown.
- Accept and Reject feedback controls.
- Live personalized-weight indicator.
- CSV export of recommended profiles.

Each recommendation includes an individual score breakdown, allowing users to understand the contribution of the different matching components.

### 9.3 Session Report

The session report becomes available after users provide feedback.

It summarizes the recommendation session through:

- Changes in personalized matching weights.
- Common characteristics of accepted and rejected profiles.
- MBTI pairings associated with user decisions.
- A structured table of recorded feedback.

This provides visibility into how user preferences influence the recommendation process.

### 9.4 Model Performance Dashboard

The performance dashboard displays information derived from the user's actual feedback interactions.

It includes:

- Changes in personalized weights after each interaction.
- Predicted scores before feedback was submitted.
- The relationship between individual matching signals and user decisions.
- Visualizations of the learning process.

An optional offline benchmark is also available after executing `python analysis.py`.

The benchmark evaluates the system using the 120 simulated profiles and displays the corresponding performance metrics.

### 9.5 Application Interface

The application uses:

- Streamlit for the interactive web interface.
- Playfair Display for headings.
- Inter for general interface text.
- Custom theme configuration.
- Interactive dashboard components.

Both bundled fonts are distributed under the SIL Open Font License.

---

## 10. Performance Evaluation

The system includes an offline evaluation module implemented in `analysis.py`.

The evaluation uses synthetic user profiles with simulated hidden preferences. These preferences determine the probability of accepting or rejecting candidate profiles.

The model does not directly receive these hidden preference weights during training. They are used for generating simulated feedback and establishing an oracle reference for comparison.

### 10.1 Hold-Out ROC-AUC Evaluation

The dataset is divided chronologically for each eligible user:

- First 60% of feedback records: Training data.
- Remaining 40%: Testing data.

The evaluation calculates the mean per-user ROC-AUC across 87 users who have both positive and negative outcomes in their test sets.

| Weighting Strategy | ROC-AUC |
|---|---:|
| Default Weights (0.5 / 0.3 / 0.2) | 0.834 |
| Learned Personalized Weights | 0.871 |
| Oracle (Hidden True Weights) | 0.921 |

**Observation:** The learned personalized model achieves a ROC-AUC of 0.871 compared with 0.834 for the default weighting strategy.

This indicates improved discrimination between accepted and rejected profiles within the simulated evaluation setting.

### 10.2 Top-5 Acceptance Rate

The second evaluation measures the percentage of recommended top-five profiles accepted by simulated users.

The experiment uses:

- Eight recommendation rounds.
- A fresh pool of 25 candidates per round.
- Five random seeds.
- A cold-start recommendation setting.

| Recommendation Strategy | Round 1 | Round 8 |
|---|---:|---:|
| Static Default Weights | 70% | 72% |
| Adaptive Personalized Weights | 70% | 87% |

**Observation:** The adaptive approach improves the top-five acceptance rate from 70% in the first round to 87% by the eighth round, whereas static default weighting reaches 72%.

The results demonstrate the potential benefit of incorporating user feedback into the recommendation process.

**Evaluation Limitation:** Since the hidden preferences are generated using the same three compatibility features available to the model, the results demonstrate recovery of simulated preferences. They should not be interpreted as validated performance on real-world users.

The complete performance report, charts, and evaluation outputs are generated in the `results/` directory by executing `analysis.py`.

---

## 11. Technology Stack

| Technology | Application in the Project |
|---|---|
| Python | Core programming language and algorithm implementation |
| Pandas | Dataset processing and manipulation |
| NumPy | Numerical computations and mathematical operations |
| NLTK | Text preprocessing, stopword removal, and lemmatization |
| Scikit-learn | TF-IDF vectorization, cosine similarity, and ROC-AUC evaluation |
| Logistic Regression | Learning personalized user preferences |
| Gradient Descent | Optimization of model parameters |
| Streamlit | Interactive web application and dashboards |
| Matplotlib | Performance visualization |
| Seaborn | Statistical data visualization |
| CSV | Profile and feedback data storage |

### Core Machine Learning and Algorithmic Concepts

- Natural Language Processing.
- TF-IDF Vectorization.
- Cosine Similarity.
- Feature Engineering and Weighting.
- Logistic Regression.
- Gradient Descent.
- Recommendation Systems.
- Personalized Learning.
- MBTI Compatibility Scoring.
- Haversine Distance Calculation.
- Model Evaluation using ROC-AUC.

---

## 12. Project Directory Structure

```text
Profile-Based-Matching/
│
├── app.py
│   └── Streamlit application and interactive dashboards
│
├── matching.py
│   └── Core matching engine, NLP, scoring,
│       feedback learning, and recommendation logic
│
├── generate_data.py
│   └── Generates simulated user profiles,
│       feedback records, and hidden preferences
│
├── data_pipeline.py
│   └── Data validation, preprocessing,
│       and MBTI compatibility matrix generation
│
├── analysis.py
│   └── Model evaluation, performance analysis,
│       charts, and report generation
│
├── users.csv
│   └── Dataset containing 120 user profiles
│
├── feedback.csv
│   └── Dataset containing 1,083 feedback records
│
├── requirements.txt
│   └── Project dependencies
│
├── .streamlit/
│   └── config.toml
│       └── Application theme and configuration
│
├── static/
│   └── fonts/
│       └── Playfair Display and Inter fonts
│
├── results/
│   └── Generated charts, CSV files,
│       metrics.json, and performance_report.md
│
└── README.md
```

---

## 13. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone <repository-url>
```

### Step 2: Navigate to the Project Directory

```bash
cd Profile-Based-Matching
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Launch the Application

```bash
streamlit run app.py
```

The application will launch in the default web browser.

Required NLTK resources, including stopwords and WordNet, are downloaded during the initial application startup.

### Step 5: Execute Additional Modules (Optional)

Generate the datasets:

```bash
python generate_data.py
```

Validate and preprocess the datasets:

```bash
python data_pipeline.py
```

Run the offline performance evaluation:

```bash
python analysis.py
```

Execute the matching engine independently:

```bash
python matching.py
```

### Using the Matching Engine Programmatically

The matching engine can also be imported and used independently of the Streamlit interface.

```python
from matching import MatchEngine

eng = MatchEngine()

# Calculate compatibility between two users
eng.match_score("U001", "U025")

# Retrieve the top five recommendations
eng.top_matches("U001", k=5)
```

The `match_score()` function returns the compatibility score between two user profiles, while `top_matches()` retrieves the highest-ranked recommendations.

---

## 14. Example of Recommendation Scoring

Consider two profiles being evaluated by the recommendation engine.

Assume the following compatibility scores:

| Matching Component | Similarity Score |
|---|---:|
| Professional Summary | 85% |
| About Me | 75% |
| MBTI Compatibility | 80% |
| Location Compatibility | 90% |

First, calculate the combined text score:

```text
Text Score =
(0.6 × 85) + (0.4 × 75)

Text Score = 81%
```

Next, calculate the overall matching score:

```text
Final Score =
(0.5 × 81)
+
(0.3 × 80)
+
(0.2 × 90)

Final Score = 82.5%
```

Therefore, the overall compatibility score is **82.5%**.

The recommendation engine performs similar calculations for available profiles and ranks them according to their final scores.

---

## 15. Limitations

Although the system incorporates multiple recommendation techniques and personalized learning, certain limitations remain:

1. **Synthetic Dataset:** The current implementation uses simulated profiles and feedback, limiting the generalizability of evaluation results to real-world users.
2. **MBTI-Based Compatibility:** Personality compatibility depends on predefined rules and should not be considered a scientifically validated measure of interpersonal compatibility.
3. **Location Accuracy:** The quality of geographical matching depends on the accuracy and availability of location coordinates.
4. **Feedback Dependency:** Personalized learning requires sufficient user interactions to estimate meaningful preferences.
5. **Cold-Start Problem:** New users initially receive recommendations based on default weights until sufficient feedback becomes available.
6. **Feature Limitations:** The current recommendation engine considers three primary matching dimensions and may not capture every factor influencing compatibility.

---

## 16. Future Enhancements

The project can be extended through the following improvements:

- **Semantic Text Matching:** Integrate Sentence-BERT or transformer-based embeddings to capture contextual meaning beyond keyword similarity.
- **Advanced Recommendation Models:** Explore collaborative filtering and hybrid recommendation techniques.
- **Enhanced Personalization:** Investigate more sophisticated online-learning and feedback-optimization methods.
- **Expanded Matching Features:** Incorporate additional profile attributes, interests, skills, and user preferences.
- **Real-World Dataset Integration:** Evaluate the system using larger, more diverse, and representative datasets.
- **Database Integration:** Replace CSV-based storage with a structured database for scalable data management.
- **Authentication and User Management:** Introduce individual accounts and persistent user profiles.
- **Cloud Deployment:** Deploy the application using a scalable cloud infrastructure.
- **Advanced Explainability:** Provide more detailed explanations of why particular profiles are recommended.

---

## 17. Conclusion

The **Profile-Based Matching Algorithm** demonstrates the development of an adaptive hybrid recommendation system that combines Natural Language Processing, personality compatibility, geographical proximity, and feedback-based machine learning.

By integrating TF-IDF-based text similarity, MBTI compatibility scoring, and Haversine distance calculations, the system evaluates profiles across multiple dimensions rather than relying exclusively on textual similarity.

Its primary distinguishing feature is the integration of a personalized feedback-learning mechanism using logistic regression and gradient descent. This enables the system to adjust matching weights based on individual Accept/Reject interactions and dynamically re-rank recommendations.

The Streamlit application provides an interactive environment for exploring recommendations, understanding compatibility scores, submitting feedback, and monitoring model behaviour.

The offline evaluation further demonstrates how adaptive weighting can improve recommendation outcomes within a controlled synthetic environment.

**Core Concept:**

```text
MULTI-DIMENSIONAL PROFILE ANALYSIS
                 |
                 v
       HYBRID MATCHING ENGINE
                 |
                 v
       PERSONALIZED RECOMMENDATIONS
                 |
                 v
          USER FEEDBACK
                 |
                 v
       MACHINE LEARNING MODEL
                 |
                 v
       ADAPTIVE WEIGHT LEARNING
                 |
                 v
        DYNAMIC RE-RANKING
                 |
                 v
       IMPROVED PERSONALIZATION
```

**Project Focus:** Building an intelligent, explainable, and adaptive recommendation system that evolves with user preferences.