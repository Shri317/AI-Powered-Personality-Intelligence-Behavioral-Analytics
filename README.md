# 🧠 Personality AI — Personality Intelligence Dashboard

A Streamlit-based machine learning application that analyzes 26 behavioral and lifestyle characteristics and predicts a personality class using a trained **Logistic Regression** model.

The application combines a trained ML model, feature scaling, an interactive assessment interface, probability visualization, personality dimension summaries, feature-level analysis, and CSV report export.

## ✨ Features

- Interactive 1–5 personality assessment
- 26 model input features
- Logistic Regression classification
- StandardScaler preprocessing
- Text-based personality prediction
- Prediction probability visualization
- Social, Behavioral, and Lifestyle profile scores
- Radar chart for personality dimensions
- Complete feature profile
- Higher/lower rated characteristic highlights
- Model input data preview
- Downloadable CSV personality report
- Streamlit responsive dashboard layout

## 🧠 Model Pipeline

```text
User Assessment
      ↓
26 Behavioral / Lifestyle Features
      ↓
Pandas DataFrame
      ↓
StandardScaler
      ↓
Logistic Regression
      ↓
Personality Class
      ↓
Probability + Visual Report
```

The supplied model artifact reports **26 input features** and three model classes (`0`, `1`, `2`).

> **Important:** The current application contains a manual label mapping for `0 → Extrovert` and `1 → Introvert`. The supplied model also reports class `2`, so the exact human-readable name for class `2` should be confirmed from the original training/label-encoding process before treating that class as a named personality category.

## 📁 Project Structure

```text
personality-ai/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
├── .gitignore
├── LICENSE
├── README.md
└── .streamlit/
    └── config.toml
```

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/personality-ai.git
cd personality-ai
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## ☁️ Streamlit Community Cloud Deployment

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app.
4. Select your GitHub repository.
5. Set the main file to:
   `app.py`
6. Deploy.

The repository already includes `requirements.txt` and the required model artifacts.

## 📊 Input Features

### Social & Communication
- social_energy
- alone_time_preference
- talkativeness
- deep_reflection
- group_comfort
- party_liking
- listening_skill
- empathy
- friendliness
- public_speaking_comfort

### Behavioral Characteristics
- organization
- leadership
- risk_taking
- curiosity
- routine_preference
- excitement_seeking
- planning
- spontaneity
- adventurousness
- decision_speed

### Lifestyle & Preferences
- reading_habit
- sports_interest
- online_social_usage
- travel_desire
- gadget_usage
- work_style_collaborative

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Logistic Regression
- StandardScaler

## 📈 Dashboard Output

After submitting the assessment, the dashboard provides:

- Predicted personality class
- Prediction confidence when probabilities are available
- Social profile score
- Behavioral profile score
- Lifestyle profile score
- Personality dimension radar chart
- Personality probability distribution
- Complete feature-level chart
- Profile highlights
- Model input table
- CSV export

## 🔐 Model Files

`model.pkl` contains the trained Logistic Regression classifier.

`scaler.pkl` contains the StandardScaler used before model prediction.

These files are required for the Streamlit application to run.

### Security note

Only load pickle files that you trust. Pickle files can execute code during deserialization.

## 🚧 Known Limitation

The application currently defines:

```text
0 → Extrovert
1 → Introvert
```

but the supplied classifier exposes three classes:

```text
0, 1, 2
```

Before publishing a production/demo version, verify the original target-label encoding and add the correct label for class `2` (or include the original `label_encoder.pkl` if one was used during training).

## 🎯 Project Purpose

This project demonstrates an end-to-end machine learning workflow:

**Model → preprocessing → interactive application → prediction → visualization → report export**

It is intended as an educational and portfolio project demonstrating practical machine learning deployment with Streamlit.

## 👤 Author

**Shriyansh Barsaiya**

Information Technology Student  
Interests: Data Analytics • Data Science • Business Intelligence • Data Visualization

## 📄 License

This project is released under the MIT License.
