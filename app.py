import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Personality AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# IMPORTANT: PERSONALITY LABEL DECODING
# ============================================================
#
# Your screenshot shows prediction "0".
#
# That means your target was most likely encoded into numbers
# before training, for example:
#
#     Extrovert -> 0
#     Introvert -> 1
#
# If you used sklearn LabelEncoder, the mapping is normally
# alphabetical. For ["Extrovert", "Introvert"]:
#
#     0 -> Extrovert
#     1 -> Introvert
#
# If your original encoding was different, ONLY change the
# PERSONALITY_LABELS dictionary below.
#
# If you have label_encoder.pkl, this application will use it
# automatically instead.
# ============================================================

PERSONALITY_LABELS = {
    0: "Extrovert",
    1: "Introvert"
}


# ============================================================
# LOAD MODEL / SCALER / OPTIONAL LABEL ENCODER
# ============================================================

@st.cache_resource
def load_artifacts():

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    label_encoder = None

    try:
        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
    except FileNotFoundError:
        pass

    return model, scaler, label_encoder


try:

    model, scaler, label_encoder = load_artifacts()

except FileNotFoundError as e:

    st.error("Required model files are missing.")

    st.write("Your folder must contain:")

    st.code(
        "app.py\n"
        "model.pkl\n"
        "scaler.pkl\n"
        "requirements.txt\n"
        "label_encoder.pkl  (optional)"
    )

    st.stop()

except Exception as e:

    st.error("Could not load your model files.")

    st.exception(e)

    st.stop()


# ============================================================
# MODEL FEATURES
# ============================================================
#
# Your original training code was:
#
# X = df.drop(
#     columns=[
#         'personality_type',
#         'emotional_stability',
#         'stress_handling',
#         'creativity'
#     ]
# )
#
# Therefore this model expects 26 input features.
# ============================================================

FEATURES = [
    "social_energy",
    "alone_time_preference",
    "talkativeness",
    "deep_reflection",
    "group_comfort",
    "party_liking",
    "listening_skill",
    "empathy",
    "organization",
    "leadership",
    "risk_taking",
    "public_speaking_comfort",
    "curiosity",
    "routine_preference",
    "excitement_seeking",
    "friendliness",
    "planning",
    "spontaneity",
    "adventurousness",
    "reading_habit",
    "sports_interest",
    "online_social_usage",
    "travel_desire",
    "gadget_usage",
    "work_style_collaborative",
    "decision_speed"
]


SOCIAL_FEATURES = [
    "social_energy",
    "alone_time_preference",
    "talkativeness",
    "deep_reflection",
    "group_comfort",
    "party_liking",
    "listening_skill",
    "empathy",
    "friendliness",
    "public_speaking_comfort"
]


BEHAVIOR_FEATURES = [
    "organization",
    "leadership",
    "risk_taking",
    "curiosity",
    "routine_preference",
    "excitement_seeking",
    "planning",
    "spontaneity",
    "adventurousness",
    "decision_speed"
]


LIFESTYLE_FEATURES = [
    "reading_habit",
    "sports_interest",
    "online_social_usage",
    "travel_desire",
    "gadget_usage",
    "work_style_collaborative"
]


# ============================================================
# VALIDATE MODEL INPUT SIZE
# ============================================================

if hasattr(model, "n_features_in_"):

    if model.n_features_in_ != len(FEATURES):

        st.error(
            f"Model mismatch: model expects "
            f"{model.n_features_in_} features, but this app "
            f"provides {len(FEATURES)}."
        )

        st.stop()


if hasattr(scaler, "n_features_in_"):

    if scaler.n_features_in_ != len(FEATURES):

        st.error(
            f"Scaler mismatch: scaler expects "
            f"{scaler.n_features_in_} features, but this app "
            f"provides {len(FEATURES)}."
        )

        st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def pretty_name(name):

    return name.replace("_", " ").title()


def average_score(feature_list, values):

    return round(
        np.mean(
            [
                values[feature]
                for feature in feature_list
            ]
        ),
        2
    )


def score_level(score):

    if score < 2:
        return "Very Low"

    if score < 3:
        return "Low"

    if score < 4:
        return "Moderate"

    if score < 5:
        return "High"

    return "Very High"


def decode_personality(value):

    # --------------------------------------------------------
    # If a LabelEncoder was saved, always use it.
    # --------------------------------------------------------

    if label_encoder is not None:

        try:

            decoded = label_encoder.inverse_transform(
                [int(value)]
            )[0]

            return str(decoded)

        except Exception:
            pass


    # --------------------------------------------------------
    # If model already returns text, use it directly.
    # --------------------------------------------------------

    if isinstance(value, str):

        return value


    # --------------------------------------------------------
    # Numeric encoded target.
    # --------------------------------------------------------

    try:

        numeric_value = int(value)

        if numeric_value in PERSONALITY_LABELS:

            return PERSONALITY_LABELS[numeric_value]

    except Exception:
        pass


    # --------------------------------------------------------
    # Last fallback.
    # --------------------------------------------------------

    return str(value)


def decode_class(value):

    return decode_personality(value)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction" not in st.session_state:

    st.session_state.prediction = None


if "raw_prediction" not in st.session_state:

    st.session_state.raw_prediction = None


if "probabilities" not in st.session_state:

    st.session_state.probabilities = None


if "input_values" not in st.session_state:

    st.session_state.input_values = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 Personality AI")

    st.caption(
        "Personality Intelligence & Prediction System"
    )

    st.divider()

    st.subheader("Model")

    st.write("**Algorithm:** Logistic Regression")

    st.write("**Preprocessing:** StandardScaler")

    st.write(f"**Input Features:** {len(FEATURES)}")

    st.divider()

    st.subheader("Rating Scale")

    st.write("1 — Very Low")
    st.write("2 — Low")
    st.write("3 — Moderate")
    st.write("4 — High")
    st.write("5 — Very High")

    st.divider()

    st.subheader("Prediction Classes")

    # Show decoded model classes in the sidebar.

    if hasattr(model, "classes_"):

        for cls in model.classes_:

            st.write(
                f"• {decode_class(cls)}"
            )

    st.divider()

    if st.button(
        "🔄 Reset Assessment",
        use_container_width=True
    ):

        st.session_state.prediction = None
        st.session_state.raw_prediction = None
        st.session_state.probabilities = None
        st.session_state.input_values = None

        for feature in FEATURES:

            st.session_state[
                f"slider_{feature}"
            ] = 3

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧠 Personality Intelligence Dashboard"
)

st.write(
    "Analyze social behavior, behavioral tendencies and "
    "lifestyle preferences using your trained machine "
    "learning personality classifier."
)

st.divider()


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Model",
        "Logistic Regression"
    )

with c2:

    st.metric(
        "Input Features",
        "26"
    )

with c3:

    st.metric(
        "Scaler",
        "StandardScaler"
    )

with c4:

    if st.session_state.prediction is not None:

        st.metric(
            "Current Result",
            st.session_state.prediction
        )

    else:

        st.metric(
            "Current Result",
            "Pending"
        )


# ============================================================
# ASSESSMENT
# ============================================================

st.header(
    "📝 Personality Assessment"
)

st.caption(
    "Rate each characteristic from 1 to 5. "
    "These 26 values are passed to your trained model."
)


with st.form(
    "personality_form"
):

    values = {}


    # ========================================================
    # SOCIAL
    # ========================================================

    st.subheader(
        "👥 Social & Communication"
    )

    st.caption(
        "Communication, interaction and interpersonal tendencies."
    )

    cols = st.columns(2)

    for i, feature in enumerate(
        SOCIAL_FEATURES
    ):

        with cols[i % 2]:

            values[feature] = st.slider(
                pretty_name(feature),
                min_value=1,
                max_value=5,
                value=st.session_state.get(
                    f"slider_{feature}",
                    3
                ),
                step=1,
                key=f"slider_{feature}"
            )


    st.divider()


    # ========================================================
    # BEHAVIOR
    # ========================================================

    st.subheader(
        "🧩 Behavioral Characteristics"
    )

    st.caption(
        "Planning, curiosity, risk-taking and decision-making."
    )

    cols = st.columns(2)

    for i, feature in enumerate(
        BEHAVIOR_FEATURES
    ):

        with cols[i % 2]:

            values[feature] = st.slider(
                pretty_name(feature),
                min_value=1,
                max_value=5,
                value=st.session_state.get(
                    f"slider_{feature}",
                    3
                ),
                step=1,
                key=f"slider_{feature}"
            )


    st.divider()


    # ========================================================
    # LIFESTYLE
    # ========================================================

    st.subheader(
        "🌎 Lifestyle & Preferences"
    )

    st.caption(
        "Interests, technology, travel and everyday preferences."
    )

    cols = st.columns(2)

    for i, feature in enumerate(
        LIFESTYLE_FEATURES
    ):

        with cols[i % 2]:

            values[feature] = st.slider(
                pretty_name(feature),
                min_value=1,
                max_value=5,
                value=st.session_state.get(
                    f"slider_{feature}",
                    3
                ),
                step=1,
                key=f"slider_{feature}"
            )


    st.divider()


    submitted = st.form_submit_button(
        "🧠 Generate Personality Analysis",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    try:

        # ----------------------------------------------------
        # EXACT FEATURE ORDER
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [
                [
                    values[feature]
                    for feature in FEATURES
                ]
            ],
            columns=FEATURES
        )


        # ----------------------------------------------------
        # SCALE
        # ----------------------------------------------------

        scaled_input = scaler.transform(
            input_df
        )


        # ----------------------------------------------------
        # RAW MODEL PREDICTION
        # ----------------------------------------------------

        raw_prediction = model.predict(
            scaled_input
        )[0]


        # ----------------------------------------------------
        # DECODE NUMERIC CLASS INTO TEXT
        # ----------------------------------------------------

        text_prediction = decode_personality(
            raw_prediction
        )


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        probabilities = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                scaled_input
            )[0]


        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        st.session_state.raw_prediction = raw_prediction

        st.session_state.prediction = text_prediction

        st.session_state.probabilities = probabilities

        st.session_state.input_values = values

        st.success(
            f"Prediction generated: {text_prediction}"
        )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)


# ============================================================
# REPORT
# ============================================================

if st.session_state.prediction is not None:

    prediction = st.session_state.prediction

    raw_prediction = st.session_state.raw_prediction

    probabilities = st.session_state.probabilities

    values = st.session_state.input_values


    st.divider()

    st.header(
        "📊 Personality Intelligence Report"
    )


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    result_left, result_right = st.columns(
        [1.5, 1]
    )


    with result_left:

        st.subheader(
            "Predicted Personality"
        )

        # IMPORTANT:
        # Display decoded text, NOT raw 0/1.

        st.success(
            f"### {prediction}"
        )

        st.caption(
            f"Model class: {raw_prediction}"
        )

        st.write(
            f"Based on your responses, the trained "
            f"Logistic Regression model classified your "
            f"profile as **{prediction}**."
        )


    with result_right:

        if probabilities is not None:

            confidence = float(
                np.max(probabilities) * 100
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                min(
                    confidence / 100,
                    1.0
                )
            )

        else:

            st.info(
                "Probability scores are not available "
                "for this model."
            )


    # ========================================================
    # PERSONALITY DIMENSIONS
    # ========================================================

    st.subheader(
        "🧬 Personality Dimensions"
    )

    social_score = average_score(
        SOCIAL_FEATURES,
        values
    )

    behavior_score = average_score(
        BEHAVIOR_FEATURES,
        values
    )

    lifestyle_score = average_score(
        LIFESTYLE_FEATURES,
        values
    )


    d1, d2, d3 = st.columns(3)


    with d1:

        st.metric(
            "👥 Social Profile",
            f"{social_score}/5"
        )

        st.caption(
            score_level(social_score)
        )


    with d2:

        st.metric(
            "🧩 Behavioral Profile",
            f"{behavior_score}/5"
        )

        st.caption(
            score_level(behavior_score)
        )


    with d3:

        st.metric(
            "🌎 Lifestyle Profile",
            f"{lifestyle_score}/5"
        )

        st.caption(
            score_level(lifestyle_score)
        )


    # ========================================================
    # RADAR
    # ========================================================

    st.subheader(
        "🕸️ Personality Dimension Map"
    )

    radar_labels = [
        "Social",
        "Behavior",
        "Lifestyle",
        "Social"
    ]

    radar_values = [
        social_score,
        behavior_score,
        lifestyle_score,
        social_score
    ]


    radar = go.Figure()


    radar.add_trace(
        go.Scatterpolar(
            r=radar_values,
            theta=radar_labels,
            fill="toself",
            name="Personality Profile"
        )
    )


    radar.update_layout(
        height=450,
        showlegend=False,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                dtick=1
            )
        )
    )


    st.plotly_chart(
        radar,
        use_container_width=True
    )


    # ========================================================
    # PROBABILITY DISTRIBUTION
    # ========================================================

    if probabilities is not None:

        st.subheader(
            "📈 Personality Probability Distribution"
        )


        decoded_classes = [
            decode_class(cls)
            for cls in model.classes_
        ]


        probability_df = pd.DataFrame(
            {
                "Personality": decoded_classes,
                "Probability": probabilities * 100
            }
        ).sort_values(
            "Probability",
            ascending=False
        )


        probability_chart = px.bar(
            probability_df,
            x="Personality",
            y="Probability",
            text="Probability"
        )


        probability_chart.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )


        probability_chart.update_layout(
            height=450,
            yaxis_title="Probability (%)",
            xaxis_title="Personality"
        )


        st.plotly_chart(
            probability_chart,
            use_container_width=True
        )


        probability_table = probability_df.copy()


        probability_table[
            "Probability"
        ] = (
            probability_table[
                "Probability"
            ].round(2).astype(str)
            + "%"
        )


        st.dataframe(
            probability_table,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # FEATURE PROFILE
    # ========================================================

    st.subheader(
        "🔍 Complete Feature Profile"
    )


    feature_df = pd.DataFrame(
        {
            "Characteristic": [
                pretty_name(f)
                for f in FEATURES
            ],

            "Score": [
                values[f]
                for f in FEATURES
            ],

            "Level": [
                score_level(values[f])
                for f in FEATURES
            ]
        }
    )


    feature_chart = px.bar(
        feature_df.sort_values(
            "Score",
            ascending=True
        ),
        x="Score",
        y="Characteristic",
        orientation="h",
        text="Score"
    )


    feature_chart.update_traces(
        textposition="outside"
    )


    feature_chart.update_layout(
        height=800,
        xaxis=dict(
            range=[0, 5],
            dtick=1,
            title="Rating"
        ),
        yaxis_title=""
    )


    st.plotly_chart(
        feature_chart,
        use_container_width=True
    )


    # ========================================================
    # PROFILE HIGHLIGHTS
    # ========================================================

    st.subheader(
        "✨ Profile Highlights"
    )


    high = feature_df[
        feature_df["Score"] >= 4
    ].sort_values(
        "Score",
        ascending=False
    )


    low = feature_df[
        feature_df["Score"] <= 2
    ].sort_values(
        "Score"
    )


    h1, h2 = st.columns(2)


    with h1:

        st.write(
            "Higher Rated Characteristics"
        )

        if high.empty:

            st.info(
                "No characteristics were rated 4 or 5."
            )

        else:

            for _, row in high.iterrows():

                st.write(
                    f"• **{row['Characteristic']}** — "
                    f"{row['Score']}/5"
                )


    with h2:

        st.write(
            "Lower Rated Characteristics"
        )

        if low.empty:

            st.info(
                "No characteristics were rated 1 or 2."
            )

        else:

            for _, row in low.iterrows():

                st.write(
                    f"• **{row['Characteristic']}** — "
                    f"{row['Score']}/5"
                )


    # ========================================================
    # MODEL INPUT DATA
    # ========================================================

    st.subheader(
        "📋 Model Input Data"
    )


    input_display = pd.DataFrame(
        {
            "Feature": [
                pretty_name(f)
                for f in FEATURES
            ],

            "Value": [
                values[f]
                for f in FEATURES
            ]
        }
    )


    st.dataframe(
        input_display,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # EXPORT
    # ========================================================

    st.subheader(
        "💾 Export Prediction"
    )


    export_df = pd.DataFrame(
        {
            feature: [values[feature]]
            for feature in FEATURES
        }
    )


    export_df[
        "predicted_personality"
    ] = prediction


    export_df[
        "raw_model_class"
    ] = raw_prediction


    if probabilities is not None:

        export_df[
            "prediction_confidence"
        ] = float(
            np.max(probabilities) * 100
        )


    csv_data = export_df.to_csv(
        index=False
    )


    st.download_button(
        "⬇️ Download Personality Report",
        data=csv_data,
        file_name="personality_ai_report.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Personality AI • Logistic Regression • "
    "StandardScaler • 26 Behavioral Features"
)
