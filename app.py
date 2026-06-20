import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load models
xgb_final = pickle.load(open("models/xgb_final.pkl", "rb"))
le = pickle.load(open("models/label_encoder.pkl", "rb"))
rf_model = pickle.load(open("models/rf_model.pkl", "rb"))

# Load feature columns
df = pd.read_csv("data/processed/steam_features.csv")
df_clean = pd.read_csv("data/processed/steam_clean.csv")
feature_cols = [c for c in df.columns if c != 'success_tier']

# Initialize session state
if 'preset' not in st.session_state:
    st.session_state.preset = None

st.set_page_config(page_title="Steam Success Analyzer", layout="wide")

# Preset values
presets = {
    "aaa": {
        "price": 40, "release_year": 2018, "english": True,
        "dev_avg_score": 0.90, "dev_game_count": 20,
        "has_achievements": True, "platform_count": 2,
        "avg_playtime": 300, "genre_action": True,
        "genre_indie": False, "tag_action": True
    },
    "indie": {
        "price": 10, "release_year": 2020, "english": True,
        "dev_avg_score": 0.75, "dev_game_count": 3,
        "has_achievements": True, "platform_count": 2,
        "avg_playtime": 120, "genre_indie": True,
        "genre_casual": True, "tag_indie": True,
        "tag_puzzle": True
    },
    "f2p": {
        "price": 0, "release_year": 2019, "english": True,
        "dev_avg_score": 0.70, "dev_game_count": 5,
        "has_achievements": False, "platform_count": 1,
        "avg_playtime": 200, "genre_action": True,
        "tag_free_to_play": True
    }
}

# Get preset values or defaults
def get_val(key, default):
    if st.session_state.preset and key in presets[st.session_state.preset]:
        return presets[st.session_state.preset][key]
    return default

# Sidebar
st.sidebar.title("🎮 Game Configuration")

st.sidebar.header("📋 Game Details")
price = st.sidebar.slider("Price ($)", 0, 60, get_val("price", 20))
release_year = st.sidebar.selectbox("Release Year", list(range(2010, 2025)),
    index=list(range(2010, 2025)).index(get_val("release_year", 2020)))
english = st.sidebar.checkbox("Supports English", value=get_val("english", True))
dev_avg_score = st.sidebar.slider("Developer Avg Score (0-1)", 0.0, 1.0, float(get_val("dev_avg_score", 0.85)))
dev_game_count = st.sidebar.number_input("Developer Game Count", min_value=1, max_value=100, value=get_val("dev_game_count", 10))
has_achievements = st.sidebar.checkbox("Has Achievements", value=get_val("has_achievements", True))
platform_count = st.sidebar.slider("Number of Platforms", 1, 3, get_val("platform_count", 2))
avg_playtime = st.sidebar.slider("Average Playtime (minutes)", 0, 500, get_val("avg_playtime", 120))

st.sidebar.header("🎭 Genres")
genre_action = st.sidebar.checkbox("Action", value=get_val("genre_action", False))
genre_indie = st.sidebar.checkbox("Indie", value=get_val("genre_indie", False))
genre_rpg = st.sidebar.checkbox("RPG", value=get_val("genre_rpg", False))
genre_strategy = st.sidebar.checkbox("Strategy", value=get_val("genre_strategy", False))
genre_casual = st.sidebar.checkbox("Casual", value=get_val("genre_casual", False))
genre_adventure = st.sidebar.checkbox("Adventure", value=get_val("genre_adventure", False))
genre_simulation = st.sidebar.checkbox("Simulation", value=get_val("genre_simulation", False))
genre_sports = st.sidebar.checkbox("Sports", value=get_val("genre_sports", False))
genre_racing = st.sidebar.checkbox("Racing", value=get_val("genre_racing", False))
genre_early_access = st.sidebar.checkbox("Early Access", value=get_val("genre_early_access", False))

st.sidebar.header("🏷️ Tags")
tag_indie = st.sidebar.checkbox("Indie (Tag)", value=get_val("tag_indie", False))
tag_action = st.sidebar.checkbox("Action (Tag)", value=get_val("tag_action", False))
tag_adventure = st.sidebar.checkbox("Adventure (Tag)", value=get_val("tag_adventure", False))
tag_casual = st.sidebar.checkbox("Casual (Tag)", value=get_val("tag_casual", False))
tag_strategy = st.sidebar.checkbox("Strategy (Tag)", value=get_val("tag_strategy", False))
tag_simulation = st.sidebar.checkbox("Simulation (Tag)", value=get_val("tag_simulation", False))
tag_rpg = st.sidebar.checkbox("RPG (Tag)", value=get_val("tag_rpg", False))
tag_early_access = st.sidebar.checkbox("Early Access (Tag)", value=get_val("tag_early_access", False))
tag_free_to_play = st.sidebar.checkbox("Free to Play (Tag)", value=get_val("tag_free_to_play", False))
tag_puzzle = st.sidebar.checkbox("Puzzle (Tag)", value=get_val("tag_puzzle", False))
tag_racing = st.sidebar.checkbox("Racing (Tag)", value=get_val("tag_racing", False))
tag_sports = st.sidebar.checkbox("Sports (Tag)", value=get_val("tag_sports", False))
tag_vr = st.sidebar.checkbox("VR", value=get_val("tag_vr", False))
tag_platformer = st.sidebar.checkbox("Platformer", value=get_val("tag_platformer", False))
tag_anime = st.sidebar.checkbox("Anime", value=get_val("tag_anime", False))
tag_horror = st.sidebar.checkbox("Horror", value=get_val("tag_horror", False))
tag_visual_novel = st.sidebar.checkbox("Visual Novel", value=get_val("tag_visual_novel", False))
tag_point_click = st.sidebar.checkbox("Point & Click", value=get_val("tag_point_click", False))

predict_button = st.sidebar.button("🎮 Predict Success")

# Main page
st.title("🎮 Steam Game Success Analyzer")
st.write("Configure your game in the sidebar and click Predict Success to see the result.")

# Preset buttons
st.write("**Or try a preset:**")
col_a, col_b, col_c = st.columns(3)
with col_a:
    if st.button("🎯 AAA Action Game"):
        st.session_state.preset = "aaa"
        st.rerun()
with col_b:
    if st.button("🧩 Indie Puzzle Game"):
        st.session_state.preset = "indie"
        st.rerun()
with col_c:
    if st.button("🆓 Free to Play Game"):
        st.session_state.preset = "f2p"
        st.rerun()

st.divider()

# Prediction section
if predict_button:
    input_data = pd.DataFrame(columns=feature_cols)
    input_data.loc[0] = 0

    input_data['log_price'] = np.log1p(price)
    input_data['release_year'] = release_year
    input_data['english'] = int(english)
    input_data['dev_avg_score'] = dev_avg_score
    input_data['dev_game_count'] = dev_game_count
    input_data['has_achievements'] = int(has_achievements)
    input_data['platform_count'] = platform_count
    input_data['log_playtime'] = np.log1p(avg_playtime)
    input_data['genre_Action'] = int(genre_action)
    input_data['genre_Indie'] = int(genre_indie)
    input_data['genre_RPG'] = int(genre_rpg)
    input_data['genre_Strategy'] = int(genre_strategy)
    input_data['genre_Casual'] = int(genre_casual)
    input_data['genre_Adventure'] = int(genre_adventure)
    input_data['genre_Simulation'] = int(genre_simulation)
    input_data['genre_Sports'] = int(genre_sports)
    input_data['genre_Racing'] = int(genre_racing)
    input_data['genre_Early Access'] = int(genre_early_access)
    input_data['tag_Indie'] = int(tag_indie)
    input_data['tag_Action'] = int(tag_action)
    input_data['tag_Adventure'] = int(tag_adventure)
    input_data['tag_Casual'] = int(tag_casual)
    input_data['tag_Strategy'] = int(tag_strategy)
    input_data['tag_Simulation'] = int(tag_simulation)
    input_data['tag_RPG'] = int(tag_rpg)
    input_data['tag_Early Access'] = int(tag_early_access)
    input_data['tag_Free to Play'] = int(tag_free_to_play)
    input_data['tag_Puzzle'] = int(tag_puzzle)
    input_data['tag_Racing'] = int(tag_racing)
    input_data['tag_Sports'] = int(tag_sports)
    input_data['tag_VR'] = int(tag_vr)
    input_data['tag_Platformer'] = int(tag_platformer)
    input_data['tag_Anime'] = int(tag_anime)
    input_data['tag_Horror'] = int(tag_horror)
    input_data['tag_Visual Novel'] = int(tag_visual_novel)
    input_data['tag_Point & Click'] = int(tag_point_click)

    prediction = xgb_final.predict(input_data)[0]
    prediction_label = le.inverse_transform([prediction])[0]
    proba = xgb_final.predict_proba(input_data)[0]

    if prediction_label == 'High':
        st.success("# 🏆 High Success")
        st.write("### This game is likely to perform very well on Steam!")
    elif prediction_label == 'Medium':
        st.warning("# 📊 Medium Success")
        st.write("### This game has potential but faces some challenges.")
    else:
        st.error("# 📉 Low Success")
        st.write("### This game may struggle to gain traction on Steam.")

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Developer Avg Score", f"{dev_avg_score:.2f}")
    with col2:
        st.metric("Price", f"${price}")
    with col3:
        st.metric("Release Year", release_year)

    st.divider()

    col4, col5 = st.columns(2)
    with col4:
        st.write("**Confidence Scores:**")
        proba_df = pd.DataFrame({
            'Tier': le.classes_,
            'Probability': np.round(proba, 3)
        })
        st.bar_chart(proba_df.set_index('Tier'), height=300)

    with col5:
        st.write("**Top 10 Important Features:**")
        importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
        top_10 = importances.sort_values(ascending=True).tail(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        top_10.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title('Top 10 Important Features')
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.info("👈 Configure your game in the sidebar and click **Predict Success** to get started.")

st.divider()

# Tabs
tab1, tab2 = st.tabs(["📊 Data Insights", "ℹ️ About"])

with tab1:
    st.subheader("📈 Data Insights")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Review Score Distribution**")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        df_clean['review_score'].hist(bins=50, ax=ax1, color='steelblue', edgecolor='white')
        ax1.set_xlabel('Review Score')
        ax1.set_ylabel('Number of Games')
        st.pyplot(fig1)

    with col2:
        st.write("**Success Tier Breakdown**")
        tier_counts = df['success_tier'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        tier_counts.plot(kind='bar', ax=ax2, color=['#ff4444', '#ffaa00', '#44bb44'])
        ax2.set_xlabel('Success Tier')
        ax2.set_ylabel('Number of Games')
        plt.xticks(rotation=0)
        st.pyplot(fig2)

    st.write("**Games Released Per Year**")
    releases = df_clean.groupby('release_year').size()
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(releases.index, releases.values, color='steelblue', marker='o')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Number of Games')
    plt.tight_layout()
    st.pyplot(fig3)

    st.write("**Model Performance Comparison**")
    model_results = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'Tuned XGBoost'],
        'Accuracy': ['65%', '85%', '86%'],
        'High F1': ['0.34', '0.60', '0.63'],
        'Medium F1': ['0.33', '0.58', '0.61'],
        'Low F1': ['0.79', '0.93', '0.93']
    })
    st.table(model_results)

with tab2:
    st.subheader("About This Project")
    st.write("""
    **Steam Game Success Analyzer** is an end-to-end machine learning project that analyzes 
    over 20,000 Steam games to predict whether a newly released game will be successful.
    
    **Dataset:** Steam Games Dataset from Kaggle (~27,000 games)
    
    **Problem Type:** Multi-class Classification (Low / Medium / High Success)
    
    **Success Definition:**
    - 🏆 **High** — 85%+ positive reviews AND 2000+ total reviews
    - 📊 **Medium** — 70%+ positive reviews AND 200+ total reviews  
    - 📉 **Low** — Everything else
    
    **Key Findings:**
    - Developer reputation is the strongest predictor of success
    - Price has weak correlation with review score
    - Steam releases exploded after 2013
    - Playtime and platform count are strong success indicators
    
    **Tech Stack:** Python, Pandas, Scikit-learn, XGBoost, Streamlit, Matplotlib
    
    **Model Accuracy:** 86% (Tuned XGBoost)
    
    **Built by:** [Your Name] | [Your LinkedIn URL] | [Your GitHub URL]
    """)

    st.code("""
steam-success-analyzer/
├── data/
│   ├── raw/          
│   └── processed/    
├── notebooks/
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation.ipynb
├── models/           
├── app.py            
└── README.md
    """)