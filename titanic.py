import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Set page title
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Prediction App")

# 1. Load Data & Train Model (Cached for performance)
@st.cache_resource
def load_and_train_model():
    df = sns.load_dataset('titanic')
    
    # Preprocessing
    df['age'] = df.groupby(['sex', 'pclass'])['age'].transform(lambda x: x.fillna(x.median()))
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    
    features = ['pclass', 'sex', 'age', 'fare', 'embarked', 'family_size', 'is_alone']
    X = df[features]
    y = df['survived']
    
    X = pd.get_dummies(X, columns=['sex', 'embarked'], drop_first=True)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    return model, X.columns

model, feature_columns = load_and_train_model()

# 2. User Input Interface
st.sidebar.header("Passenger Features")

pclass = st.sidebar.selectbox("Ticket Class (Pclass)", [1, 2, 3], format_func=lambda x: f"{x}st Class" if x==1 else f"{x}nd Class" if x==2 else "3rd Class")
sex = st.sidebar.selectbox("Sex", ["female", "male"])
age = st.sidebar.slider("Age", 0, 80, 28)
fare = st.sidebar.slider("Ticket Fare ($)", 0.0, 500.0, 32.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
sibsp = st.sidebar.number_input("Siblings / Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Parents / Children Aboard", 0, 10, 0)

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
emb_code = embarked[embarked.find("(")+1 : embarked.find(")")]

# 3. Format Input for Prediction
input_data = pd.DataFrame([{
    'pclass': pclass,
    'age': age,
    'fare': fare,
    'family_size': family_size,
    'is_alone': is_alone,
    'sex_male': 1 if sex == 'male' else 0,
    'embarked_Q': 1 if emb_code == 'Q' else 0,
    'embarked_S': 1 if emb_code == 'S' else 0
}])

# Align columns with training data
input_data = input_data.reindex(columns=feature_columns, fill_value=0)

# 4. Predict & Display Results
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.divider()
    if prediction == 1:
        st.success(f"🎉 **Survived!** (Probability: {probability*100:.1f}%)")
    else:
        st.error(f"☠️ **Did Not Survive** (Survival Probability: {probability*100:.1f}%)")