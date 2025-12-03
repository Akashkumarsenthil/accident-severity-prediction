# =============================================================================
# US ACCIDENTS SEVERITY PREDICTION - STREAMLIT APP (FULL VERSION)
# =============================================================================
# Supports both MULTI-CLASS (4 levels) and BINARY (LOW/HIGH) classification
# Run with: streamlit run streamlit_app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, time as dt_time

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Accident Severity Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# TITLE
# =============================================================================
st.markdown('<h2 class="main-header">National Highway Traffic Safety Administration (NHTSA)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict accident severity using Machine Learning models</p>', unsafe_allow_html=True)

# =============================================================================
# LOAD MODELS FUNCTION
# =============================================================================
@st.cache_resource
def load_all_models():
    """Load all trained models for both classification types"""
    
    multiclass_models = {}
    binary_models = {}
    multiclass_scaler = None
    binary_scaler = None
    multiclass_features = None
    binary_features = None
    
    # Check if models directory exists
    if not os.path.exists('models/'):
        return None, None, None, None, None, None
    
    try:
        # Load Multi-class models
        multiclass_dir = 'models/multiclass/'
        if os.path.exists(multiclass_dir):
            for file in os.listdir(multiclass_dir):
                if file.endswith('.pkl') and file not in ['scaler.pkl', 'feature_names.pkl']:
                    model_name = file.replace('.pkl', '').replace('_', ' ').title()
                    with open(f'{multiclass_dir}/{file}', 'rb') as f:
                        multiclass_models[model_name] = pickle.load(f)
            
            # Load multiclass scaler and features
            if os.path.exists(f'{multiclass_dir}/scaler.pkl'):
                with open(f'{multiclass_dir}/scaler.pkl', 'rb') as f:
                    multiclass_scaler = pickle.load(f)
            
            if os.path.exists(f'{multiclass_dir}/feature_names.pkl'):
                with open(f'{multiclass_dir}/feature_names.pkl', 'rb') as f:
                    multiclass_features = pickle.load(f)
        
        # Load Binary models
        binary_dir = 'models/binary/'
        if os.path.exists(binary_dir):
            for file in os.listdir(binary_dir):
                if file.endswith('.pkl') and file not in ['scaler.pkl', 'feature_names.pkl']:
                    model_name = file.replace('.pkl', '').replace('_', ' ').title()
                    with open(f'{binary_dir}/{file}', 'rb') as f:
                        binary_models[model_name] = pickle.load(f)
            
            # Load binary scaler and features
            if os.path.exists(f'{binary_dir}/scaler.pkl'):
                with open(f'{binary_dir}/scaler.pkl', 'rb') as f:
                    binary_scaler = pickle.load(f)
            
            if os.path.exists(f'{binary_dir}/feature_names.pkl'):
                with open(f'{binary_dir}/feature_names.pkl', 'rb') as f:
                    binary_features = pickle.load(f)
        
        # Fallback: Check old structure (models/ directly)
        if not multiclass_models and not binary_models:
            for file in os.listdir('models/'):
                if file.endswith('.pkl') and file not in ['scaler.pkl', 'feature_names.pkl', 
                                                           'multiclass_scaler.pkl', 'binary_scaler.pkl',
                                                           'multiclass_feature_names.pkl', 'binary_feature_names.pkl']:
                    model_name = file.replace('.pkl', '').replace('_', ' ').title()
                    with open(f'models/{file}', 'rb') as f:
                        model = pickle.load(f)
                        # Add to both for compatibility
                        multiclass_models[model_name] = model
                        binary_models[model_name] = model
            
            # Try loading scalers with different naming conventions
            for scaler_name in ['scaler.pkl', 'multiclass_scaler.pkl']:
                if os.path.exists(f'models/{scaler_name}'):
                    with open(f'models/{scaler_name}', 'rb') as f:
                        multiclass_scaler = pickle.load(f)
                    break
            
            for scaler_name in ['scaler.pkl', 'binary_scaler.pkl']:
                if os.path.exists(f'models/{scaler_name}'):
                    with open(f'models/{scaler_name}', 'rb') as f:
                        binary_scaler = pickle.load(f)
                    break
            
            # Load feature names
            for feat_name in ['feature_names.pkl', 'multiclass_feature_names.pkl']:
                if os.path.exists(f'models/{feat_name}'):
                    with open(f'models/{feat_name}', 'rb') as f:
                        multiclass_features = pickle.load(f)
                    break
            
            for feat_name in ['feature_names.pkl', 'binary_feature_names.pkl']:
                if os.path.exists(f'models/{feat_name}'):
                    with open(f'models/{feat_name}', 'rb') as f:
                        binary_features = pickle.load(f)
                    break
        
        return multiclass_models, binary_models, multiclass_scaler, binary_scaler, multiclass_features, binary_features
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None

# Load all models
(multiclass_models, binary_models, 
 multiclass_scaler, binary_scaler, 
 multiclass_features, binary_features) = load_all_models()

# =============================================================================
# CHECK IF MODELS LOADED
# =============================================================================
if (not multiclass_models and not binary_models):
    st.error("⚠️ No models found! Please save models first.")
    
    st.markdown("### 📋 How to Save Models")
    
    with st.expander("Click to see model saving instructions", expanded=True):
        st.code("""
# =============================================================================
# SAVE MODELS FOR STREAMLIT (Run after training)
# =============================================================================
import pickle
import os

# Create directories
os.makedirs('models/multiclass', exist_ok=True)
os.makedirs('models/binary', exist_ok=True)

# ============ SAVE MULTI-CLASS MODELS ============
print("Saving Multi-class models...")
for name, model in multiclass_models.items():
    filename = name.replace('. ', '_').replace(' ', '_').lower()
    with open(f'models/multiclass/{filename}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Saved: {filename}.pkl")

with open('models/multiclass/scaler.pkl', 'wb') as f:
    pickle.dump(multiclass_scaler, f)

with open('models/multiclass/feature_names.pkl', 'wb') as f:
    pickle.dump(multiclass_feature_names, f)

print("✅ Multi-class models saved!")

# ============ SAVE BINARY MODELS ============
print("\\nSaving Binary models...")
for name, model in binary_models.items():
    filename = name.replace('. ', '_').replace(' ', '_').lower()
    with open(f'models/binary/{filename}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Saved: {filename}.pkl")

with open('models/binary/scaler.pkl', 'wb') as f:
    pickle.dump(binary_scaler, f)

with open('models/binary/feature_names.pkl', 'wb') as f:
    pickle.dump(binary_feature_names, f)

print("✅ Binary models saved!")
print("\\n🎉 All models saved! Run: streamlit run streamlit_app.py")
        """, language="python")
    
    st.markdown("""
    ### 📁 Expected File Structure:
    ```
    your_project/
    ├── streamlit_app.py
    └── models/
        ├── multiclass/
        │   ├── 1_logistic_regression.pkl
        │   ├── 2_decision_tree.pkl
        │   ├── 3_random_forest.pkl
        │   ├── ...
        │   ├── scaler.pkl
        │   └── feature_names.pkl
        └── binary/
            ├── 1_logistic_regression.pkl
            ├── 2_decision_tree.pkl
            ├── 3_random_forest.pkl
            ├── ...
            ├── scaler.pkl
            └── feature_names.pkl
    ```
    """)
    st.stop()

# =============================================================================
# SIDEBAR - CLASSIFICATION TYPE & MODEL SELECTION
# =============================================================================
st.sidebar.markdown("## Configuration")

# Classification Type Selection
st.sidebar.markdown("### Classification Type")
classification_type = st.sidebar.radio(
    "Select prediction type:",
    ["🎯 Multi-Class (4 Levels)", "⚖️ Binary (LOW/HIGH)"],
    help="Multi-class predicts severity 1-4, Binary predicts LOW or HIGH"
)

is_multiclass = "Multi-Class" in classification_type

st.sidebar.markdown("---")

# Model Selection based on classification type
st.sidebar.markdown("### Model Selection")

if is_multiclass:
    if multiclass_models:
        selected_model = st.sidebar.selectbox(
            "Choose Multi-Class Model:",
            list(multiclass_models.keys()),
            index=3,  # 👈 This sets the 2nd item as default (0-indexed)
            help="Select a trained model for 4-class severity prediction"
        )
        current_models = multiclass_models
        current_scaler = multiclass_scaler
        current_features = multiclass_features
    else:
        st.sidebar.error("No multi-class models found!")
        st.stop()
else:
    if binary_models:
        selected_model = st.sidebar.selectbox(
            "Choose Binary Model:",
            list(binary_models.keys()),
            index=4,  
            help="Select a trained model for LOW/HIGH prediction"
        )
        current_models = binary_models
        current_scaler = binary_scaler
        current_features = binary_features
    else:
        st.sidebar.error("No binary models found!")
        st.stop()

# Display model info
st.sidebar.markdown("---")
st.sidebar.markdown("### Classification Info")

if is_multiclass:
    st.sidebar.info("""
    **Multi-Class Prediction**
    
    🟢 **Severity 1**: Low Impact  
    🟡 **Severity 2**: Moderate Impact  
    🟠 **Severity 3**: High Impact  
    🔴 **Severity 4**: Severe Impact
    """)
else:
    st.sidebar.info("""
    **Binary Prediction**
    
    🔵 **LOW** (0): Severity 1, 2  
    → Minor accidents, less traffic impact
    
    🔴 **HIGH** (1): Severity 3, 4  
    → Major accidents, significant impact
    """)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Selected Model:** {selected_model}")
st.sidebar.markdown(f"**Models Loaded:** {len(current_models)}")

# =============================================================================
# MAIN CONTENT - INPUT FORM
# =============================================================================
st.markdown("---")
st.markdown("## Enter Accident Details")

# Create three columns for inputs
col1, col2, col3 = st.columns(3)

inputs = {}

# COLUMN 1: Weather Conditions
with col1:
    st.markdown("### 🌤️ Weather Conditions")
    
    inputs['Temperature(F)'] = st.slider(
        "Temperature (°F)", 
        min_value=-20, max_value=120, value=70,
        help="Ambient temperature at accident location"
    )
    
    inputs['Humidity(%)'] = st.slider(
        "Humidity (%)", 
        min_value=0, max_value=100, value=50,
        help="Relative humidity percentage"
    )
    
    inputs['Pressure(in)'] = st.slider(
        "Pressure (in)", 
        min_value=28.0, max_value=31.0, value=29.9, step=0.1,
        help="Atmospheric pressure in inches"
    )
    
    inputs['Visibility(mi)'] = st.slider(
        "Visibility (mi)", 
        min_value=0.0, max_value=10.0, value=10.0, step=0.5,
        help="Visibility distance in miles"
    )
    
    inputs['Wind_Speed(mph)'] = st.slider(
        "Wind Speed (mph)", 
        min_value=0, max_value=60, value=5,
        help="Wind speed at location"
    )
    
    inputs['Precipitation(in)'] = st.slider(
        "Precipitation (in)", 
        min_value=0.0, max_value=5.0, value=0.0, step=0.1,
        help="Precipitation amount"
    )

# COLUMN 2: Road Features
with col2:
    st.markdown("### 🛣️ Road Features")
    
    inputs['Distance(mi)'] = st.slider(
        "Affected Road Distance (mi)", 
        min_value=0.0, max_value=10.0, value=0.5, step=0.1,
        help="Length of road affected by accident"
    )
    
    st.markdown("**Road Elements Present:**")
    
    road_col1, road_col2 = st.columns(2)
    
    with road_col1:
        inputs['Crossing'] = 1 if st.checkbox("🚸 Crossing", help="Pedestrian crossing nearby") else 0
        inputs['Junction'] = 1 if st.checkbox("🔀 Junction", help="Road junction/intersection") else 0
        inputs['Traffic_Signal'] = 1 if st.checkbox("🚦 Traffic Signal", help="Traffic light present") else 0
        inputs['Stop'] = 1 if st.checkbox("🛑 Stop Sign", help="Stop sign present") else 0
    
    with road_col2:
        inputs['Roundabout'] = 1 if st.checkbox("🔄 Roundabout", help="Roundabout nearby") else 0
        inputs['Give_Way'] = 1 if st.checkbox("⚠️ Give Way", help="Give way/yield sign") else 0
        inputs['Bump'] = 1 if st.checkbox("🔶 Speed Bump", help="Speed bump present") else 0
        inputs['Railway'] = 1 if st.checkbox("🚂 Railway", help="Railway crossing nearby") else 0
    
    # Additional road features (hidden but included)
    inputs['Amenity'] = 0
    inputs['No_Exit'] = 0
    inputs['Station'] = 0
    inputs['Traffic_Calming'] = 0

# COLUMN 3: Time & Weather Flags
with col3:
    st.markdown("### ⏰ Time & Conditions")
    
    # Date and Time inputs
    accident_date = st.date_input(
    "Accident Date", 
    datetime(2022, 12, 15),  
    help="Date of the accident"
    )
    
    accident_time = st.time_input(
        "Accident Time", 
        dt_time(7, 0),
        help="Time of the accident"
    )
    
    # Extract time features
    inputs['year'] = accident_date.year
    inputs['month'] = accident_date.month
    inputs['day'] = accident_date.day
    inputs['dow'] = accident_date.weekday()  # 0=Monday, 6=Sunday
    inputs['hour'] = accident_time.hour
    
    # Derived time features
    inputs['is_weekend'] = 1 if accident_date.weekday() >= 5 else 0
    inputs['is_morning_peak'] = 1 if 6 <= accident_time.hour <= 9 else 0
    inputs['is_evening_peak'] = 1 if 16 <= accident_time.hour <= 19 else 0
    
    st.markdown("**Weather Conditions:**")
    
    weather_col1, weather_col2 = st.columns(2)
    
    with weather_col1:
        inputs['is_rain'] = 1 if st.checkbox("🌧️ Rain", help="Rainy conditions") else 0
        inputs['is_fog'] = 1 if st.checkbox("🌫️ Fog", help="Foggy conditions") else 0
        inputs['is_wind'] = 1 if st.checkbox("💨 Windy", help="High wind conditions") else 0
    
    with weather_col2:
        inputs['is_snow'] = 1 if st.checkbox("❄️ Snow", help="Snowy conditions") else 0
        inputs['is_thunder'] = 1 if st.checkbox("⛈️ Thunder", help="Thunderstorm") else 0
        inputs['is_night'] = 1 if st.checkbox("🌙 Night", help="Nighttime (auto-calculated if unchecked)") else 0
    
    # Auto-calculate night if not checked
    if inputs['is_night'] == 0:
        inputs['is_night'] = 1 if (accident_time.hour < 6 or accident_time.hour >= 20) else 0

# Fill any missing features with 0
if current_features:
    for feat in current_features:
        if feat not in inputs:
            inputs[feat] = 0

# =============================================================================
# PREDICTION BUTTON & RESULTS
# =============================================================================
st.markdown("---")

predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:
    predict_button = st.button(
        "🔮 Predict Severity",
        type="primary",
        use_container_width=True
    )

if predict_button:
    try:
        # Create input dataframe
        if current_features:
            input_df = pd.DataFrame([inputs])[current_features]
        else:
            input_df = pd.DataFrame([inputs])
        
        # Get selected model
        model = current_models[selected_model]
        
        # Check if scaling is needed
        needs_scaling = any(x in selected_model.lower() for x in 
                          ['logistic', 'knn', 'neighbors', 'neural', 'naive', 'mlp'])
        
        # Make prediction
        if needs_scaling and current_scaler is not None:
            input_scaled = current_scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
        else:
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
        
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")
        
        # =================================================================
        # MULTI-CLASS RESULTS
        # =================================================================
        if is_multiclass:
            # Convert 0-3 to 1-4
            severity = int(prediction) + 1
            
            # Define colors and labels
            severity_info = {
                1: {"color": "#28a745", "bg": "#d4edda", "label": "Low Impact", "emoji": "🟢"},
                2: {"color": "#ffc107", "bg": "#fff3cd", "label": "Moderate Impact", "emoji": "🟡"},
                3: {"color": "#fd7e14", "bg": "#ffe5d0", "label": "High Impact", "emoji": "🟠"},
                4: {"color": "#dc3545", "bg": "#f8d7da", "label": "Severe Impact", "emoji": "🔴"}
            }
            
            info = severity_info[severity]
            
            # Display main prediction
            st.markdown(f"""
            <div class="prediction-box" style="background-color: {info['bg']}; border: 3px solid {info['color']};">
                <h1 style="color: {info['color']}; margin: 0; font-size: 3rem;">
                    {info['emoji']} Severity Level: {severity}
                </h1>
                <h2 style="color: {info['color']}; margin-top: 10px;">
                    {info['label']}
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Display probabilities
            st.markdown("### 📈 Confidence Scores")
            
            prob_cols = st.columns(4)
            for i, (col, prob) in enumerate(zip(prob_cols, probabilities)):
                sev = i + 1
                s_info = severity_info[sev]
                with col:
                    st.metric(
                        label=f"{s_info['emoji']} Level {sev}",
                        value=f"{prob*100:.1f}%",
                        delta="PREDICTED" if sev == severity else None
                    )
            
            # Probability bar chart
            st.markdown("### 📊 Probability Distribution")
            prob_df = pd.DataFrame({
                'Severity Level': [f"Level {i+1}" for i in range(4)],
                'Probability (%)': probabilities * 100
            })
            st.bar_chart(prob_df.set_index('Severity Level'))
        
        # =================================================================
        # BINARY RESULTS
        # =================================================================
        else:
            # Binary prediction (0=LOW, 1=HIGH)
            is_high = int(prediction) == 1
            probability_high = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            probability_low = 1 - probability_high
            
            if is_high:
                color = "#dc3545"
                bg_color = "#f8d7da"
                label = "HIGH SEVERITY"
                emoji = "🔴"
                description = "Major accident with significant traffic impact"
            else:
                color = "#007bff"
                bg_color = "#cce5ff"
                label = "LOW SEVERITY"
                emoji = "🔵"
                description = "Minor accident with limited traffic impact"
            
            # Display main prediction
            st.markdown(f"""
            <div class="prediction-box" style="background-color: {bg_color}; border: 3px solid {color};">
                <h1 style="color: {color}; margin: 0; font-size: 3rem;">
                    {emoji} {label}
                </h1>
                <h3 style="color: #666; margin-top: 10px;">
                    {description}
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display probabilities
            st.markdown("### 📈 Confidence Scores")
            
            prob_cols = st.columns(2)
            
            with prob_cols[0]:
                st.metric(
                    label="🔵 LOW Severity (1,2)",
                    value=f"{probability_low*100:.1f}%",
                    delta="PREDICTED" if not is_high else None
                )
            
            with prob_cols[1]:
                st.metric(
                    label="🔴 HIGH Severity (3,4)",
                    value=f"{probability_high*100:.1f}%",
                    delta="PREDICTED" if is_high else None
                )
            
            # Probability gauge
            st.markdown("### 📊 Risk Assessment")
            
            # Create a simple progress bar for HIGH severity probability
            st.markdown(f"""
            <div style="background-color: #e9ecef; border-radius: 10px; height: 40px; margin: 20px 0;">
                <div style="background: linear-gradient(90deg, #007bff {probability_low*100}%, #dc3545 {probability_low*100}%); 
                            width: 100%; height: 100%; border-radius: 10px; position: relative;">
                    <span style="position: absolute; left: 25%; top: 50%; transform: translate(-50%, -50%); 
                                 color: white; font-weight: bold;">LOW: {probability_low*100:.1f}%</span>
                    <span style="position: absolute; left: 75%; top: 50%; transform: translate(-50%, -50%); 
                                 color: white; font-weight: bold;">HIGH: {probability_high*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Bar chart
            prob_df = pd.DataFrame({
                'Severity': ['LOW (1,2)', 'HIGH (3,4)'],
                'Probability (%)': [probability_low * 100, probability_high * 100]
            })
            st.bar_chart(prob_df.set_index('Severity'))
        
        # =================================================================
        # INPUT SUMMARY
        # =================================================================
        with st.expander("📋 View Input Summary", expanded=False):
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.markdown("**Weather:**")
                st.write(f"• Temperature: {inputs['Temperature(F)']}°F")
                st.write(f"• Humidity: {inputs['Humidity(%)']}%")
                st.write(f"• Visibility: {inputs['Visibility(mi)']} mi")
                st.write(f"• Wind Speed: {inputs['Wind_Speed(mph)']} mph")
            
            with summary_col2:
                st.markdown("**Road Features:**")
                road_features = ['Crossing', 'Junction', 'Traffic_Signal', 'Stop', 'Roundabout']
                active_features = [f for f in road_features if inputs.get(f, 0) == 1]
                if active_features:
                    for f in active_features:
                        st.write(f"• {f.replace('_', ' ')}: ✅")
                else:
                    st.write("• No special features")
            
            with summary_col3:
                st.markdown("**Time & Conditions:**")
                st.write(f"• Date: {accident_date}")
                st.write(f"• Time: {accident_time}")
                st.write(f"• Weekend: {'Yes' if inputs['is_weekend'] else 'No'}")
                weather_flags = ['is_rain', 'is_fog', 'is_snow', 'is_night']
                active_weather = [f.replace('is_', '').title() for f in weather_flags if inputs.get(f, 0) == 1]
                if active_weather:
                    st.write(f"• Weather: {', '.join(active_weather)}")
    
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        with st.expander("🔍 Debug Information"):
            st.write("**Error details:**", str(e))
            st.write("**Input features:**", list(inputs.keys()))
            if current_features:
                st.write("**Expected features:**", current_features)
                missing = set(current_features) - set(inputs.keys())
                if missing:
                    st.write("**Missing features:**", missing)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🚗 <strong>US Accident Severity Predictor</strong></p>
    <p>Built with Streamlit & Machine Learning</p>
    <p style="font-size: 0.8rem;">
        Multi-class: 4 severity levels (1-4) | Binary: LOW vs HIGH severity
    </p>
</div>
""", unsafe_allow_html=True)
