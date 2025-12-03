# =============================================================================
# STREAMLIT APP SETUP GUIDE
# =============================================================================
# Complete instructions for deploying the Accident Severity Predictor
# =============================================================================

"""
STREAMLIT APP: ACCIDENT SEVERITY PREDICTOR
===========================================

This guide will help you set up and run the interactive web application
for predicting accident severity using your trained machine learning models.


=============================================================================
STEP 1: TRAIN AND SAVE MODELS
=============================================================================

1.1 Train Multi-Class Models
-----------------------------
Run your Jupyter notebook with:
    multiclass_ml_corrected.py

This will train 7 models and create these variables:
- multiclass_models
- multiclass_scaler
- multiclass_feature_names


1.2 Save Models for Streamlit
------------------------------
In Jupyter, run:
    save_models_for_streamlit.py

OR manually run this code in Jupyter:

```python
import pickle
import os

# Create models directory
os.makedirs('models', exist_ok=True)

# Save each model
for name, model in multiclass_models.items():
    filename = name.replace('. ', '_').replace(' ', '_').lower()
    with open(f'models/{filename}.pkl', 'wb') as f:
        pickle.dump(model, f)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(multiclass_scaler, f)

# Save feature names
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(multiclass_feature_names, f)

print("✅ Models saved successfully!")
```


=============================================================================
STEP 2: INSTALL STREAMLIT
=============================================================================

2.1 Install via pip
-------------------
Open terminal/command prompt and run:

    pip install streamlit

Or install all dependencies:

    pip install -r requirements.txt


2.2 Verify Installation
------------------------
Check Streamlit version:

    streamlit --version

You should see something like: Streamlit, version 1.29.0


=============================================================================
STEP 3: SET UP PROJECT STRUCTURE
=============================================================================

Your project folder should look like this:

    accident_severity_app/
    ├── streamlit_app.py              ← Main app file
    ├── requirements.txt               ← Dependencies
    ├── models/                        ← Trained models folder
    │   ├── 1_logistic_regression.pkl
    │   ├── 2_decision_tree.pkl
    │   ├── 3_random_forest.pkl
    │   ├── 4_gradient_boosting.pkl
    │   ├── 5_k-nearest_neighbors.pkl
    │   ├── 6_xgboost.pkl
    │   ├── 7_lightgbm.pkl
    │   ├── scaler.pkl
    │   └── feature_names.pkl


=============================================================================
STEP 4: RUN THE STREAMLIT APP
=============================================================================

4.1 Navigate to Project Folder
-------------------------------
Open terminal and navigate to your project folder:

    cd /path/to/accident_severity_app


4.2 Run Streamlit
-----------------
Execute:

    streamlit run streamlit_app.py


4.3 Access the App
------------------
Streamlit will automatically open your browser at:

    http://localhost:8501

If not, manually open the URL shown in terminal.


=============================================================================
STEP 5: USING THE APP
=============================================================================

5.1 App Features
----------------
The app has 4 main tabs:

🌡️ WEATHER TAB:
   - Temperature, Humidity, Pressure
   - Visibility, Wind Speed, Precipitation
   - Weather flags (Rain, Snow, Fog, etc.)

🚦 ROAD FEATURES TAB:
   - Road infrastructure (Crossing, Junction, etc.)
   - Traffic control (Signals, Stop signs)
   - Affected road distance

⏰ TIME TAB:
   - Date and time of accident
   - Automatically calculates:
     * Day of week
     * Weekend flag
     * Morning/Evening peak flags

📍 LOCATION TAB:
   - Placeholder for future features


5.2 Making Predictions
-----------------------
1. Select a model from sidebar
2. Fill in accident details in tabs
3. Click "🔮 Predict Severity" button
4. View results:
   - Predicted severity level (1-4)
   - Confidence distribution chart
   - Individual class probabilities
   - Feature importance (for tree models)


5.3 Understanding Results
--------------------------
Severity Levels:
- Level 1 (Green):  Low Impact - Minor delay
- Level 2 (Yellow): Moderate Impact - Some disruption
- Level 3 (Orange): High Impact - Significant delay
- Level 4 (Red):    Severe Impact - Major disruption


=============================================================================
STEP 6: CUSTOMIZATION OPTIONS
=============================================================================

6.1 Change App Title/Icon
--------------------------
Edit in streamlit_app.py:

    st.set_page_config(
        page_title="Your Custom Title",
        page_icon="🚙",  # Any emoji
        layout="wide"
    )


6.2 Modify Input Ranges
------------------------
Adjust slider ranges in the tab sections:

    inputs['Temperature(F)'] = st.slider(
        "Temperature (°F)",
        min_value=-20.0,  # Change this
        max_value=120.0,  # Change this
        value=70.0        # Default value
    )


6.3 Add Custom Styling
-----------------------
Modify CSS in the st.markdown() section at the top


6.4 Add More Features
----------------------
To add new input features:

1. Add input widget in appropriate tab
2. Store in inputs dictionary
3. Make sure feature name matches training data


=============================================================================
STEP 7: DEPLOYMENT (OPTIONAL)
=============================================================================

7.1 Deploy to Streamlit Cloud (FREE)
-------------------------------------
1. Push your code to GitHub
2. Go to https://share.streamlit.io/
3. Sign in with GitHub
4. Deploy your app
5. Share the public URL


7.2 Deploy to Other Platforms
------------------------------
- Heroku
- AWS EC2
- Google Cloud
- Azure


=============================================================================
TROUBLESHOOTING
=============================================================================

❌ ERROR: "Models not found!"
------------------------------
SOLUTION:
- Make sure you ran save_models_for_streamlit.py
- Check that models/ folder exists in same directory as streamlit_app.py
- Verify .pkl files are in models/ folder


❌ ERROR: "ModuleNotFoundError: No module named 'streamlit'"
------------------------------------------------------------
SOLUTION:
- Install Streamlit: pip install streamlit
- Or: pip install -r requirements.txt


❌ ERROR: "Feature mismatch"
----------------------------
SOLUTION:
- Make sure saved models use same features as app inputs
- Check feature_names.pkl contains all required features


❌ ERROR: App won't start
-------------------------
SOLUTION:
- Check Python version (3.8+ required)
- Update packages: pip install --upgrade streamlit
- Check for syntax errors in streamlit_app.py


❌ Prediction gives weird results
---------------------------------
SOLUTION:
- Verify input values are in reasonable ranges
- Check that scaler was trained on same features
- Try different models to compare results


=============================================================================
TESTING CHECKLIST
=============================================================================

Before deploying, test these scenarios:

✅ Test with typical values
   - Temperature: 70°F
   - Humidity: 50%
   - Clear weather
   - No road features

✅ Test with extreme weather
   - Very low/high temperature
   - Low visibility
   - Heavy precipitation
   - Multiple weather flags

✅ Test rush hour scenarios
   - Morning peak (7-9 AM)
   - Evening peak (5-7 PM)
   - Weekend vs weekday

✅ Test all models
   - Try each model
   - Compare predictions
   - Verify probabilities sum to 100%

✅ Test feature importance
   - Check tree-based models show importance
   - Verify top features make sense


=============================================================================
PERFORMANCE OPTIMIZATION
=============================================================================

For Large Datasets:
-------------------
1. Use @st.cache_resource for loading models
   (Already implemented in the app)

2. Reduce model complexity if needed:
   - Fewer trees in Random Forest
   - Lower max_depth in trees

3. Consider model quantization for deployment


For Faster Loading:
-------------------
1. Use smaller model files
2. Load only selected model (not all at once)
3. Implement lazy loading


=============================================================================
SUPPORT & RESOURCES
=============================================================================

Streamlit Documentation:
https://docs.streamlit.io/

Streamlit Gallery (Examples):
https://streamlit.io/gallery

Streamlit Forum:
https://discuss.streamlit.io/

GitHub Issues:
https://github.com/streamlit/streamlit/issues


=============================================================================
QUICK REFERENCE COMMANDS
=============================================================================

Start app:
    streamlit run streamlit_app.py

Stop app:
    Ctrl + C (in terminal)

Clear cache:
    streamlit cache clear

Install dependencies:
    pip install -r requirements.txt

Update Streamlit:
    pip install --upgrade streamlit

Check version:
    streamlit --version


=============================================================================
END OF GUIDE
=============================================================================

Good luck with your Accident Severity Predictor app! 🚗✨

For questions or issues, refer to the troubleshooting section above.
"""

print(__doc__)