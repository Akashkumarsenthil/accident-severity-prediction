# =============================================================================
# SAVE MODELS FOR STREAMLIT APP
# =============================================================================
# Run this AFTER training your multi-class models
# This saves models, scaler, and feature names for the Streamlit app
# =============================================================================

import pickle
import os

print("=" * 70)
print("SAVING MODELS FOR STREAMLIT APP")
print("=" * 70)

# Check if required variables exist
required_vars = ['multiclass_models', 'multiclass_scaler', 'multiclass_feature_names']
missing = [var for var in required_vars if var not in globals()]

if missing:
    print("\n❌ ERROR: Required variables not found!")
    print(f"Missing: {', '.join(missing)}")
    print("\n⚠️  Please run multiclass_ml_corrected.py first!")
    exit()

print("\n✅ All required variables found!")

# Create models directory
os.makedirs('models', exist_ok=True)
print("\n📁 Created 'models/' directory")

# =============================================================================
# SAVE EACH MODEL
# =============================================================================
print("\n[1] Saving trained models...")
print("-" * 50)

saved_count = 0
for name, model in multiclass_models.items():
    try:
        # Create filename (remove numbers and spaces)
        filename = name.replace('. ', '_').replace(' ', '_').lower()
        filepath = f'models/{filename}.pkl'
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"  ✓ Saved: {filename}.pkl")
        saved_count += 1
    except Exception as e:
        print(f"  ✗ Error saving {name}: {e}")

print(f"\n✅ Saved {saved_count}/{len(multiclass_models)} models")

# =============================================================================
# SAVE SCALER
# =============================================================================
print("\n[2] Saving scaler...")
print("-" * 50)

try:
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(multiclass_scaler, f)
    print("  ✓ Saved: scaler.pkl")
except Exception as e:
    print(f"  ✗ Error saving scaler: {e}")

# =============================================================================
# SAVE FEATURE NAMES
# =============================================================================
print("\n[3] Saving feature names...")
print("-" * 50)

try:
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(multiclass_feature_names, f)
    print("  ✓ Saved: feature_names.pkl")
    print(f"  Features: {len(multiclass_feature_names)}")
except Exception as e:
    print(f"  ✗ Error saving feature names: {e}")

# =============================================================================
# VERIFICATION
# =============================================================================
print("\n[4] Verifying saved files...")
print("-" * 50)

files_to_check = ['scaler.pkl', 'feature_names.pkl']
for name in multiclass_models.keys():
    filename = name.replace('. ', '_').replace(' ', '_').lower() + '.pkl'
    files_to_check.append(filename)

all_exist = True
for filename in files_to_check:
    filepath = f'models/{filename}'
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / 1024  # KB
        print(f"  ✓ {filename} ({size:.1f} KB)")
    else:
        print(f"  ✗ {filename} - NOT FOUND")
        all_exist = False

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
if all_exist:
    print("✅ ALL MODELS SAVED SUCCESSFULLY!")
    print("=" * 70)
    print("\n📦 Files saved in 'models/' directory:")
    print(f"   - {len(multiclass_models)} model files (.pkl)")
    print(f"   - 1 scaler file (scaler.pkl)")
    print(f"   - 1 feature names file (feature_names.pkl)")
    
    print("\n🚀 Next Steps:")
    print("-" * 50)
    print("1. Make sure streamlit_app.py is in the same directory as models/")
    print("2. Install Streamlit: pip install streamlit")
    print("3. Run the app: streamlit run streamlit_app.py")
    print("4. Open browser at: http://localhost:8501")
    
    print("\n📝 Files structure should look like:")
    print("""
    your_project/
    ├── streamlit_app.py
    ├── models/
    │   ├── 1_logistic_regression.pkl
    │   ├── 2_decision_tree.pkl
    │   ├── 3_random_forest.pkl
    │   ├── 4_gradient_boosting.pkl
    │   ├── 5_k-nearest_neighbors.pkl
    │   ├── 6_xgboost.pkl (or 6_gradient_boosting_v2.pkl)
    │   ├── 7_lightgbm.pkl (or 7_neural_network.pkl)
    │   ├── scaler.pkl
    │   └── feature_names.pkl
    """)
else:
    print("⚠️  SOME FILES FAILED TO SAVE")
    print("=" * 70)
    print("Please check the errors above and try again.")

print("=" * 70)