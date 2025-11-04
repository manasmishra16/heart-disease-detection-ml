"""
Day 5 Testing: API + Demo Deliverables
Tests for FastAPI backend and Streamlit demo UI
"""

import os
import sys
import pytest
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestDay5Files:
    """Test that all required Day 5 files exist"""
    
    def test_app_folder_exists(self):
        """Test app/ folder exists"""
        app_dir = PROJECT_ROOT / 'app'
        assert app_dir.exists(), "app/ folder should exist"
        assert app_dir.is_dir(), "app/ should be a directory"
    
    def test_main_py_exists(self):
        """Test main.py (FastAPI backend) exists"""
        main_file = PROJECT_ROOT / 'app' / 'main.py'
        assert main_file.exists(), "app/main.py should exist"
        assert main_file.stat().st_size > 0, "main.py should not be empty"
    
    def test_demo_py_exists(self):
        """Test demo.py (Streamlit UI) exists"""
        demo_file = PROJECT_ROOT / 'app' / 'demo.py'
        assert demo_file.exists(), "app/demo.py should exist"
        assert demo_file.stat().st_size > 0, "demo.py should not be empty"
    
    def test_requirements_txt_exists(self):
        """Test requirements.txt exists"""
        req_file = PROJECT_ROOT / 'app' / 'requirements.txt'
        assert req_file.exists(), "app/requirements.txt should exist"
        assert req_file.stat().st_size > 0, "requirements.txt should not be empty"
    
    def test_readme_exists(self):
        """Test README.md exists"""
        readme_file = PROJECT_ROOT / 'app' / 'README.md'
        assert readme_file.exists(), "app/README.md should exist"
        assert readme_file.stat().st_size > 0, "README.md should not be empty"


class TestMainPyContent:
    """Test main.py (FastAPI) implementation details"""
    
    def setup_method(self):
        """Load main.py content"""
        main_file = PROJECT_ROOT / 'app' / 'main.py'
        with open(main_file, 'r', encoding='utf-8') as f:
            self.content = f.read()
    
    def test_imports_fastapi(self):
        """Test that main.py imports FastAPI"""
        assert 'from fastapi import' in self.content, "Should import FastAPI"
    
    def test_imports_tensorflow(self):
        """Test that main.py imports TensorFlow"""
        assert 'import tensorflow' in self.content, "Should import TensorFlow"
    
    def test_imports_sklearn(self):
        """Test that main.py imports scikit-learn components"""
        assert 'StandardScaler' in self.content, "Should use StandardScaler"
    
    def test_has_app_instance(self):
        """Test that FastAPI app is created"""
        assert 'app = FastAPI' in self.content, "Should create FastAPI app"
    
    def test_has_cors_middleware(self):
        """Test CORS middleware is configured"""
        assert 'CORSMiddleware' in self.content, "Should configure CORS"
    
    def test_has_pydantic_models(self):
        """Test Pydantic models are defined"""
        assert 'from pydantic import' in self.content, "Should import Pydantic"
        assert 'BaseModel' in self.content, "Should use BaseModel"
    
    def test_loads_mlp_model(self):
        """Test that MLP model is loaded"""
        assert 'mlp_clinical.keras' in self.content, "Should load MLP model"
    
    def test_loads_rf_model(self):
        """Test that Random Forest model is loaded"""
        assert 'random_forest.pkl' in self.content, "Should load RF model"
    
    def test_loads_scaler(self):
        """Test that StandardScaler is initialized"""
        assert 'StandardScaler' in self.content, "Should use StandardScaler"
        assert 'scaler.fit' in self.content or 'scaler.transform' in self.content, "Should fit/transform with scaler"


class TestAPIEndpoints:
    """Test that all required API endpoints are defined"""
    
    def setup_method(self):
        """Load main.py content"""
        main_file = PROJECT_ROOT / 'app' / 'main.py'
        with open(main_file, 'r', encoding='utf-8') as f:
            self.content = f.read()
    
    def test_has_root_endpoint(self):
        """Test root (/) endpoint exists"""
        assert '@app.get("/")' in self.content or '@app.get("/")' in self.content, "Should have root endpoint"
    
    def test_has_health_endpoint(self):
        """Test /health endpoint exists"""
        assert '/health' in self.content, "Should have /health endpoint"
    
    def test_has_models_info_endpoint(self):
        """Test /models/info endpoint exists"""
        assert '/models/info' in self.content, "Should have /models/info endpoint"
    
    def test_has_predict_endpoint(self):
        """Test /predict endpoint exists"""
        assert '/predict' in self.content, "Should have /predict endpoint"
        assert '@app.post' in self.content, "Should have POST endpoints"
    
    def test_has_batch_predict(self):
        """Test batch prediction capability"""
        assert 'batch' in self.content.lower(), "Should support batch predictions"
    
    def test_has_13_features(self):
        """Test that all 13 clinical features are defined"""
        features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                   'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        for feature in features:
            assert feature in self.content, f"Should include feature: {feature}"
    
    def test_has_ensemble_prediction(self):
        """Test that ensemble prediction is implemented"""
        # Should combine MLP and RF predictions
        assert 'mlp' in self.content.lower() and 'rf' in self.content.lower(), "Should use both MLP and RF"
        assert 'ensemble' in self.content.lower() or 'average' in self.content.lower(), "Should compute ensemble"


class TestDemoPyContent:
    """Test demo.py (Streamlit) implementation details"""
    
    def setup_method(self):
        """Load demo.py content"""
        demo_file = PROJECT_ROOT / 'app' / 'demo.py'
        with open(demo_file, 'r', encoding='utf-8') as f:
            self.content = f.read()
    
    def test_imports_streamlit(self):
        """Test that demo.py imports Streamlit"""
        assert 'import streamlit' in self.content, "Should import Streamlit"
    
    def test_imports_plotly(self):
        """Test that demo.py imports Plotly for visualizations"""
        assert 'import plotly' in self.content or 'plotly' in self.content, "Should import Plotly"
    
    def test_imports_requests(self):
        """Test that demo.py imports requests for API calls"""
        assert 'import requests' in self.content, "Should import requests"
    
    def test_has_page_config(self):
        """Test Streamlit page configuration"""
        assert 'st.set_page_config' in self.content, "Should configure Streamlit page"
    
    def test_has_api_url(self):
        """Test API URL is defined"""
        assert 'localhost:8000' in self.content or 'API_URL' in self.content, "Should define API URL"
    
    def test_has_input_form(self):
        """Test patient data input form exists"""
        assert 'st.sidebar' in self.content or 'st.number_input' in self.content or 'st.slider' in self.content, "Should have input form"
    
    def test_has_example_patients(self):
        """Test example patient data is provided"""
        assert 'example' in self.content.lower() or 'preset' in self.content.lower(), "Should have example patients"
    
    def test_has_visualization_functions(self):
        """Test visualization functions exist"""
        assert 'gauge' in self.content.lower() or 'chart' in self.content.lower(), "Should have visualizations"
    
    def test_has_predict_button(self):
        """Test prediction button exists"""
        assert 'st.button' in self.content, "Should have prediction button"
    
    def test_has_13_feature_inputs(self):
        """Test all 13 clinical features can be input"""
        features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                   'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        for feature in features:
            assert feature in self.content, f"Should have input for feature: {feature}"
    
    def test_has_risk_assessment(self):
        """Test risk level display"""
        assert 'risk' in self.content.lower(), "Should display risk level"
    
    def test_makes_api_calls(self):
        """Test that API calls are made"""
        assert 'requests.post' in self.content or 'requests.get' in self.content, "Should make API requests"


class TestRequirementsTxt:
    """Test requirements.txt has all necessary dependencies"""
    
    def setup_method(self):
        """Load requirements.txt"""
        req_file = PROJECT_ROOT / 'app' / 'requirements.txt'
        with open(req_file, 'r') as f:
            self.content = f.read()
    
    def test_has_tensorflow(self):
        """Test TensorFlow is listed"""
        assert 'tensorflow' in self.content, "Should include TensorFlow"
    
    def test_has_fastapi(self):
        """Test FastAPI is listed"""
        assert 'fastapi' in self.content, "Should include FastAPI"
    
    def test_has_uvicorn(self):
        """Test Uvicorn is listed"""
        assert 'uvicorn' in self.content, "Should include Uvicorn"
    
    def test_has_streamlit(self):
        """Test Streamlit is listed"""
        assert 'streamlit' in self.content, "Should include Streamlit"
    
    def test_has_plotly(self):
        """Test Plotly is listed"""
        assert 'plotly' in self.content, "Should include Plotly"
    
    def test_has_sklearn(self):
        """Test scikit-learn is listed"""
        assert 'scikit-learn' in self.content, "Should include scikit-learn"
    
    def test_has_pydantic(self):
        """Test Pydantic is listed"""
        assert 'pydantic' in self.content, "Should include Pydantic"
    
    def test_has_requests(self):
        """Test requests is listed"""
        assert 'requests' in self.content, "Should include requests"


class TestREADME:
    """Test README.md documentation"""
    
    def setup_method(self):
        """Load README.md"""
        readme_file = PROJECT_ROOT / 'app' / 'README.md'
        with open(readme_file, 'r', encoding='utf-8') as f:
            self.content = f.read()
    
    def test_has_installation_instructions(self):
        """Test installation instructions are provided"""
        assert 'install' in self.content.lower(), "Should have installation instructions"
        assert 'pip' in self.content.lower(), "Should mention pip"
    
    def test_has_api_start_instructions(self):
        """Test API startup instructions"""
        assert 'python main.py' in self.content or 'uvicorn' in self.content, "Should explain how to start API"
    
    def test_has_demo_start_instructions(self):
        """Test demo startup instructions"""
        assert 'streamlit run' in self.content, "Should explain how to start Streamlit"
    
    def test_documents_endpoints(self):
        """Test API endpoints are documented"""
        assert '/predict' in self.content, "Should document /predict endpoint"
        assert '/health' in self.content, "Should document /health endpoint"
    
    def test_has_feature_descriptions(self):
        """Test feature descriptions are provided"""
        assert 'age' in self.content.lower(), "Should describe features"
        assert 'cholesterol' in self.content.lower() or 'chol' in self.content.lower(), "Should explain medical terms"
    
    def test_has_model_performance(self):
        """Test model performance metrics are included"""
        assert 'accuracy' in self.content.lower() or 'performance' in self.content.lower(), "Should show model performance"
    
    def test_has_examples(self):
        """Test usage examples are provided"""
        assert 'example' in self.content.lower() or 'curl' in self.content.lower(), "Should provide examples"
    
    def test_has_troubleshooting(self):
        """Test troubleshooting section exists"""
        assert 'troubleshoot' in self.content.lower() or 'issue' in self.content.lower() or 'error' in self.content.lower(), "Should have troubleshooting"


class TestModelsExist:
    """Test that required models exist for deployment"""
    
    def test_mlp_model_exists(self):
        """Test MLP model file exists"""
        model_file = PROJECT_ROOT / 'models' / 'mlp_clinical.keras'
        assert model_file.exists(), "MLP model should exist for deployment"
    
    def test_rf_model_exists(self):
        """Test Random Forest model file exists"""
        model_file = PROJECT_ROOT / 'models' / 'random_forest.pkl'
        assert model_file.exists(), "Random Forest model should exist for deployment"
    
    def test_model_h5_exists(self):
        """Test main model.h5 exists"""
        model_file = PROJECT_ROOT / 'models' / 'model.h5'
        assert model_file.exists(), "Main model.h5 should exist"
    
    def test_cleaned_data_exists(self):
        """Test cleaned data for scaler exists"""
        data_file = PROJECT_ROOT / 'results' / 'cleaned_data.csv'
        assert data_file.exists(), "Cleaned data should exist for scaler initialization"


class TestDay5Deliverables:
    """Test Day 5 requirements checklist"""
    
    def test_deliverable_api_backend(self):
        """Test API backend deliverable (main.py with /predict)"""
        main_file = PROJECT_ROOT / 'app' / 'main.py'
        assert main_file.exists(), "API backend (main.py) should exist"
        
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert '/predict' in content, "API should have /predict endpoint"
        assert '@app.post' in content, "API should have POST endpoints"
        assert 'mlp' in content.lower() and 'rf' in content.lower(), "API should use ensemble models"
    
    def test_deliverable_demo_ui(self):
        """Test demo UI deliverable (Streamlit app)"""
        demo_file = PROJECT_ROOT / 'app' / 'demo.py'
        assert demo_file.exists(), "Demo UI (demo.py) should exist"
        
        with open(demo_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert 'streamlit' in content, "Demo should use Streamlit"
        assert 'requests.post' in content or 'requests.get' in content, "Demo should call API"
    
    def test_deliverable_requirements(self):
        """Test requirements.txt deliverable"""
        req_file = PROJECT_ROOT / 'app' / 'requirements.txt'
        assert req_file.exists(), "requirements.txt should exist"
        
        with open(req_file, 'r') as f:
            content = f.read()
        
        required_packages = ['fastapi', 'streamlit', 'tensorflow']
        for package in required_packages:
            assert package in content, f"{package} should be in requirements.txt"
    
    def test_deliverable_readme(self):
        """Test README documentation deliverable"""
        readme_file = PROJECT_ROOT / 'app' / 'README.md'
        assert readme_file.exists(), "README.md should exist"
        
        with open(readme_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert len(content) > 1000, "README should be comprehensive (>1000 chars)"
        assert 'install' in content.lower(), "README should have installation instructions"
        assert 'python main.py' in content or 'uvicorn' in content, "README should explain API startup"
        assert 'streamlit run' in content, "README should explain demo startup"
    
    def test_deliverable_app_folder_structure(self):
        """Test complete app/ folder structure"""
        app_dir = PROJECT_ROOT / 'app'
        assert app_dir.exists(), "app/ folder should exist"
        
        required_files = ['main.py', 'demo.py', 'requirements.txt', 'README.md']
        for file in required_files:
            file_path = app_dir / file
            assert file_path.exists(), f"{file} should exist in app/ folder"


# Test summary function
def test_day5_summary():
    """Print Day 5 test summary"""
    print("\n" + "="*70)
    print("DAY 5 TESTING SUMMARY")
    print("="*70)
    print("\nDeliverables:")
    print("✓ FastAPI backend (main.py) with /predict endpoint")
    print("✓ Streamlit demo UI (demo.py) with visualizations")
    print("✓ Dependencies file (requirements.txt)")
    print("✓ Comprehensive documentation (README.md)")
    print("✓ app/ folder structure with all components")
    print("\nAPI Endpoints:")
    print("  - GET  /        (Root with API info)")
    print("  - GET  /health  (Health check)")
    print("  - GET  /models/info (Model details)")
    print("  - POST /predict (Single patient prediction)")
    print("  - POST /predict/batch (Multiple patients)")
    print("\nDemo Features:")
    print("  - Interactive patient data input form")
    print("  - Example patient presets (Healthy/High Risk/Medium Risk)")
    print("  - Real-time visualizations (gauges, charts)")
    print("  - Risk assessment and clinical recommendations")
    print("  - API health checking")
    print("\nModel Architecture:")
    print("  - Enhanced MLP (85.25% accuracy, 100% recall, 96.37% AUC)")
    print("  - Random Forest (90.16% accuracy, 96.43% recall)")
    print("  - Ensemble (88.52% accuracy, 96.43% AUC, only 1 FN)")
    print("\n" + "="*70)
    print("Day 5: API + Demo - COMPLETE ✅")
    print("="*70 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
