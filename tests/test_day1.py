"""
Test script for Day 1 - Environment Setup and Dataset Preparation

This script verifies:
1. All required packages are installed
2. Datasets are downloaded and accessible
3. Notebook structure is correct
"""

import sys
import os
import importlib
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

class TestDay1:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        
    def test_package_imports(self):
        """Test if all required packages can be imported"""
        print("\n" + "="*60)
        print("TEST 1: Package Installation")
        print("="*60)
        
        required_packages = [
            'pandas',
            'numpy',
            'matplotlib',
            'scipy',
            'sklearn',
            'tensorflow',
            'wfdb',
            'librosa'
        ]
        
        for package in required_packages:
            try:
                if package == 'sklearn':
                    # sklearn is imported as scikit-learn
                    mod = importlib.import_module(package)
                else:
                    mod = importlib.import_module(package)
                
                version = getattr(mod, '__version__', 'unknown')
                print(f"{GREEN}✓{RESET} {package:20s} - version {version}")
                self.passed += 1
            except ImportError as e:
                print(f"{RED}✗{RESET} {package:20s} - NOT INSTALLED")
                self.failed += 1
                
    def test_dataset_files(self):
        """Test if dataset files exist"""
        print("\n" + "="*60)
        print("TEST 2: Dataset Files")
        print("="*60)
        
        # Check Cleveland dataset
        cleveland_files = [
            'datasets/cleveland/heart.csv',
            'datasets/cleveland/processed.cleveland.data',
            'datasets/cleveland/README.md'
        ]
        
        print("\nCleveland Dataset:")
        for file_path in cleveland_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"{GREEN}✓{RESET} {file_path:50s} ({size} bytes)")
                self.passed += 1
            else:
                print(f"{RED}✗{RESET} {file_path:50s} - NOT FOUND")
                self.failed += 1
        
        # Check MIT-BIH dataset
        print("\nMIT-BIH Dataset:")
        mitbih_records = ['100', '101', '102', '103', '104']
        mitbih_extensions = ['.dat', '.hea', '.atr']
        
        for record in mitbih_records:
            all_found = True
            for ext in mitbih_extensions:
                file_path = f'datasets/mit-bih/{record}{ext}'
                if not os.path.exists(file_path):
                    all_found = False
                    
            if all_found:
                print(f"{GREEN}✓{RESET} Record {record} - All files present")
                self.passed += 1
            else:
                print(f"{RED}✗{RESET} Record {record} - Missing files")
                self.failed += 1
                
    def test_notebook_structure(self):
        """Test if notebook exists and has proper structure"""
        print("\n" + "="*60)
        print("TEST 3: Notebook Structure")
        print("="*60)
        
        notebook_path = 'heart_disease_detection.ipynb'
        
        if os.path.exists(notebook_path):
            print(f"{GREEN}✓{RESET} Notebook file exists: {notebook_path}")
            self.passed += 1
            
            # Check notebook size
            size = os.path.getsize(notebook_path)
            if size > 100:  # Should have some content
                print(f"{GREEN}✓{RESET} Notebook has content ({size} bytes)")
                self.passed += 1
            else:
                print(f"{YELLOW}⚠{RESET} Notebook seems empty ({size} bytes)")
                self.warnings += 1
                
            # Try to read notebook structure
            try:
                import json
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    nb = json.load(f)
                    
                cell_count = len(nb.get('cells', []))
                print(f"{GREEN}✓{RESET} Notebook has {cell_count} cells")
                self.passed += 1
                
                # Check for key sections
                required_sections = [
                    'Section A',
                    'Section B',
                    'Section C',
                    'Section D',
                    'Section E'
                ]
                
                notebook_text = json.dumps(nb)
                for section in required_sections:
                    if section in notebook_text:
                        print(f"{GREEN}✓{RESET} {section} found")
                        self.passed += 1
                    else:
                        print(f"{YELLOW}⚠{RESET} {section} not found")
                        self.warnings += 1
                        
            except Exception as e:
                print(f"{RED}✗{RESET} Error reading notebook: {e}")
                self.failed += 1
        else:
            print(f"{RED}✗{RESET} Notebook file not found: {notebook_path}")
            self.failed += 1
            
    def test_directory_structure(self):
        """Test if project directory structure is correct"""
        print("\n" + "="*60)
        print("TEST 4: Directory Structure")
        print("="*60)
        
        required_dirs = [
            'datasets',
            'datasets/cleveland',
            'datasets/mit-bih',
            'docs',
            'models',
            'results',
            'scripts',
            'tests'
        ]
        
        for dir_path in required_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                print(f"{GREEN}✓{RESET} {dir_path:30s} exists")
                self.passed += 1
            else:
                print(f"{RED}✗{RESET} {dir_path:30s} NOT FOUND")
                self.failed += 1
                
    def test_requirements_file(self):
        """Test if requirements.txt exists"""
        print("\n" + "="*60)
        print("TEST 5: Requirements File")
        print("="*60)
        
        if os.path.exists('requirements.txt'):
            print(f"{GREEN}✓{RESET} requirements.txt exists")
            self.passed += 1
            
            with open('requirements.txt', 'r') as f:
                requirements = f.read()
                
            required_packages = [
                'tensorflow',
                'scikit-learn',
                'pandas',
                'numpy',
                'matplotlib',
                'scipy',
                'wfdb',
                'librosa'
            ]
            
            for package in required_packages:
                if package in requirements:
                    print(f"{GREEN}✓{RESET} {package} listed in requirements")
                    self.passed += 1
                else:
                    print(f"{YELLOW}⚠{RESET} {package} NOT in requirements.txt")
                    self.warnings += 1
        else:
            print(f"{RED}✗{RESET} requirements.txt not found")
            self.failed += 1
            
    def run_all_tests(self):
        """Run all tests and print summary"""
        print("\n" + "="*60)
        print("DAY 1 TEST SUITE - Environment Setup & Dataset Preparation")
        print("="*60)
        
        self.test_package_imports()
        self.test_dataset_files()
        self.test_notebook_structure()
        self.test_directory_structure()
        self.test_requirements_file()
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"{GREEN}Passed:{RESET}   {self.passed}")
        print(f"{RED}Failed:{RESET}   {self.failed}")
        print(f"{YELLOW}Warnings:{RESET} {self.warnings}")
        print("="*60)
        
        if self.failed == 0:
            print(f"\n{GREEN}✓ ALL TESTS PASSED! Day 1 setup is complete.{RESET}")
            return 0
        else:
            print(f"\n{RED}✗ SOME TESTS FAILED. Please review the errors above.{RESET}")
            return 1

if __name__ == "__main__":
    tester = TestDay1()
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)
