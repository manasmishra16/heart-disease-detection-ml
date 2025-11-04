"""
Master Test Runner - Execute All Days' Tests
Run this to verify all project deliverables
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def run_test(test_file, day_name):
    """Run a single test file"""
    print_header(f"Running {day_name} Tests")
    print(f"Test File: {test_file}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, test_file, "-v"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace problematic characters
            cwd=Path(__file__).parent
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        return success, result.stdout
    
    except Exception as e:
        print(f"❌ Error running {day_name}: {e}")
        return False, str(e)

def main():
    """Main test runner"""
    print_header("Heart Disease Detection - Complete Test Suite")
    print(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {Path.cwd()}")
    
    # Define all tests
    tests = [
        ("tests/test_day1.py", "Day 1 - Setup & Data Loading"),
        ("tests/test_day2.py", "Day 2 - EDA & Preprocessing"),
        ("tests/test_day3.py", "Day 3 - Baseline Models"),
        ("tests/test_day4.py", "Day 4 - Deep Learning Models"),
        ("tests/test_day5.py", "Day 5 - API & Demo")
    ]
    
    results = {}
    
    # Run each test
    for test_file, day_name in tests:
        test_path = Path(test_file)
        
        if not test_path.exists():
            print(f"⚠️  Test file not found: {test_file}")
            results[day_name] = (False, "File not found")
            continue
        
        success, output = run_test(test_file, day_name)
        results[day_name] = (success, output)
    
    # Summary
    print_header("Test Summary")
    
    passed_count = sum(1 for success, _ in results.values() if success)
    total_count = len(results)
    
    for day_name, (success, output) in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {day_name}")
    
    print("\n" + "-"*80)
    print(f"Overall: {passed_count}/{total_count} test suites passed")
    print(f"Success Rate: {(passed_count/total_count)*100:.1f}%")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Project is complete and verified.")
    else:
        print(f"\n⚠️  {total_count - passed_count} test suite(s) failed. Review output above.")
    
    print("="*80 + "\n")
    
    return passed_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
