import sys, os
import numpy as np
import pandas as pd

passed = failed = 0
print('\nTEST 1: Cleaned Data')
if os.path.exists('results/cleaned_data.csv'):
    print('PASS: cleaned_data.csv exists')
    passed += 1
else:
    print('FAIL: cleaned_data.csv missing')
    failed += 1

print('\nTEST 2: ECG Segments')
if os.path.exists('results/ecg_segments.npy'):
    print('PASS: ecg_segments.npy exists')
    passed += 1
else:
    print('FAIL: ecg_segments.npy missing')
    failed += 1

print('\nTEST 3: Visualizations')
plots = ['feature_distributions.png', 'correlation_matrix.png', 'target_distribution.png', 'ecg_sample_segments.png', 'ecg_spectrograms.png']
for p in plots:
    if os.path.exists(f'results/{p}'):
        print(f'PASS: {p}')
        passed += 1
    else:
        print(f'FAIL: {p}')
        failed += 1

print(f'\n=== SUMMARY ===')
print(f'Passed: {passed}')
print(f'Failed: {failed}')
sys.exit(0 if failed == 0 else 1)
