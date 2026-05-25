# PPG Stress / Non-Stress Classification

## Overview
WESAD wrist BVP 기반 stress/non-stress 이진 분류 연구 코드.

## Dataset
WESAD dataset은 repository에 포함하지 않음.
사용자는 WESAD 공식 경로에서 데이터를 다운로드한 뒤 data/raw에 배치해야 함.

## Pipeline
1. Label simplification
2. BVP label synchronization
3. Bandpass / Kalman / Moving average
4. Hanning window
5. Peak detection
6. NN interval extraction
7. Peak correction
8. HRV feature extraction
9. SVM LOSO training

## Final Result
Baseline:
Accuracy 92.83%, F1 87.71%, AUC 98.04%

Final:
0.5~3.5Hz bandpass + short NN interval peak correction
Accuracy 94.08%, F1 88.96%, AUC 99.91%