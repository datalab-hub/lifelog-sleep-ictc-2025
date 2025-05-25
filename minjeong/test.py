import lightgbm as lgb
print(lgb.__version__)

import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_classification

# 테스트 데이터 생성
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# GPU 파라미터로 모델 학습 시도
try:
    model = lgb.LGBMClassifier(device='gpu', gpu_platform_id=0, gpu_device_id=0)
    model.fit(X, y)
    print("✅ GPU 버전 설치 성공! GPU 사용 가능합니다.")
except Exception as e:
    print(f"❌ GPU 사용 불가: {e}")
    print("CPU 버전만 사용 가능합니다.")