# # 디바이스 확인 코드 



# import lightgbm as lgb
# import numpy as np
# from sklearn.datasets import make_classification

# # 더 구체적인 GPU 설정으로 테스트
# X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
# train_data = lgb.Dataset(X, label=y)

# # NVIDIA GPU를 명시적으로 지정
# params = {
#     'objective': 'binary',
#     'device': 'gpu',
#     'gpu_platform_id': 0,  # 첫 번째 OpenCL 플랫폼
#     'gpu_device_id': 0,    # 첫 번째 GPU 디바이스
#     'verbosity': 2
# }

# print("NVIDIA GPU 사용 시도...")
# try:
#     model = lgb.train(params, train_data, num_boost_round=5)
#     print("✅ NVIDIA GPU 사용 성공!")
# except Exception as e:
#     print(f"❌ NVIDIA GPU 사용 실패: {e}")
    
#     # 다른 플랫폼/디바이스 시도
#     for platform_id in range(3):  # 0, 1, 2 플랫폼 시도
#         for device_id in range(3):  # 0, 1, 2 디바이스 시도
#             try:
#                 params_test = {
#                     'objective': 'binary',
#                     'device': 'gpu',
#                     'gpu_platform_id': platform_id,
#                     'gpu_device_id': device_id,
#                     'verbosity': 1
#                 }
#                 print(f"플랫폼 {platform_id}, 디바이스 {device_id} 시도...")
#                 model = lgb.train(params_test, train_data, num_boost_round=1)
#                 print(f"✅ 플랫폼 {platform_id}, 디바이스 {device_id} 성공!")
#                 break
#             except:
#                 continue
#         else:
#             continue
#         break
import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_classification

print(f"LightGBM 버전: {lgb.__version__}")

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
train_data = lgb.Dataset(X, label=y)

# GPU 테스트
params = {
    'objective': 'binary',
    'device': 'gpu',  # 'gpu' 대신 'cuda'
    'verbosity': 2
}

try:
    print("GPU 사용 시도...")
    model = lgb.train(params, train_data, num_boost_round=5)
    print("✅ GPU 사용 성공!")
except Exception as e:
    print(f"❌ GPU 사용 실패: {e}")