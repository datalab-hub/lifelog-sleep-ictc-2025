## 🗓️6. 25/06/07
### 🗐 dacon_etri_base_mod5.ipynb  
* from dacon_etri_base_mod4_1.ipynb 
* Changed: model    
>  (*) 
* Model: (LGBM, L1, L2) -> MTL(MultiTaskLearning)

## 🗓️5. 25/06/07
### 🗐 dacon_etri_base_mod4_1_1.ipynb  
* from dacon_etri_base_mod4_1.ipynb 
* Changed: 파생변수 추가


## 🗓️4. 25/06/01
### 🗐 dacon_etri_base_mod3_2_1.ipynb  
* Preprocessing from dacon_etri_base_mod3_1.ipynb 
* Changed: Preprocess 
> mAcitivity - MinMaxScaler 
* Model: (LGBM, L1, L2) -> (+ Preprocess before getting Feature Importance variable) (⏬)


## 🗓️3. 25/05/24
### 🗐 dacon_etri_base_mod3_2.ipynb  
* Preprocessing from dacon_etri_base_mod3_1.ipynb 
* Changed: Preprocess 
> - mAcitivity - MinMaxScaler 
> - mAmbient - MaxAbsScaler
* Model: (LGBM, L1, L2) -> (+ Preprocess before getting Feature Importance variable) (⏬) 
> (GridSearch -> Optuna) 

## 🗓️2. 25/05/23
### 🗐 dacon_etri_gpu_mod3_1.ipynb 
* not yet (setting..)
* GPU version of dacon_etri_base_mod3_1.ipynb

----------
## 🗓️1. 25/05/17
### 🗐 dacon_etri_base_mod2_1.ipynb  
* Modified dacon_etri_base_mod2.ipynb 
* Changed: Model
> RandomforestClassifier -> Tabnet Deep learning (⏬)

### 🗐 dacon_etri_base_mod1_1.ipynb
* Improved upon dacon_etri_base_mod1.ipynb
* Changed: Model
> RandomforestClassifier -> LGBM (⏫)