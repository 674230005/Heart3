# HeartCare AI - KNN

เว็บแอป Streamlit สำหรับจำแนกข้อมูล Heart Dataset ด้วย K-Nearest Neighbors (KNN)

## Features
- Upload CSV dataset
- Dashboard summary and charts
- KNN model evaluation (Accuracy, Confusion Matrix, Classification Report)
- Compare accuracy across K values
- Form for individual prediction
- Dataset explorer and CSV download

## Dataset requirements
The CSV must include the following numeric columns:
`Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease`

Target `HeartDisease` should be encoded as 0 or 1. Categorical feature values should be numeric codes, as in the supplied dataset.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py` and `requirements.txt`.
2. Create a new app at Streamlit Community Cloud and select this repository.
3. Set the main file path to `app.py`.
4. Deploy, then upload the dataset CSV from the app's sidebar.


> Educational demonstration only. This app is not a medical diagnostic tool.


### Profile header
The app includes a compact profile header for Pawaret Phakdeecharoen (Information Technology, student ID 674230005) with the supplied profile photo.
