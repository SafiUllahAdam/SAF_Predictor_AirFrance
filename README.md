# SAF_Predictor_AirFrance

> **Air France — SAF Contribution Predictor**  
> This project demonstrates how explainable AI can support **sustainable aviation** initiatives by predicting and optimizing **passenger likelihood to contribute to SAF**.  
> Built with **LightGBM**, **SHAP**, and **Gradio**, it highlights **data-driven sustainability strategy** and **customer insight modeling** in the aviation sector.
>  
> 🟢 Live Demo: [Hugging Face Space](https://huggingface.co/spaces/SafiUllahAdam/SAF_Predictor_AirFrance)  
> 📊 GitHub Repo: [SafiUllahAdam/SAF_Predictor_AirFrance](https://github.com/SafiUllahAdam/SAF_Predictor_AirFrance)


**Air France — Sustainable Aviation Fuel (SAF) Contribution Predictor**

**Author: Muhammad Safi Ullah Adam**

**Platforms:** VS Code | Hugging Face Space

 **Tech Stack:**  | Python · LightGBM · SHAP · Gradio · Pandas · Joblib · Git

🧭 Project Overview

This project simulates a real-world Air France Data Science use-case:
predicting the likelihood of a passenger contributing to Sustainable Aviation Fuel (SAF) and recommending the most suitable SAF offer (percent + price) for them.

The system merges **machine learning, explainable AI (XAI), and a live interactive interface** to demonstrate both the technical and business value of predictive analytics for sustainable aviation.

# 🎯 Objectives

**Goal	Description**

Predict SAF Contribution	Estimate whether a customer will choose a SAF contribution option based on booking & behavior data.
Recommend SAF Offer	Suggest an optimized SAF percentage (10–25 %) and price range (€10–€50) for high-probability customers.
Explain Model Decisions (SHAP)	Provide transparent, feature-level insights on what drives each prediction.

**🧩 Model & Data Flow**

**Data Preparation:** Synthetic passenger data simulating Air France booking & SAF purchase patterns.

**Model Training:** Linear Regression & LightGBM classifier trained on 12 features (loyalty, class, purchase history, price etc.).

**Explainability:** SHAP (TreeExplainer) to identify top drivers of SAF contribution (e.g., past_saf_purchase, loyalty_tier, ticket_price_eur).

**Deployment:** Interactive Gradio web app integrated directly in Hugging Face Space.

_**The notebook pipeline is organized as:**_

notebooks/

    ├── 01_setup_test.ipynb

    ├── 02_saf_customer_data.ipynb

    ├── 03_models_training.ipynb


_**Models are saved in:**_

data/models/

    ├── lightgbm_model.pkl
    ├── logistic_regression_model.pkl

**💻 How to Use the App**

**🚀 1. Launch Locally (in VS Code or Conda)**

pip install -r requirements.txt
python app.py


Gradio will open a local URL: http://127.0.0.1:7860

**☁️ 2. Run on Hugging Face Space**

All dependencies are defined in requirements.txt.
Once deployed, the app runs automatically and is publicly accessible via its Space URL.

**🪄 3. User Interface Layout**

Section	Function
Predict SAF Contribution	Predicts whether the given passenger is likely to contribute.
Recommend SAF Offer	Suggests the SAF % and price, plus a short explanation (top 3 SHAP drivers).

**Example inputs:**

Loyalty Tier	Booking Class	Past SAF	Adds Options	Passengers	Region	Distance (km)	Price (€)
Platinum	Business	1	1	2	Intercontinental	1200	500

**Example output:**

_**Prediction:**_
Higher Probability of Contributing to SAF according to Customer Details

**Recommendation:**
High probability of contribution.
Recommended offer: SAF 25% at €20–€50.
Key drivers: past_saf_purchase, loyalty_tier, booking_class

# 🧠 Explainable AI ( SHAP )

SHAP (TreeExplainer) quantifies how each feature affects model output.
The following summary plot (from training notebook) highlights the global drivers:

**Interpretation:**
past_saf_purchase → Customers who contributed before are most likely to do so again.

loyalty_tier → Gold and Platinum members show stronger environmental engagement.

ticket_price_eur → Higher fare segments show higher propensity for SAF add-ons.

**⚙️ Repository Structure**

_SAF-PROJECT_AIR_FRANCE/_

     ├── app.py                     # Main Gradio application
     ├── requirements.txt           # Dependencies
     ├── notebooks/                 # Jupyter notebooks for model pipeline
     │   ├── data/outputs/saf_shap_summary.png
     │   └── models/*.pkl
     ├── README.md

**🧾 Technical Highlights**

- Area	Implementation
- ML Algorithm	LightGBM Classifier (gradient boosted trees)
- Explainability	SHAP (TreeExplainer) for per-passenger interpretability
- Deployment	Gradio Blocks UI in Hugging Face Space
- Error Handling	Robust try/except blocks to avoid runtime failure
- Feature Alignment	Automatic zero-padding for missing columns (8 → 12)
- Performance	Real-time predictions and recommendations under 1 second


# 🌍 Business Relevance for Air France

**Customer Segmentation & Targeting:** Identify passengers most likely to opt for SAF offers.

**Personalized Incentives:** Dynamic price recommendations based on loyalty, distance, and class.

**Sustainability Visibility:** Integrates predictive AI with transparency through explainable models.

**Scalability:** Can be extended for real Air France datasets to support corporate SAF goals by 2030 to incorporate at least 10% of SAF globally.

**📦 Installation Requirements**

- gradio
- pandas
- numpy
- joblib
- scikit-learn
- lightgbm
- shap

# 📈 Future Extensions

Integrate real Air France loyalty and booking data (securely via API).

Add logistic regression baseline for comparative analysis.

Enhance UI with SHAP force plots per customer.

Integrate into Air France’s digital customer portal for live offer testing.

# 🏁 Summary

This project demonstrates a deployable, explainable, data-driven solution to increase Air France’s SAF adoption through machine learning.
It shows expertise in AI modeling, XAI, and business application design, precisely aligned with Air France’s digital innovation and sustainability objectives.

**Author: Muhammad Safi Ullah Adam**

**Contact: adam.muhammad01210@gmail.com**

 **LinkedIn Profile | https://www.linkedin.com/in/safi-ullah-adam/**
