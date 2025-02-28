import streamlit as st
import pandas as pd
import time
from metam import METAM
import matplotlib.pyplot as plt
# Custom CSS for nicer styling
st.markdown("""
    <style>
    .main-title {
        font-size: 3em;
        text-align: center;
        color: #2C3E50;
    }
    .subheader {
        font-size: 1.8em;
        color: #34495E;
        margin-bottom: 10px;
    }
    .metric {
        font-size: 2.5em;
        color: #27AE60;
        font-weight: bold;
    }
    .experiment-box {
        background-color: #ECF0F1;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .analysis {
        background-color: #FDFEFE;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #D5D8DC;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------- Streamlit UI ----------------------
def main():
    st.markdown("<h1 class='main-title'>METAM: Goal-Oriented Data Augmentation</h1>", unsafe_allow_html=True)
    st.sidebar.header("Select Experiment")
    experiment = st.sidebar.selectbox("Choose Experiment",
                                      ("Expensive Housing Prediction",
                                       "High Cat Ratio Prediction",
                                       "Boston Housing Experiment"))

    start_time = time.time()

    if experiment == "Expensive Housing Prediction":
        st.markdown("<div class='experiment-box'><h2>Experiment 1: Predicting Expensive Housing</h2></div>",
                    unsafe_allow_html=True)
        # Load base dataset: Seattle housing prices.
        base_df = pd.read_csv("data/seattle_housing_prices.csv")
        base_df.rename(columns={"zip_code": "zipcode"}, inplace=True)
        median_price = base_df["price"].median()
        base_df["expensive"] = (base_df["price"] >= median_price).astype(int)
        # Candidate augmentations: Crime Rate, Pet Licenses, Incomes.
        candidate_crime = pd.read_csv("data/seattle_prop_crime_rate.csv")
        candidate_crime_tuple = (candidate_crime.copy(), "zipcode", "Crime Rate Data")
        candidate_pet = pd.read_csv("data/seattle_pet_licenses.csv")
        candidate_pet.rename(columns={"zip_code": "zipcode"}, inplace=True)
        candidate_pet_tuple = (candidate_pet.copy(), "zipcode", "Pet Licenses Data")
        candidate_incomes = pd.read_csv("data/seattle_wa_incomes_zip_code.csv")
        candidate_incomes.rename(columns={"Zip Code": "zipcode"}, inplace=True)
        candidate_incomes_tuple = (candidate_incomes.copy(), "zipcode", "Incomes Data")
        candidates = [candidate_crime_tuple, candidate_pet_tuple, candidate_incomes_tuple]
        theta = st.sidebar.slider("Utility Threshold (theta)", 0.9, 1.0, 0.98)

        metam = METAM(
            base_data=base_df,
            base_target_col="expensive",
            candidate_datasets=candidates,
            join_on="zipcode",
            theta=theta
        )
        final_data, final_util, chosen_augs = metam.run_metam()

    elif experiment == "High Cat Ratio Prediction":
        st.markdown("<div class='experiment-box'><h2>Experiment 2: Predicting High Cat Ratio per Zipcode</h2></div>",
                    unsafe_allow_html=True)
        # Load pet licenses dataset.
        pets_df = pd.read_csv("data/seattle_pet_licenses.csv")
        pets_df.rename(columns={"zip_code": "zipcode"}, inplace=True)
        # Aggregate by zipcode.
        agg_df = pets_df.groupby("zipcode").agg(
            total_pets=pd.NamedAgg(column="species", aggfunc="count"),
            cat_count=pd.NamedAgg(column="species", aggfunc=lambda x: (x.str.contains("Cat", case=False)).sum())
        ).reset_index()
        agg_df["cat_ratio"] = agg_df["cat_count"] / agg_df["total_pets"]
        agg_df["cat_present"] = (agg_df["cat_ratio"] > 0.3).astype(int)
        base_df = agg_df[["zipcode", "total_pets", "cat_present"]].copy()
        candidate_incomes = pd.read_csv("data/seattle_wa_incomes_zip_code.csv")
        candidate_incomes.rename(columns={"Zip Code": "zipcode"}, inplace=True)
        candidate_incomes_tuple = (candidate_incomes.copy(), "zipcode", "Incomes Data")
        candidates = [candidate_incomes_tuple]
        theta = st.sidebar.slider("Utility Threshold (theta)", 0.6, 0.8, 0.7)

        metam = METAM(
            base_data=base_df,
            base_target_col="cat_present",
            candidate_datasets=candidates,
            join_on="zipcode",
            theta=theta
        )
        final_data, final_util, chosen_augs = metam.run_metam()

    else:  # Boston Housing Experiment
        st.markdown("<div class='experiment-box'><h2>Experiment 3: Boston Housing Experiment</h2></div>",
                    unsafe_allow_html=True)
        # Load the Boston Housing dataset.
        df = pd.read_csv("data/boston_housing.csv")
        df["Id"] = df.index.astype(str)
        median_medv = df["MEDV"].median()
        df["expensive"] = (df["MEDV"] >= median_medv).astype(int)
        # Base dataset: select a few key predictors.
        base_columns = ["Id", "RM", "LSTAT", "expensive"]
        base_df = df[base_columns].copy()
        # Candidate 1: Crime/Industrial related features.
        cand1_columns = ["Id", "CRIM", "INDUS", "NOX", "AGE", "DIS"]
        cand1_df = df[cand1_columns].copy()
        candidate1 = (cand1_df, "Id", "Crime/Industrial Features")
        # Candidate 2: Zoning/Taxation features.
        cand2_columns = ["Id", "ZN", "RAD", "TAX", "PTRATIO"]
        cand2_df = df[cand2_columns].copy()
        candidate2 = (cand2_df, "Id", "Zoning/Tax Features")
        # Candidate 3: Structural/Demographic features.
        cand3_columns = ["Id", "CHAS", "B"]
        cand3_df = df[cand3_columns].copy()
        candidate3 = (cand3_df, "Id", "Structural/Demographic Features")
        candidates = [candidate1, candidate2, candidate3]
        theta = st.sidebar.slider("Utility Threshold (theta)", 0.8, 1.0, 0.85)

        metam = METAM(
            base_data=base_df,
            base_target_col="expensive",
            candidate_datasets=candidates,
            join_on="Id",
            theta=theta
        )
        final_data, final_util, chosen_augs = metam.run_metam()

    running_time = time.time() - start_time

    st.markdown("<h3>Results</h3>", unsafe_allow_html=True)
    st.markdown(f"<p class='metric'>Final Utility: {final_util:.4f}</p>", unsafe_allow_html=True)
    st.write("**Running Time (seconds):**", f"{running_time:.2f}")
    st.write("**Selected Augmentations:**", [cand["name"] for cand in chosen_augs])

    st.markdown("<h3>Merged Data Summary</h3>", unsafe_allow_html=True)
    st.write(final_data.describe(include='all'))

    st.markdown("<h3>Analysis Summary</h3>", unsafe_allow_html=True)
    if experiment == "Expensive Housing Prediction":
        st.markdown("""
        **Experiment 1: Predicting Expensive Housing**  
        - **Setup:** Base dataset from Seattle Housing Prices with target 'expensive'.  
          Candidate augmentations include Crime Rate Data, Pet Licenses Data, and Incomes Data.  
        - **Observation:** The base model achieved near-perfect performance (≈98% accuracy), so candidate augmentations did not yield significant improvement.  
        - **Conclusion:** METAM avoided unnecessary augmentation because the base data was already highly predictive.
        """)
    elif experiment == "High Cat Ratio Prediction":
        st.markdown("""
        **Experiment 2: Predicting High Cat Ratio per Zipcode**  
        - **Setup:** Base dataset aggregated from Seattle Pet Licenses (using 'zipcode' and weak predictor 'total_pets') with target 'cat_present'.  
        - **Observation:** The base model produced modest performance (≈62% accuracy). Incorporating the Incomes Data improved accuracy to ≈64%, so the candidate was selected.  
        - **Conclusion:** METAM recognized modest improvements when the base dataset is weak, demonstrating its goal-oriented approach.
        """)
    else:
        st.markdown("""
        **Experiment 3: Boston Housing Experiment**  
        - **Setup:** Base dataset derived from the Boston Housing dataset (using 'Id', 'RM', and 'LSTAT' as predictors with target 'expensive').  
        - **Candidate Augmentations:**  
          - Crime/Industrial Features (CRIM, INDUS, NOX, AGE, DIS)  
          - Zoning/Tax Features (ZN, RAD, TAX, PTRATIO)  
          - Structural/Demographic Features (CHAS, B)  
        - **Observation:** The base model’s performance is moderate, and candidate augmentations are evaluated to improve prediction.  
        - **Conclusion:** This experiment illustrates how METAM can explore augmentations in a more complex scenario where the base features are limited.
        """)


if __name__ == '__main__':
    main()