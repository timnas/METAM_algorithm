import time
import pandas as pd
import numpy as np
from streamlit_app import METAM, compute_utility  # Ensure metam.py contains your METAM class and functions

def run_experiment(exp_name, metam_instance, base_df):
    print("------------------------------------------------------")
    print("Running experiment:", exp_name)
    # Compute base utility (the performance of the unaugmented base dataset)
    base_util = metam_instance.utility(base_df.copy())
    print("Base Utility:", base_util)

    start = time.time()
    final_data, final_util, chosen_augs = metam_instance.run_metam()
    end = time.time()

    running_time = end - start
    print("Final Utility:", final_util)
    print("Running Time (seconds):", running_time)

    chosen_aug_names = [cand["name"] for cand in chosen_augs]
    if chosen_aug_names:
        print("Selected Augmentations:", ", ".join(chosen_aug_names))
    else:
        print("Selected Augmentations: None")
    print("------------------------------------------------------\n")


def experiment_seattle_housing_regression():
    # Experiment: Housing Price in Seattle Regression
    base_df = pd.read_csv("data/seattle_housing_prices.csv")
    base_df.rename(columns={"zip_code": "zipcode"}, inplace=True)
    base_df["price"] = pd.to_numeric(base_df["price"], errors="coerce")
    base_df = base_df.dropna(subset=["price"])

    candidate_crime = pd.read_csv("data/seattle_prop_crime_rate.csv")
    candidate_crime_tuple = (candidate_crime.copy(), "zipcode", "Crime Rate Data")

    candidate_pet = pd.read_csv("data/seattle_pet_licenses.csv")
    candidate_pet.rename(columns={"zip_code": "zipcode"}, inplace=True)
    candidate_pet_tuple = (candidate_pet.copy(), "zipcode", "Pet Licenses Data")

    candidate_incomes = pd.read_csv("data/seattle_wa_incomes_zip_code.csv")
    candidate_incomes.rename(columns={"Zip Code": "zipcode"}, inplace=True)
    candidate_incomes_tuple = (candidate_incomes.copy(), "zipcode", "Incomes Data")

    candidates = [candidate_crime_tuple, candidate_pet_tuple, candidate_incomes_tuple]
    theta = 0.7
    metam_instance = METAM(base_df, "price", candidates, "zipcode", task_type="regression", theta=theta)
    run_experiment("Housing Price in Seattle Regression", metam_instance, base_df)


def experiment_expensive_housing_classification_seattle():
    # Experiment: Expensive Housing Classification in Seattle
    base_df = pd.read_csv("data/seattle_housing_prices.csv")
    base_df.rename(columns={"zip_code": "zipcode"}, inplace=True)
    median_price = base_df["price"].median()
    base_df["expensive"] = (base_df["price"] >= median_price).astype(int)

    candidate_crime = pd.read_csv("data/seattle_prop_crime_rate.csv")
    candidate_crime_tuple = (candidate_crime.copy(), "zipcode", "Crime Rate Data")

    candidate_pet = pd.read_csv("data/seattle_pet_licenses.csv")
    candidate_pet.rename(columns={"zip_code": "zipcode"}, inplace=True)
    candidate_pet_tuple = (candidate_pet.copy(), "zipcode", "Pet Licenses Data")

    candidate_incomes = pd.read_csv("data/seattle_wa_incomes_zip_code.csv")
    candidate_incomes.rename(columns={"Zip Code": "zipcode"}, inplace=True)
    candidate_incomes_tuple = (candidate_incomes.copy(), "zipcode", "Incomes Data")

    candidates = [candidate_crime_tuple, candidate_pet_tuple, candidate_incomes_tuple]
    theta = 0.98
    metam_instance = METAM(base_df, "expensive", candidates, "zipcode", task_type="classification", theta=theta)
    run_experiment("Expensive Housing Classification in Seattle", metam_instance, base_df)


def experiment_high_cat_ratio_classification():
    # Experiment: High Cat Ratio Classification
    pets_df = pd.read_csv("data/seattle_pet_licenses.csv")
    pets_df.rename(columns={"zip_code": "zipcode"}, inplace=True)
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
    theta = 0.7
    metam_instance = METAM(base_df, "cat_present", candidates, "zipcode", task_type="classification", theta=theta)
    run_experiment("High Cat Ratio Classification", metam_instance, base_df)


def experiment_boston_housing_classification():
    # Experiment: Classification - Expensive Housing in Boston
    df = pd.read_csv("data/boston_housing.csv")
    df["Id"] = df.index.astype(str)
    median_medv = df["MEDV"].median()
    df["expensive"] = (df["MEDV"] >= median_medv).astype(int)
    base_columns = ["Id", "RM", "LSTAT", "expensive"]
    base_df = df.loc[:, base_columns].copy()

    cand1_columns = ["Id", "CRIM", "INDUS", "NOX", "AGE", "DIS"]
    cand1_df = df[cand1_columns].copy()
    candidate1 = (cand1_df, "Id", "Crime/Industrial Features")

    cand2_columns = ["Id", "ZN", "RAD", "TAX", "PTRATIO"]
    cand2_df = df[cand2_columns].copy()
    candidate2 = (cand2_df, "Id", "Zoning/Tax Features")

    cand3_columns = ["Id", "CHAS", "B"]
    cand3_df = df[cand3_columns].copy()
    candidate3 = (cand3_df, "Id", "Structural/Demographic Features")

    candidates = [candidate1, candidate2, candidate3]
    theta = 0.85
    metam_instance = METAM(base_df, "expensive", candidates, "Id", task_type="classification", theta=theta)
    run_experiment("Classification - Expensive Housing in Boston", metam_instance, base_df)


if __name__ == '__main__':
    print("Starting METAM experiments...\n")
    experiment_seattle_housing_regression()
    experiment_expensive_housing_classification_seattle()
    experiment_high_cat_ratio_classification()
    experiment_boston_housing_classification()
    print("All experiments completed.")
