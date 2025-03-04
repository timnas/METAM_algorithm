import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------- Custom CSS with the New Color Palette ----------------------
st.markdown("""
    <style>
    /* Overall page background color */
    .body {
        background-color: #FAF2E4 /* Off-white / light beige */
    }

    /* Title styling */
    .main-title {
        background-color: #F2A34C; /* Light green */
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        # font-size: 3em;
        text-align: center;
        # color: #F2A34C; /* Warm orange */
    }
    .slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 18px;
          height: 18px;
          border-radius: 10px;
          background-color: var(--SliderColor);
          overflow: visible;
          cursor: pointer;
    }

    .flex-container {
      display: flex;
      background-color: DodgerBlue;
      justify-content: space-between; 
      margin-top: 20px;
    }

    .flex-container > div {
      background-color: #f1f1f1;
      margin: 10px;
      padding: 20px;
      font-size: 30px;
      justify-content: space-between; 
    }

    /* Large metric styling */
    .metric {
        font-size: 2.5em;
        color: #F2A34C; /* Warm orange */
        font-weight: bold;
    }

    /* Experiment box styling */
    .experiment-box {
        background-color: #B7CDB0; /* Light green */
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }

    /* Results box styling */
    .results-box {
        background-color: #B7CDB0; /* Dark green */
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 5px;
        text-align: center;
    }

    /* Analysis box or other accent styling */
    .analysis {
        background-color: #F2A34C; /* Warm orange for accent */
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #D5D8DC;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------- Utility Function ----------------------
def compute_utility(df, target_col, drop_cols, task_type='classification', random_state=42):
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    X_train = train_df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    X_test = test_df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    train_means = X_train.mean()
    X_train = X_train.fillna(train_means)
    X_test = X_test.fillna(train_means)
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    if task_type == 'classification':
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds)
    elif task_type == 'regression':
        model = LinearRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)  # R² score
        return max(score, 0)  # Clip negative values to 0
    else:
        raise NotImplementedError("Task type not implemented.")


# ---------------------- Semantic Similarity Function ----------------------
def compute_semantic_similarity(base_df, candidate_df, baseline=0.2):
    base_cols = base_df.columns.tolist()
    candidate_cols = candidate_df.columns.tolist()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    base_embeddings = model.encode(base_cols)
    candidate_embeddings = model.encode(candidate_cols)
    similarity_matrix = cosine_similarity(base_embeddings, candidate_embeddings)
    best_match_scores = similarity_matrix.max(axis=1)
    raw_score = np.mean(best_match_scores)
    adjusted_score = (raw_score - baseline) / (1 - baseline)
    adjusted_score = np.clip(adjusted_score, 0, 1)
    return adjusted_score


# ---------------------- METAM Algorithm ----------------------
class METAM:
    def __init__(self, base_data, base_target_col, candidate_datasets, join_on,
                 task_type='classification', epsilon=0.05, tau=None, theta=0.8):
        self.base_data = base_data.copy()
        self.base_target_col = base_target_col
        self.candidate_datasets = candidate_datasets
        self.join_on = join_on
        self.task_type = task_type
        self.epsilon = epsilon
        self.theta = theta
        self.selected_augmentations = []

        self.candidates = self.generate_candidates()
        # For each candidate, compute the utility after a single merge
        for cand in self.candidates:
            merged_once = self.merge_augmentation(self.base_data, cand)
            cand_utility = self.utility(merged_once)
            cand["candidate_utility"] = cand_utility

        st.markdown("### Candidate Augmentation Details")
        candidate_details = []
        for i, cand in enumerate(self.candidates):
            candidate_details.append({
                "Candidate": cand["name"],
                "Join Key": cand["join_on"],
                "Profile": np.array2string(cand["profile"], precision=2),
                "Initial Quality Score": f"{cand['quality_score']:.4f}"
            })
        st.table(pd.DataFrame(candidate_details))

        self.clusters = self.cluster_partition(self.candidates, epsilon)
        self.tau = len(self.clusters) if tau is None else tau

    def utility(self, data):
        return compute_utility(data, self.base_target_col, [self.base_target_col, self.join_on],
                               task_type=self.task_type)

    def merge_augmentation(self, data, candidate):
        df_candidate = candidate['df'].copy()
        cand_join = candidate['join_on']
        data[self.join_on] = data[self.join_on].astype(str)
        df_candidate[cand_join] = df_candidate[cand_join].astype(str)
        merged = pd.merge(data, df_candidate.dropna(subset=[cand_join]),
                          how='left', left_on=self.join_on, right_on=cand_join,
                          suffixes=('', '_cand'))
        if cand_join != self.join_on:
            merged = merged.drop(columns=[cand_join], errors='ignore')
        for col in merged.select_dtypes(include=[np.number]).columns:
            merged[col] = merged[col].fillna(merged[col].mean())
        return merged

    def compute_profile(self, candidate):
        base_keys = set(self.base_data[self.join_on].dropna().astype(str).unique())
        cand_keys = set(candidate['df'][candidate['join_on']].dropna().astype(str).unique())
        overlap = len(base_keys.intersection(cand_keys)) / (len(base_keys) + 1e-6)
        missing_rate = candidate['df'][candidate['join_on']].isna().mean()
        num_cols = candidate['df'].shape[1] / 100.0
        semantic_similarity = compute_semantic_similarity(self.base_data, candidate['df'], baseline=0.2)
        profile_vector = np.array([overlap, 1 - missing_rate, num_cols, semantic_similarity])
        return profile_vector

    def generate_candidates(self):
        candidates = []
        for tup in self.candidate_datasets:
            if len(tup) == 3:
                df, cand_join, name = tup
            else:
                df, cand_join = tup
                name = cand_join
            candidate = {"df": df.copy(), "join_on": cand_join, "name": name}
            candidate["profile"] = self.compute_profile(candidate)
            candidate["quality_score"] = np.mean(candidate["profile"])
            candidate["queried"] = False
            candidates.append(candidate)
        return candidates

    def cluster_partition(self, candidates, epsilon):
        profiles = np.array([cand["profile"] for cand in candidates])
        if len(profiles) < 2:
            return {0: candidates}
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=epsilon, linkage='average')
        labels = clustering.fit_predict(profiles)
        clusters = {}
        for label, candidate in zip(labels, candidates):
            clusters.setdefault(label, []).append(candidate)
        return clusters

    def update_quality_scores(self, candidate, observed_gain):
        alpha = 0.5
        candidate["quality_score"] = alpha * candidate["quality_score"] + (1 - alpha) * observed_gain

    def identify_group(self, clusters, t):
        group = []
        for cluster in clusters.values():
            unqueried = [cand for cand in cluster if not cand["queried"]]
            if unqueried:
                best = max(unqueried, key=lambda x: x["quality_score"])
                group.append(best)
            if len(group) >= t:
                break
        return group

    def check_stop_criterion(self, current_util, prev_util, tol=1e-3):
        return (current_util - prev_util) < tol

    def identify_minimal(self, solution_set):
        minimal_set = solution_set.copy()
        for aug in solution_set:
            temp_set = [a for a in minimal_set if a != aug]
            temp_data = self.base_data.copy()
            for a in temp_set:
                temp_data = self.merge_augmentation(temp_data, a)
            util_without = self.utility(temp_data)
            if util_without >= self.theta:
                minimal_set.remove(aug)
        return minimal_set

    def run_metam(self, max_iter=50):
        current_data = self.base_data.copy()
        current_util = self.utility(current_data)
        prev_util = current_util
        iteration = 0

        while current_util < self.theta and any(
                not cand["queried"] for cand in self.candidates) and iteration < max_iter:
            iteration += 1
            unqueried = [cand for cand in self.candidates if not cand["queried"]]
            if not unqueried:
                break
            candidate = max(unqueried, key=lambda x: x["quality_score"])
            candidate["queried"] = True
            merged_candidate = self.merge_augmentation(current_data, candidate)
            cand_util = self.utility(merged_candidate)
            observed_gain = max(cand_util - current_util, 0)
            self.update_quality_scores(candidate, observed_gain)

            group = self.identify_group(self.clusters, t=self.tau)
            group_results = []
            for cand in group:
                if not cand["queried"]:
                    merged_group = self.merge_augmentation(current_data, cand)
                    util_group = self.utility(merged_group)
                    group_results.append((cand, util_group))
                    cand["queried"] = True
                    self.update_quality_scores(cand, max(util_group - current_util, 0))

            candidates_to_consider = [candidate] + [cand for cand, _ in group_results]
            best_candidate = max(candidates_to_consider,
                                 key=lambda c: self.utility(self.merge_augmentation(current_data, c)))
            best_util = self.utility(self.merge_augmentation(current_data, best_candidate))

            if best_util > current_util:
                current_data = self.merge_augmentation(current_data, best_candidate)
                self.selected_augmentations.append(best_candidate)
                current_util = best_util
            else:
                break

            if self.check_stop_criterion(current_util, prev_util):
                break
            prev_util = current_util

        minimal_solution = self.identify_minimal(self.selected_augmentations)
        # st.write("Final selected augmentations:", [cand["name"] for cand in minimal_solution])
        return current_data, current_util, minimal_solution

def main():
    st.markdown("""
        <style>
        [data-baseweb="slider"] .ThumbContainer {
            color: #FFA725 !important;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("<div class='main-title'><h1>METAM: Goal-Oriented Data Augmentation</h1></div>", unsafe_allow_html=True)
    st.sidebar.header("Select Experiment")

    # Added "Expensive Housing Classification" to the experiment list
    experiment = st.sidebar.selectbox(
        "Choose Experiment",
        (
            "Classification - Expensive Housing in Boston",
            "Expensive Housing Classification in Seattle",
            "High Cat Ratio Classification",
            "Housing Price in Seattle Regression "
        )
    )

    start_time = time.time()

    if experiment == "Housing Price in Seattle Regression ":
        st.markdown("<div class='experiment-box'><h3>Experiment 4: Housing Price in Seattle Regression </h3></div>", unsafe_allow_html=True)
        base_df = pd.read_csv("data/seattle_housing_prices.csv")
        base_df.rename(columns={"zip_code": "zipcode"}, inplace=True)
        base_df["price"] = pd.to_numeric(base_df["price"], errors='coerce')
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
        theta = st.sidebar.slider("Utility Threshold (theta)", 0.2, 1.0, 0.7)

        metam = METAM(
            base_data=base_df,
            base_target_col="price",
            candidate_datasets=candidates,
            join_on="zipcode",
            theta=theta,
            task_type="regression"
        )
        final_data, final_util, chosen_augs = metam.run_metam()

    elif experiment == "Expensive Housing Classification in Seattle":
        # NEW experiment block for classification
        st.markdown("<div class='experiment-box'><h3>Experiment 2: Expensive Housing Classification in Seattle</h3></div>", unsafe_allow_html=True)
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
        theta = st.sidebar.slider("Utility Threshold (theta)", 0.8, 1.0, 0.98)

        metam = METAM(
            base_data=base_df,
            base_target_col="expensive",
            candidate_datasets=candidates,
            join_on="zipcode",
            theta=theta,
            task_type="classification"
        )
        final_data, final_util, chosen_augs = metam.run_metam()

    elif experiment == "High Cat Ratio Classification":
        st.markdown("<div class='experiment-box'><h3>Experiment 3: High Cat Ratio Classification</h3></div>", unsafe_allow_html=True)
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
        theta = st.sidebar.slider("Utility Threshold (theta)", 0.6, 0.8, 0.7)

        metam = METAM(
            base_data=base_df,
            base_target_col="cat_present",
            candidate_datasets=candidates,
            join_on="zipcode",
            theta=theta,
            task_type="classification"
        )
        final_data, final_util, chosen_augs = metam.run_metam()

    elif experiment == "Classification - Expensive Housing in Boston":
        st.markdown("<div class='experiment-box'><h3>Experiment 1: Classification - Expensive Housing in Boston</h3></div>", unsafe_allow_html=True)
        df = pd.read_csv("data/boston_housing.csv")
        df["Id"] = df.index.astype(str)
        median_medv = df["MEDV"].median()
        df["expensive"] = (df["MEDV"] >= median_medv).astype(int)
        base_columns = ["Id", "RM", "LSTAT", "expensive"]
        base_df = df[base_columns].copy()
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
        theta = st.sidebar.slider("Utility Threshold (theta)", 0.8, 1.0, 0.85)

        metam = METAM(
            base_data=base_df,
            base_target_col="expensive",
            candidate_datasets=candidates,
            join_on="Id",
            theta=theta,
            task_type="classification"
        )
        final_data, final_util, chosen_augs = metam.run_metam()

    # 1) Compute base utility before run_metam
    base_utility = metam.utility(base_df.copy())  # Utility of the unaugmented base dataset

    running_time = time.time() - start_time
    st.markdown("<div class='results-box'><h3>Results</h3></div>", unsafe_allow_html=True)

    chosen_aug_names = [cand["name"] for cand in chosen_augs]
    chosen_aug_str = ", ".join(chosen_aug_names) if chosen_aug_names else "None"

    st.markdown("""
        <div style="
            justify-content: space-between; 
            margin-top: 20px;
        ">
            <!-- Box 1: Initial Utility -->
            <div style="
                background-color: #6A9C89; 
                border-radius: 8px; 
                padding: 15px; 
                margin-right: 10px; 
                text-align: center;
            ">
                <h4 style="margin: 5px 0;">Initial Utility</h4>
                <p style="font-size:1.5em; font-weight:bold;">""" + f"{base_utility:.4f}" + """</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        justify-content: space-between; 
        margin-top: 20px;
    ">
        <!-- Box 2: Final Utility -->
        <div style="
            background-color: #6A9C89; 
            border-radius: 8px; 
            padding: 15px; 
            margin-right: 10px; 
            text-align: center;
        ">
            <h4 style="margin: 5px 0;">Final Utility</h4>
            <p style="font-size:1.5em; font-weight:bold;">""" + f"{final_util:.4f}" + """</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        justify-content: space-between; 
        margin-top: 20px;
    ">
        <!-- Box 3: Running Time -->
        <div style="
            background-color: #6A9C89; 
            border-radius: 8px; 
            padding: 15px; 
            margin-right: 10px; 
            text-align: center;
        ">
            <h4 style="margin: 5px 0;">Running Time (seconds)</h4>
            <p style="font-size:1.5em; font-weight:bold;">""" + f"{running_time:.2f}" + """</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        margin-top: 20px;
    ">
        <!-- Box 4: Selected Augmentations -->
        <div style="
            background-color: #6A9C89; 
            border-radius: 8px; 
            padding: 15px; 
            text-align: center;
        ">
            <h4 style="margin: 5px 0;">Selected Augmentations</h4>
            <p style="font-size:1.2em;">""" + chosen_aug_str + """</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3>Analysis Summary</h3>", unsafe_allow_html=True)
    if experiment == "Housing Price in Seattle Regression ":
        st.markdown("""
        **Experiment 4: Housing Price in Seattle Regression **  
        - **Setup:** Base dataset from Seattle Housing Prices with target 'price' (continuous).  
          Candidate augmentations include Crime Rate Data, Pet Licenses Data, and Incomes Data.  
        - **Conclusion:** This experiment demonstrates that METAM can be applied to complex regression tasks. By augmenting the base housing data with external factors, the predictive power (R² score) of the model increases, indicating improved performance.
        """)
    elif experiment == "Expensive Housing Classification in Seattle ":
        st.markdown("""
        **Experiment 2: Expensive Housing Classification in Seattle**  
        - **Setup:** Base dataset from Seattle Housing Prices (using 'zipcode') with a binary target 'expensive' (price >= median).  
          Candidate augmentations include Crime Rate Data, Pet Licenses Data, and Incomes Data.  
        - **Observation:** The base model already provides a strong baseline for classification. METAM checks each candidate augmentation’s utility gain and only merges those that improve accuracy above the threshold.  
        - **Conclusion:** This scenario highlights METAM’s ability to avoid unnecessary augmentations when the base dataset is already highly predictive, consistent with the minimality principle.
        """)
    elif experiment == "High Cat Ratio Classification":
        st.markdown("""
        **Experiment 3: High Cat Ratio Classification**  
        - **Setup:** Base dataset aggregated from Seattle Pet Licenses (using 'zipcode' and weak predictor 'total_pets') with target 'cat_present'.  
        - **Observation:** The base model produced modest performance (≈62% accuracy). Incorporating the Incomes Data improved accuracy to ≈64%, so the candidate was selected.  
        - **Conclusion:** METAM recognized modest improvements when the base dataset is weak, demonstrating its goal-oriented approach.
        """)
    elif experiment == "Classification - Expensive Housing in Boston":
        st.markdown("""
        **Experiment 1: Classification - Expensive Housing in Boston**  
        - **Setup:** Base dataset derived from the Boston Housing dataset (using 'Id', 'RM', and 'LSTAT' as predictors with target 'expensive').  
        - **Candidate Augmentations:**  
          - Crime/Industrial Features (CRIM, INDUS, NOX, AGE, DIS)  
          - Zoning/Tax Features (ZN, RAD, TAX, PTRATIO)  
          - Structural/Demographic Features (CHAS, B)  
        - **Observation:** The base model’s performance was moderate (≈81.37% accuracy). The algorithm selected "Zoning/Tax Features," boosting accuracy to ≈85.29%.  
        - **Conclusion:**  This experiment demonstrates that when a rich dataset is partitioned into distinct feature groups, METAM can be used for feature reduction—effectively identifying which groups add incremental predictive power and which are redundant. 
        """)


if __name__ == '__main__':
    main()