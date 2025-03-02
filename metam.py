import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def compute_utility(df, target_col, drop_cols, random_state=42):
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
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

def compute_semantic_similarity(base_df, candidate_df):
    """
    Compute semantic similarity between two datasets at the column level.
    For each column in the base dataset, find the most similar column in the candidate dataset.
    The final similarity score is the average of these best match similarities.
    """
    base_cols = base_df.columns.tolist()
    candidate_cols = candidate_df.columns.tolist()

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed each column name individually
    base_embeddings = model.encode(base_cols)
    candidate_embeddings = model.encode(candidate_cols)

    # Compute pairwise cosine similarity (base vs candidate columns)
    similarity_matrix = cosine_similarity(base_embeddings, candidate_embeddings)

    # For each base column, find the max similarity to any candidate column (best match for each column)
    best_match_scores = similarity_matrix.max(axis=1)

    # Final semantic similarity is the average of best match scores across all base columns
    final_similarity_score = np.mean(best_match_scores)

    return final_similarity_score


class METAM:
    def __init__(self, base_data, base_target_col, candidate_datasets, join_on,
                 task_type='classification', epsilon=0.05, tau=None, theta=0.8):
        self.base_data = base_data.copy()
        self.base_target_col = base_target_col
        # Each candidate tuple can be (df, join_key, candidate_name) or (df, join_key)
        self.candidate_datasets = candidate_datasets
        self.join_on = join_on
        self.task_type = task_type
        self.epsilon = epsilon
        self.theta = theta
        self.selected_augmentations = []

        self.candidates = self.generate_candidates()
        print("DEBUG: Generated Candidate Profiles:")
        for i, cand in enumerate(self.candidates):
            print(
                f"Candidate {i + 1} ({cand['name']}): join_on={cand['join_on']}, Profile={cand['profile']}, Initial Quality Score={cand['quality_score']:.4f}")

        self.clusters = self.cluster_partition(self.candidates, epsilon)
        self.tau = len(self.clusters) if tau is None else tau

    def utility(self, data):
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(0.8 * len(data))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        features_to_drop = [self.base_target_col, self.join_on]
        X_train = train_data.drop(columns=features_to_drop).select_dtypes(include=[np.number])
        X_test = test_data.drop(columns=features_to_drop).select_dtypes(include=[np.number])
        train_means = X_train.mean()
        X_train = X_train.fillna(train_means)
        X_test = X_test.fillna(train_means)
        y_train = train_data[self.base_target_col]
        y_test = test_data[self.base_target_col]
        if self.task_type == 'classification':
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return accuracy_score(y_test, preds)
        else:
            raise NotImplementedError("Only classification is implemented.")

    def merge_augmentation(self, data, candidate):
        df_candidate = candidate['df'].copy()
        cand_join = candidate['join_on']
        # Ensure join keys are strings for compatibility.
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

        semantic_similarity = compute_semantic_similarity(self.base_data, candidate['df'])

        profile_vector = np.array([overlap, 1 - missing_rate, num_cols, semantic_similarity])

        return profile_vector

    def generate_candidates(self):
        candidates = []
        for tup in self.candidate_datasets:
            if len(tup) == 3:
                df, cand_join, name = tup
            else:
                df, cand_join = tup
                name = cand_join  # default name
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
        print("DEBUG: Clustering results:", clusters)
        return clusters

    def update_quality_scores(self, candidate, observed_gain):
        alpha = 0.5
        candidate["quality_score"] = alpha * candidate["quality_score"] + (1 - alpha) * observed_gain
        print(f"DEBUG: Updated quality score for candidate ({candidate['name']}): {candidate['quality_score']:.4f}")

    def identify_group(self, clusters, t):
        group = []
        for cluster in clusters.values():
            unqueried = [cand for cand in cluster if not cand["queried"]]
            if unqueried:
                best = max(unqueried, key=lambda x: x["quality_score"])
                group.append(best)
            if len(group) >= t:
                break
        print("DEBUG: Identified group for querying:", [cand["name"] for cand in group])
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
            print(f"DEBUG: Utility without candidate ({aug['name']}): {util_without:.4f}")
            if util_without >= self.theta:
                print(f"DEBUG: Removing candidate ({aug['name']}) from solution set")
                minimal_set.remove(aug)
        return minimal_set

    def run_metam(self):
        current_data = self.base_data.copy()
        current_util = self.utility(current_data)
        print(f"DEBUG: Initial utility: {current_util:.4f}")
        prev_util = current_util
        iteration = 0

        while current_util < self.theta and any(not cand["queried"] for cand in self.candidates):
            iteration += 1
            print(f"\nDEBUG: Iteration {iteration}, current utility: {current_util:.4f}")
            unqueried = [cand for cand in self.candidates if not cand["queried"]]
            if not unqueried:
                break
            candidate = max(unqueried, key=lambda x: x["quality_score"])
            candidate["queried"] = True
            merged_candidate = self.merge_augmentation(current_data, candidate)
            cand_util = self.utility(merged_candidate)
            observed_gain = max(cand_util - current_util, 0)
            print(
                f"DEBUG: Candidate ({candidate['name']}) utility after merge: {cand_util:.4f} (Gain: {observed_gain:.4f})")
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
                    print(f"DEBUG: Group candidate ({cand['name']}) utility after merge: {util_group:.4f}")

            candidates_to_consider = [candidate] + [cand for cand, _ in group_results]
            best_candidate = max(candidates_to_consider,
                                 key=lambda c: self.utility(self.merge_augmentation(current_data, c)))
            best_util = self.utility(self.merge_augmentation(current_data, best_candidate))
            print(f"DEBUG: Best candidate selected has utility: {best_util:.4f}")

            if best_util > current_util:
                print(
                    f"DEBUG: Selected candidate ({best_candidate['name']}) improves utility from {current_util:.4f} to {best_util:.4f}")
                current_data = self.merge_augmentation(current_data, best_candidate)
                self.selected_augmentations.append(best_candidate)
                current_util = best_util
            else:
                print("DEBUG: No candidate improved utility significantly. Stopping.")
                break

            if self.check_stop_criterion(current_util, prev_util):
                print("DEBUG: Stop criterion met (minimal improvement).")
                break
            prev_util = current_util

        minimal_solution = self.identify_minimal(self.selected_augmentations)
        print("DEBUG: Final selected augmentations:", [cand["name"] for cand in minimal_solution])
        return current_data, current_util, minimal_solution


if __name__ == "__main__":
    # Load the Boston Housing dataset (assumed to be saved as "boston_housing.csv")
    # The CSV should have columns: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV.
    df = pd.read_csv("data/AmesHousing.csv")
    df.columns = (df.columns
                    .str.replace(" ", "_")
                    .str.replace("-", "_")
                    .str.replace(r"[^\w_]", "")  # Remove any special symbols like $, #
                    .str.strip())

    # Create a dummy join key "Id" from the row index.
    df["Id"] = df.index.astype(str)

    # Create a binary target "expensive": 1 if MEDV (in $1000's) is at or above the median, else 0.
    df["expensive"] = (df["SalePrice"] >= df["SalePrice"].median()).astype(int)

    # Define the base dataset using a few key predictors.
    base_columns = ["Id", "Lot_Area", "Overall_Qual", "expensive"]
    base_df = df[base_columns].copy()

    # Define candidate augmentation datasets by splitting the remaining columns.
    # Candidate 1: Use columns related to crime and industrial activity.
    cand1_columns = ["Id", "Neighborhood", "Year_Built", "Garage_Cars"]
    cand1_df = df[cand1_columns].copy()

    # Candidate 2: Use columns related to zoning and taxation.
    cand2_df = pd.DataFrame({
        "Id": df["Id"],
        "Median_Income": np.random.randint(30000, 100000, size=len(df)),
        "Unemployment_Rate": np.random.uniform(3, 12, size=len(df))
    })

    cand3_df = pd.DataFrame({
        "Id": df["Id"],
        "Year_Built": df["Year_Built"],
        "Avg_Temp_During_Build": np.random.uniform(-5, 30, size=len(df))
    })

    # Create candidate tuples (each with the global join key "Id") and add candidate names.
    candidate1 = (cand1_df, "Id", "Neighborhood/Structural Features")
    candidate2 = (cand2_df, "Id", "Economic Indicators")
    candidate3 = (cand3_df, "Id", "Weather Data During Construction")
    candidates = [candidate1, candidate2, candidate3]

    # Initialize and run METAM.
    # Here, our task is to predict "expensive".
    metam = METAM(
        base_data=base_df,
        base_target_col="expensive",
        candidate_datasets=candidates,
        join_on="Id",
        theta=0.85  # Adjust theta as needed
    )

    final_data, final_perf, chosen_augs = metam.run_metam()
    print("\nFinal Performance:", final_perf)
    print("Final selected augmentations (names):", [cand["name"] for cand in chosen_augs])
    print("Resulting Data (first 10 rows):")
    print(final_data.head(10))

