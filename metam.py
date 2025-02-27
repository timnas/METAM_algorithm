import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering


class METAM:
    def __init__(self, base_data, base_target_col, candidate_datasets, join_on,
                 task_type='classification', epsilon=0.05, tau=None, theta=0.8):
        """
        base_data: DataFrame for the base dataset.
        base_target_col: Column name for the target.
        candidate_datasets: List of candidate tuples. Each tuple is (df, join_key)
          where df is a DataFrame and join_key is the column to merge on.
        join_on: Global join key in the base dataset.
        task_type: Only 'classification' is implemented.
        epsilon: Clustering radius for candidate profiles.
        tau: Number of queries per iteration (if None, set to number of clusters).
        theta: Desired utility threshold.
        """
        self.base_data = base_data.copy()
        self.base_target_col = base_target_col
        self.candidate_datasets = candidate_datasets
        self.join_on = join_on
        self.task_type = task_type
        self.epsilon = epsilon
        self.theta = theta
        self.selected_augmentations = []

        # Generate candidate objects with profiles and initial quality scores.
        self.candidates = self.generate_candidates()
        # Cluster candidates based on their profile vectors.
        self.clusters = self.cluster_partition(self.candidates, epsilon)
        self.tau = len(self.clusters) if tau is None else tau

    def utility(self, data):
        """Compute utility via an 80/20 train-test split using logistic regression accuracy."""
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(0.8 * len(data))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        X_train = train_data.drop(columns=[self.base_target_col])
        y_train = train_data[self.base_target_col]
        X_test = test_data.drop(columns=[self.base_target_col])
        y_test = test_data[self.base_target_col]
        if self.task_type == 'classification':
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return accuracy_score(y_test, preds)
        else:
            raise NotImplementedError("Only classification is implemented.")

    def merge_augmentation(self, data, candidate):
        """Merge a candidate dataset onto data using the candidate join key."""
        df_candidate = candidate['df']
        cand_join = candidate['join_on']
        merged = pd.merge(data, df_candidate.dropna(subset=[cand_join]),
                          how='left', left_on=self.join_on, right_on=cand_join,
                          suffixes=('', '_cand'))
        if cand_join != self.join_on:
            merged = merged.drop(columns=[cand_join], errors='ignore')
        for col in merged.select_dtypes(include=[np.number]).columns:
            merged[col] = merged[col].fillna(merged[col].mean())
        return merged

    def compute_profile(self, candidate):
        """
        Compute a profile vector:
          1. Overlap ratio of join keys between base_data and candidate.
          2. Data completeness (1 - missing rate) in the candidate join column.
          3. Normalized column count.
        """
        base_keys = set(self.base_data[self.join_on].dropna().unique())
        cand_keys = set(candidate['df'][candidate['join_on']].dropna().unique())
        overlap = len(base_keys.intersection(cand_keys)) / (len(base_keys) + 1e-6)
        missing_rate = candidate['df'][candidate['join_on']].isna().mean()
        num_cols = candidate['df'].shape[1] / 100.0
        return np.array([overlap, 1 - missing_rate, num_cols])

    def generate_candidates(self):
        """Create candidate objects with computed profiles and initial quality scores."""
        candidates = []
        for tup in self.candidate_datasets:
            df, cand_join = tup
            candidate = {"df": df.copy(), "join_on": cand_join}
            candidate["profile"] = self.compute_profile(candidate)
            candidate["quality_score"] = np.mean(candidate["profile"])
            candidate["queried"] = False
            candidates.append(candidate)
        return candidates

    def cluster_partition(self, candidates, epsilon):
        """Cluster candidates based on their profile vectors using agglomerative clustering."""
        profiles = np.array([cand["profile"] for cand in candidates])
        if len(profiles) == 0:
            return {}
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=epsilon, linkage='average')
        labels = clustering.fit_predict(profiles)
        clusters = {}
        for label, candidate in zip(labels, candidates):
            clusters.setdefault(label, []).append(candidate)
        return clusters

    def update_quality_scores(self, candidate, observed_gain):
        """Update a candidate's quality score based on observed utility gain."""
        alpha = 0.5
        candidate["quality_score"] = alpha * candidate["quality_score"] + (1 - alpha) * observed_gain

    def identify_group(self, clusters, t):
        """Select up to t unqueried candidates (one per cluster) with high quality scores."""
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
        """Remove any augmentation whose removal still keeps utility above theta."""
        minimal_set = solution_set.copy()
        for aug in solution_set:
            temp_set = [a for a in minimal_set if a != aug]
            temp_data = self.base_data.copy()
            for a in temp_set:
                temp_data = self.merge_augmentation(temp_data, a)
            if self.utility(temp_data) >= self.theta:
                minimal_set.remove(aug)
        return minimal_set

    def run_metam(self):
        """Run the adaptive querying loop until the utility threshold is met or candidates are exhausted."""
        current_data = self.base_data.copy()
        current_util = self.utility(current_data)
        prev_util = current_util
        iteration = 0

        while current_util < self.theta and any(not cand["queried"] for cand in self.candidates):
            iteration += 1
            print(f"Iteration {iteration}, current utility: {current_util:.4f}")
            unqueried = [cand for cand in self.candidates if not cand["queried"]]
            if not unqueried:
                break
            candidate = max(unqueried, key=lambda x: x["quality_score"])
            candidate["queried"] = True
            merged_candidate = self.merge_augmentation(current_data, candidate)
            cand_util = self.utility(merged_candidate)
            observed_gain = max(cand_util - current_util, 0)
            self.update_quality_scores(candidate, observed_gain)
            print(
                f"Queried candidate (join key: {candidate['join_on']}) | Quality: {candidate['quality_score']:.4f} | Utility: {cand_util:.4f}")

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
                print(f"Selected candidate (join key: {best_candidate['join_on']}) improves utility to {best_util:.4f}")
                current_data = self.merge_augmentation(current_data, best_candidate)
                self.selected_augmentations.append(best_candidate)
                current_util = best_util
            else:
                print("No candidate improved utility significantly. Stopping.")
                break

            if self.check_stop_criterion(current_util, prev_util):
                break
            prev_util = current_util

        minimal_solution = self.identify_minimal(self.selected_augmentations)
        print("Final selected augmentations:", [cand["join_on"] for cand in minimal_solution])
        return current_data, current_util, minimal_solution


# --- Example using Boston Housing (split into base and candidates) ---

if __name__ == "__main__":
    # Load the Boston Housing dataset (assumed to be saved as "boston_housing.csv")
    # The CSV should have columns: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV.
    df = pd.read_csv("boston_housing.csv")

    # Create a dummy join key "Id" from the row index.
    df["Id"] = df.index

    # Create a binary target "expensive": 1 if MEDV (in $1000's) is at or above the median, else 0.
    median_medv = df["MEDV"].median()
    df["expensive"] = (df["MEDV"] >= median_medv).astype(int)

    # Define the base dataset using a few columns.
    base_columns = ["Id", "RM", "LSTAT", "expensive"]
    base_df = df[base_columns].copy()

    # Define candidate augmentation datasets by splitting the remaining columns.
    # Candidate 1: Use columns related to crime and industrial activity.
    cand1_columns = ["Id", "CRIM", "INDUS", "NOX", "AGE", "DIS"]
    cand1_df = df[cand1_columns].copy()

    # Candidate 2: Use columns related to zoning and taxation.
    cand2_columns = ["Id", "ZN", "RAD", "TAX", "PTRATIO"]
    cand2_df = df[cand2_columns].copy()

    # Candidate 3: Use columns related to structural or demographic features.
    cand3_columns = ["Id", "CHAS", "B"]
    cand3_df = df[cand3_columns].copy()

    # Create candidate tuples (each with the global join key "Id").
    candidate1 = (cand1_df, "Id")
    candidate2 = (cand2_df, "Id")
    candidate3 = (cand3_df, "Id")
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
    print("Final selected augmentations (join keys):", [cand["join_on"] for cand in chosen_augs])
    print("Resulting Data (first 10 rows):")
    print(final_data.head(10))
