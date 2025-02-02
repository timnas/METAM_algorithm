import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class METAM:
    def __init__(self, base_data, base_target_col, candidate_datasets, join_on, task_type='classification'):
        """
        base_data: pd.DataFrame containing the main dataset.
        base_target_col: str name of the column representing the target variable in base_data.
        candidate_datasets: list of (df, join_on_colname) pairs for possible augmentations.
        join_on: str or list, columns on which to join the base_data and candidate_data.
        task_type: 'classification' or 'regression'.
        """
        self.base_data = base_data
        self.base_target_col = base_target_col
        self.candidate_datasets = candidate_datasets
        self.join_on = join_on
        self.task_type = task_type

        # Keep track of chosen augmentations for reference
        self.chosen_augmentations = []

    def _evaluate_performance(self, data):
        """Trains a simple model and returns performance on a validation split."""
        # Split train/valid for demonstration (simple 80/20 split)
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
        split_idx = int(0.8 * len(data))

        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]

        X_train = train_data.drop(columns=[self.base_target_col])
        y_train = train_data[self.base_target_col]

        X_test = test_data.drop(columns=[self.base_target_col])
        y_test = test_data[self.base_target_col]

        # Simple model: logistic regression for classification, or linear reg for regression
        if self.task_type == 'classification':
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return accuracy_score(y_test, preds)
        else:
            # Could import and use LinearRegression or similar if needed
            raise NotImplementedError("Only classification is implemented in this demo.")

    def run_metam(self, improvement_threshold=0.0):
        """
        Iteratively select the candidate that yields the best improvement in performance
        over the current dataset. Stop when improvement <= threshold or no candidates remain.
        """
        current_performance = self._evaluate_performance(self.base_data)
        improved = True

        while improved and len(self.candidate_datasets) > 0:
            best_gain = 0
            best_idx = -1
            best_augmented_df = None

            for i, (candidate, candidate_join_col) in enumerate(self.candidate_datasets):
                # Merge with base_data
                joined = pd.merge(self.base_data,
                                  candidate.dropna(subset=[candidate_join_col]),
                                  how='left',
                                  left_on=self.join_on,
                                  right_on=candidate_join_col,
                                  suffixes=('', f'_cand{i}'))

                # Possibly remove duplicate join columns if needed
                if candidate_join_col != self.join_on:
                    joined.drop(columns=[candidate_join_col], inplace=True, errors='ignore')

                # Fill numeric NaN with 0 or mean, etc. for simplicity
                for col in joined.select_dtypes(include=[np.number]).columns:
                    joined[col].fillna(joined[col].mean(), inplace=True)

                new_performance = self._evaluate_performance(joined)
                gain = new_performance - current_performance

                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_augmented_df = joined

            # Check if we found any improvement
            if best_gain > improvement_threshold:
                # Accept the best augmentation
                self.base_data = best_augmented_df
                self.chosen_augmentations.append(self.candidate_datasets[best_idx])
                del self.candidate_datasets[best_idx]

                # Update current performance
                current_performance += best_gain
            else:
                improved = False

        return self.base_data, current_performance, self.chosen_augmentations


#
# if __name__ == "__main__":
#     # Example usage
#     base_df = pd.DataFrame({
#         'id': [1, 2, 3, 4, 5],
#         'feature1': [10, 20, 30, 40, 50],
#         'target': [0, 1, 0, 1, 0]
#     })
#
#     cand1 = pd.DataFrame({
#         'cid': [1, 2, 3, 6],
#         'extra1': [5.5, 2.2, 0.1, 9.9]
#     })
#
#     cand2 = pd.DataFrame({
#         'some_id': [1, 4, 5],
#         'extra2': [100, 200, 300]
#     })
#
#     candidates = [
#         (cand1, 'cid'),
#         (cand2, 'some_id')
#     ]
#
#     metam = METAM(
#         base_data=base_df,
#         base_target_col='target',
#         candidate_datasets=candidates,
#         join_on='id'
#     )
#
#     final_data, final_perf, chosen_augs = metam.run_metam(improvement_threshold=0.0)
#     print("Final Performance:", final_perf)
#     print("Augmentations Chosen:", [c[1] for c in chosen_augs])
#     print("Resulting Data:\n", final_data)
