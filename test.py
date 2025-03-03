import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import the function from your script
from metam import compute_semantic_similarity, METAM

# Initialize the model globally for speed
model = SentenceTransformer('all-MiniLM-L6-v2')


def test_semantic_similarity():
    base_df = pd.DataFrame(columns=["House_Price", "Number_of_Bedrooms", "Year_Built"])
    candidate_df = pd.DataFrame(columns=["SalePrice", "Bedrooms", "Construction_Year"])

    similarity_score = compute_semantic_similarity(base_df, candidate_df)

    assert 0 <= similarity_score <= 1, f"Similarity out of bounds: {similarity_score}"
    print(
        f"Semantic similarity between {base_df.columns.tolist()} and {candidate_df.columns.tolist()} = {similarity_score:.4f}")


# Run the test
# test_semantic_similarity()

# Base dataset: Ames Housing
# base_df = pd.DataFrame({
#     "Id": [1, 2, 3, 4, 5],
#     "Lot_Area": [8500, 9000, 9250, 9400, 9600],
#     "Overall_Qual": [7, 6, 8, 7, 5],
#     "expensive": [1, 0, 1, 1, 0]
# })
#
# # Candidate dataset with semantically similar features
# cand1_df = pd.DataFrame({
#     "Id": [1, 2, 3, 4, 5],
#     "SalePrice": [200000, 180000, 250000, 220000, 170000],  # Similar to "House_Price"
#     "Number_of_Bedrooms": [3, 2, 4, 3, 2],  # Similar to "Bedrooms"
#     "Construction_Year": [2005, 2003, 2010, 2008, 1999]  # Similar to "Year_Built"
# })
#
# cand2_df = pd.DataFrame({
#     "Id": [1, 2, 3, 4, 5],
#     "Unrelated_Column_1": ["A", "B", "C", "D", "E"],
#     "Unrelated_Column_2": [100, 200, 300, 400, 500]
# })
#
# # Creating candidates with explicit names
# candidate1 = (cand1_df, "Id", "Similar Features")  # Should be chosen
# candidate2 = (cand2_df, "Id", "Unrelated Data")  # Should not be chosen
#
# candidates = [candidate1, candidate2]
#
# # Run METAM with similarity check
# metam = METAM(
#     base_data=base_df,
#     base_target_col="expensive",
#     candidate_datasets=candidates,
#     join_on="Id",
#     theta=0.85  # Higher threshold
# )
#
# final_data, final_perf, chosen_augs = metam.run_metam()
#
# # Print results
# print("\nFinal Performance:", final_perf)
# print("Final selected augmentations:", [cand["name"] for cand in chosen_augs])
# print("Resulting Data (first 5 rows):")
# print(final_data.head())

base_df = pd.DataFrame(columns=["Price", "Rooms", "BuiltYear"])
cand_df = pd.DataFrame(columns=["Apple", "Banana", "Giraffe"])
similarity = compute_semantic_similarity(base_df, cand_df)
assert similarity < 0.1, f"Unexpectedly high similarity: {similarity}"
print("Semantic similarity for unrelated datasets is low:", similarity)

base_df = pd.DataFrame(columns=["Price", "Rooms", "BuiltYear"])
cand_df = pd.DataFrame(columns=["Price", "Rooms", "BuiltYear"])
similarity = compute_semantic_similarity(base_df, cand_df)
assert similarity == 1.0, f"Unexpected similarity: {similarity}"
print("Semantic similarity for identical datasets is 1.0:", similarity)
