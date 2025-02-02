import streamlit as st
import pandas as pd
from metam import METAM


def main():
    st.title("METAM: Goal-Oriented Data Discovery")

    st.sidebar.header("Upload Base Dataset")
    base_file = st.sidebar.file_uploader("Base CSV", type=["csv"])

    st.sidebar.header("Upload Candidate Datasets")
    cand_files = st.sidebar.file_uploader("Candidates (multiple)", type=["csv"], accept_multiple_files=True)

    if base_file and cand_files:
        base_df = pd.read_csv(base_file)

        # Let user specify target col
        target_col = st.text_input("Target Column Name", "target")

        # Let user specify join col
        join_col = st.text_input("Join Column in Base", "id")

        # Collect candidate info
        candidate_list = []
        for i, f in enumerate(cand_files):
            cand_df = pd.read_csv(f)
            candidate_join_col = st.text_input(f"Join Column for Candidate {f.name}", "id")
            candidate_list.append((cand_df, candidate_join_col))

        if st.button("Run METAM"):
            metam = METAM(base_data=base_df,
                          base_target_col=target_col,
                          candidate_datasets=candidate_list,
                          join_on=join_col)
            final_data, final_perf, chosen_augs = metam.run_metam()
            st.write("**Final Performance**:", final_perf)
            st.write("**Chosen Augmentations**:", [c[1] for c in chosen_augs])
            st.dataframe(final_data.head(20))


if __name__ == "__main__":
    main()
