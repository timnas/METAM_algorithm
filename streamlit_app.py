def main():
    st.title("METAM: Goal-Oriented Data Augmentation")

    st.sidebar.header("Upload Datasets")
    base_file = st.sidebar.file_uploader("Upload Base Dataset (CSV)", type=["csv"])
    candidate_files = st.sidebar.file_uploader("Upload Candidate Datasets (CSV)", type=["csv"],
                                               accept_multiple_files=True)

    st.sidebar.header("Parameters")
    base_join = st.sidebar.text_input("Base Dataset Join Column", value="Id")
    candidate_join = st.sidebar.text_input("Candidate Join Column", value="Id")
    target_col = st.sidebar.text_input("Target Column", value="expensive")
    theta = st.sidebar.slider("Utility Threshold (theta)", 0.0, 1.0, 0.8)
    epsilon = st.sidebar.slider("Clustering Epsilon", 0.01, 0.1, 0.05)

    if st.sidebar.button("Run METAM"):
        if base_file is not None and candidate_files:
            base_df = pd.read_csv(base_file)
            candidates = []
            for file in candidate_files:
                df_candidate = pd.read_csv(file)
                # Each candidate tuple is (df, candidate join column)
                candidates.append((df_candidate, candidate_join))

            start_time = time.time()
            metam = METAM(
                base_data=base_df,
                base_target_col=target_col,
                candidate_datasets=candidates,
                join_on=base_join,
                theta=theta,
                epsilon=epsilon
            )
            final_data, final_util, selected = metam.run_metam()
            running_time = time.time() - start_time

            st.write("### Final Utility:", final_util)
            st.write("### Running Time (seconds):", running_time)
            st.write("### Selected Augmentations (join keys):", [cand["join_on"] for cand in selected])
            st.write("### Resulting Data (first 10 rows):")
            st.dataframe(final_data.head(10))
        else:
            st.error("Please upload the base dataset and at least one candidate dataset.")


if __name__ == '__main__':
    main()