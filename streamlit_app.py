def main():
    st.title("METAM: Goal-Oriented Data Augmentation UI")

    st.sidebar.header("Select Dataset")
    dataset_options = {
        "Seattle Airbnb & Augmentations": {
            "base": "data/seattle_airbnb.csv",
            "candidates": [
                "data/seattle_crime.csv",
                "data/seattle_demographics.csv",
                "data/seattle_housing.csv",
                "data/seattle_amenities.csv",
                "data/seattle_transportation.csv"
            ]
        }
        # You can add more dataset options here.
    }

    selected_dataset = st.sidebar.selectbox("Choose Dataset", list(dataset_options.keys()))
    ds_info = dataset_options[selected_dataset]

    # Load base dataset
    try:
        base_df = pd.read_csv(ds_info["base"])
    except Exception as e:
        st.error(f"Error loading base dataset: {e}")
        return

    # Use selectbox for join and target columns based on actual base_df columns.
    st.sidebar.header("Base Dataset Columns")
    base_join = st.sidebar.selectbox("Select Base Join Column", options=list(base_df.columns))
    target_col = st.sidebar.selectbox("Select Target Column", options=list(base_df.columns))

    st.sidebar.header("Parameters")
    theta = st.sidebar.slider("Utility Threshold (theta)", 0.0, 1.0, 0.8)
    epsilon = st.sidebar.slider("Clustering Epsilon", 0.01, 0.1, 0.05)

    st.sidebar.header("Candidate Augmentations")
    candidate_files = ds_info["candidates"]
    selected_candidates = []
    for file in candidate_files:
        if st.sidebar.checkbox(f"Include {file.split('/')[-1]}", value=True):
            selected_candidates.append(file)

    if st.sidebar.button("Run METAM"):
        if selected_candidates:
            candidates = []
            for file in selected_candidates:
                try:
                    df_candidate = pd.read_csv(file)
                    # For simplicity, we assume the candidate join column is the same as the base join.
                    candidates.append((df_candidate, base_join))
                except Exception as e:
                    st.error(f"Error loading candidate {file}: {e}")

            start_time = time.time()
            metam = METAM(
                base_data=base_df,
                base_target_col=target_col,
                candidate_datasets=candidates,
                join_on=base_join,
                theta=theta,
                epsilon=epsilon
            )
            final_data, final_util, selected_aug, iter_utils = metam.run_metam()
            running_time = time.time() - start_time

            st.subheader("Results")
            st.write("**Final Utility:**", final_util)
            st.write("**Running Time (seconds):**", running_time)
            st.write("**Selected Augmentations (join keys):**", [cand["join_on"] for cand in selected_aug])
            st.write("**Resulting Data (first 10 rows):**")
            st.dataframe(final_data.head(10))

            st.subheader("Utility Improvement Over Iterations")
            fig, ax = plt.subplots()
            ax.plot(iter_utils, marker='o')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Utility")
            ax.set_title("Utility Improvement")
            st.pyplot(fig)
        else:
            st.error("Please select at least one candidate augmentation.")


if __name__ == '__main__':
    main()