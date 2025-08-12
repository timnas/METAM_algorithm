METAM: Goal-Oriented Data Discovery

Streamlit UI link:
https://metam-algorithm-analysis.streamlit.app/

Original paper link: https://arxiv.org/pdf/2304.09068

METAM is a framework for selecting a minimal, diverse subset of external datasets that improve model performance.
METAM computes lightweight profile vectors (e.g. overlap, feature count, semantic similarity), clusters similar candidates, and iteratively evaluates the most promising representatives based on actual utility gain, pruning redundant datasets at the end.

METAM algorithm implementation in file "streamlit_app.py"
