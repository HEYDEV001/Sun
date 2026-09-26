import pandas as pd

def write_submission(df, path, id_col, list_col):
    """df: one row per S1 entity; list_col holds a python list of IDs."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{id_col}\t{list_col}\n")
        for _, row in df.iterrows():
            ids = ",".join(dict.fromkeys(row[list_col]))  # dedupe, keep order
            f.write(f"{row[id_col]}\t{ids}\n")


test_s1 = pd.read_csv("../student_resource/dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id"])
# dummy: empty match lists for everyone — must still PASS validation
dummy = pd.DataFrame({
    "source1_entity_id": test_s1["entity_id"],
    "matched": [[] for _ in range(len(test_s1))],
})
cand  = pd.DataFrame({
    "source1_entity_id": test_s1["entity_id"],
    "candidates": [[] for _ in range(len(test_s1))],
})

write_submission(dummy, "output/matching_results.tsv", "source1_entity_id", "matched")
write_submission(cand,  "output/candidate_pairs.tsv", "source1_entity_id", "candidates")