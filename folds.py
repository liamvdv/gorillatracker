from gorillatracker.utils.embedding_generator import generate_embeddings_from_run, read_embeddings_from_disk
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
import math
import numpy as np
import pandas as pd

"""
We ignore all individuals with less then 2 annotations. [Why?]


We split all the images of individuals we have into 5 partitions (called folds).
Each partition has N_PERCENT images of individuals only known in that partition.

We grid search over the thresholds.
    For every partition, we use 4 partitions as a database and the remaining one as a query set.
    We mask the parition distinct labels as 'new' and calculate the mAP of the query set. This means we calculate the annotations of the query set using the database and a threshold.
    We then calculate three mAP scores of 
    - the 'new' annotations only,
    - the existing annotations only and
    - then of all annotations.
"""


def chunk(arr, n):
    return [arr[i:i+n] for i in range(0, len(arr), n)]

def stripe(arr, n):
    return [arr[i::n] for i in range(n)]

class DistinctClassFold:
    def __init__(self, n_buckets=5, shuffle=True, random_state=None):
        self.n_buckets = n_buckets
        self.shuffle = shuffle
        self.random = np.random.RandomState(random_state)

    def split(self, df: pd.DataFrame, label_column: str = "label_string", label_mask="new", unique_percentage=0.2):
        # Shuffle DF
        shuffled = df.sample(frac=1, random_state=self.random).reset_index(drop=True) if self.shuffle else df

        # labels must be shuffeled
        unique_labels = shuffled[label_column].unique()
        n_unique_split = int(len(unique_labels) * unique_percentage)
        new_labels = unique_labels[:n_unique_split]
        new_labels_per_bucket = stripe(new_labels, self.n_buckets)
        assert len(new_labels_per_bucket) > 0, "Not enough unique labels"

        # now fill all buckets with images of the unique labels,
        buckets = [None] * self.n_buckets
        rest = shuffled[~shuffled[label_column].isin(new_labels)]
        n_rest_per_bucket = math.ceil(len(rest) / self.n_buckets)
        rest_per_bucket = chunk(rest, n_rest_per_bucket)
        for i in range(self.n_buckets):
            new_per_bucket = shuffled[shuffled[label_column].isin(new_labels_per_bucket[i])]
            bucket = pd.concat(
                [
                    new_per_bucket,
                    rest_per_bucket[i],
                ]
            )
            buckets[i] = bucket

        # Yields folds as (train, test) dataframes
        for i in range(self.n_buckets):
            # overwrite label with label_string values for unique labels in the parititon with mask_label
            test = buckets[i].copy()
            test.loc[test[label_column].isin(new_labels_per_bucket[i]), label_column] = label_mask

            train = pd.concat(buckets[:i] + buckets[i + 1 :])
            assert train.shape[0] + test.shape[0] == df.shape[0], f"{train.shape[0]} + {test.shape[0]} != {df.shape[0]}"
            yield train, test


def test_distinct_class_fold():
    folder = DistinctClassFold()
    # Check logic
    df = read_embeddings_from_disk("example-embeddings.pkl")
    for database, query_set in folder.split(df, label_mask="new"):
        print(database.shape, query_set.shape)
        assert "new" not in database["label_string"].unique()
        assert "new" in query_set["label_string"].unique()

    # Check Deterministic
    determistic = DistinctClassFold(random_state=42)
    determistic2 = DistinctClassFold(random_state=42)
    for (database1, query_set1), (database2, query_set2) in zip(determistic.split(df, label_mask="new"), determistic2.split(df, label_mask="new")):
        assert database2.equals(database1)
        assert query_set2.equals(query_set1)

if __name__ == "__main__":
    test_distinct_class_fold()
