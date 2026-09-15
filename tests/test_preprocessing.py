from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from case_study_2.preprocessing import (
    apply_imputer,
    fit_imputer,
    make_features,
    preprocess,
    read_last_rows,
)


class PreprocessingTests(unittest.TestCase):
    def test_rate_stays_inside_each_vehicle(self) -> None:
        history = pd.DataFrame(
            {
                "vehicle_id": [2, 1, 2, 1],
                "time_step": [2.0, 1.0, 1.0, 2.0],
                "171_0": [105.0, 1.0, 100.0, 3.0],
            }
        )
        ids, features = make_features(history)

        self.assertEqual(ids["vehicle_id"].tolist(), [1, 2])
        np.testing.assert_allclose(features["rate__171_0"], [2.0, 5.0])

    def test_constant_features_are_not_created(self) -> None:
        history = pd.DataFrame(
            {
                "vehicle_id": [1, 1],
                "time_step": [1.0, 2.0],
                "171_0": [1.0, 3.0],
            }
        )
        _, features = make_features(history)

        self.assertNotIn("context__has_previous", features)
        self.assertNotIn("quality__negative_delta_count", features)

    def test_duplicate_time_is_rejected(self) -> None:
        history = pd.DataFrame(
            {
                "vehicle_id": [1, 1],
                "time_step": [1.0, 1.0],
                "171_0": [1.0, 2.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "dupliqué"):
            make_features(history)

    def test_reader_handles_chunk_boundaries(self) -> None:
        frame = pd.DataFrame(
            {
                "vehicle_id": [1, 1, 1, 2, 2, 3],
                "time_step": [1, 2, 3, 1, 2, 1],
                "171_0": [1, 2, 3, 10, 20, 100],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            frame.to_csv(path, index=False)
            result = read_last_rows(path, n=2, chunksize=2)

        expected = pd.DataFrame(
            {
                "vehicle_id": [1, 1, 2, 2, 3],
                "time_step": [2, 3, 1, 2, 1],
                "171_0": [2, 3, 10, 20, 100],
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_median_is_learned_on_train_only(self) -> None:
        train = pd.DataFrame({"feature": [1.0, 3.0, np.nan]})
        validation = pd.DataFrame({"feature": [100.0, np.nan]})

        imputer = fit_imputer(train)
        transformed = apply_imputer(imputer, validation)

        self.assertEqual(transformed.loc[1, "feature"], 2.0)
        self.assertEqual(transformed.loc[1, "missingindicator_feature"], 1.0)

    def test_preprocess_exports_csv_files(self) -> None:
        history = pd.DataFrame(
            {
                "vehicle_id": [1, 1, 2, 2],
                "time_step": [1.0, 2.0, 1.0, 2.0],
                "171_0": [1.0, 3.0, 10.0, 14.0],
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            output_dir = root / "preprocessed"
            data_dir.mkdir()
            for split in ("train", "validation", "test"):
                history.to_csv(
                    data_dir / f"{split}_operational_readouts.csv", index=False
                )

            preprocess(data_dir, output_dir)

            for split in ("train", "validation", "test"):
                self.assertTrue((output_dir / f"{split}_features.csv").exists())
                self.assertFalse((output_dir / f"{split}_features.parquet").exists())


if __name__ == "__main__":
    unittest.main()
