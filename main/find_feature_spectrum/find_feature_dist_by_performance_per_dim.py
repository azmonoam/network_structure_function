from typing import Optional

import pandas as pd

from find_feature_spectrum.find_feature_dist import FindFeaturesDist


class FindFeaturesDistByPerformancePerDim(FindFeaturesDist):
    def __init__(self,
                 all_features_values,
                 num_features: int,
                 samples_path: Optional[str] = None,
                 ):
        self.all_features_values = all_features_values
        super().__init__(num_features, samples_path)

    def _get_samples_df(
            self,
    ) -> pd.DataFrame:
        return pd.DataFrame(self.all_features_values).astype(float)

    @staticmethod
    def _get_all_archs_data(
            samples_path: str,
    ):
        pass
