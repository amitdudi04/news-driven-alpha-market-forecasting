"""Run the saved-data research experiment.

The command never calls external APIs. It rebuilds descriptive evidence and
model features from the committed inputs. Predictive-model performance is only
reported after the pre-specified out-of-sample reporting threshold is reached.
"""

import logging

import descriptive_analysis
import module4_features
import module5_xgboost
import module6_garch
import module7_backtesting
from config.research_config import (
    MIN_DIRECTION_TRAIN_OBSERVATIONS,
    MIN_OOS_REPORTING_OBSERVATIONS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    descriptive_analysis.main()
    module4_features.main()

    labelled = module5_xgboost.load_data()
    minimum_public_sample = (
        MIN_DIRECTION_TRAIN_OBSERVATIONS + MIN_OOS_REPORTING_OBSERVATIONS
    )
    if len(labelled) < minimum_public_sample:
        logging.warning(
            "Descriptive results and feature data were rebuilt, but only %s "
            "labelled model rows are available. The directional model requires "
            "%s initial training rows, and this repository requires at least %s "
            "genuine OOS forecasts before model-performance statistics are "
            "reported (minimum %s labelled rows in total).",
            len(labelled),
            MIN_DIRECTION_TRAIN_OBSERVATIONS,
            MIN_OOS_REPORTING_OBSERVATIONS,
            minimum_public_sample,
        )
        return

    module5_xgboost.main()
    module6_garch.main()
    module7_backtesting.main()


if __name__ == "__main__":
    main()
