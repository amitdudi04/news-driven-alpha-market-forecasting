"""Run the saved-data research experiment.

The command never calls external APIs. It rebuilds the feature set from the
committed market/news inputs and only proceeds to model evaluation when the
clean sample is large enough for the public walk-forward design.
"""

import logging

import module4_features
import module5_xgboost
import module6_garch
import module7_backtesting

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MIN_PUBLIC_SAMPLE = 61


def main():
    module4_features.main()

    labelled = module5_xgboost.load_data()
    if len(labelled) < MIN_PUBLIC_SAMPLE:
        logging.warning(
            "Feature dataset built, but only %s labelled rows are available. "
            "At least %s are required (60 training rows plus one OOS row) before publishing the OOS model comparison "
            "and GARCH-scaled backtest.",
            len(labelled),
            MIN_PUBLIC_SAMPLE,
        )
        return

    module5_xgboost.main()
    module6_garch.main()
    module7_backtesting.main()


if __name__ == "__main__":
    main()
