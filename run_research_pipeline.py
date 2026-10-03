"""Run the reproducible, saved-data research experiment.

This command does not call live APIs. It rebuilds features from the committed
sentiment/market inputs, creates time-safe OOS XGBoost predictions, produces
one-step-ahead GARCH forecasts, and evaluates the paper strategy.
"""

import module4_features
import module5_xgboost
import module6_garch
import module7_backtesting


def main():
    module4_features.main()
    module5_xgboost.main()
    module6_garch.main()
    module7_backtesting.main()


if __name__ == "__main__":
    main()
