"""Optional daily research refresh for the News-Driven Alpha project.

The daily path refreshes external data, updates sentiment and features, creates
an updated GARCH volatility forecast, runs the canonical XGBoost artifact, and
writes a paper-trading signal. It is a research workflow, not a live-capital
execution system.
"""

import logging
import uuid

import module1_news
import module2_sentiment
import module3_market
import module4_features
import module6_garch
import module12_inference
import module13_signal_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    execution_uuid = str(uuid.uuid4())
    logging.info(
        "Starting News-Driven Alpha daily research refresh: %s",
        execution_uuid,
    )

    module3_market.main(execution_uuid=execution_uuid)
    module1_news.main(execution_uuid=execution_uuid)
    module2_sentiment.main(execution_uuid=execution_uuid)
    module4_features.main(execution_uuid=execution_uuid)
    module6_garch.main()
    module12_inference.main(execution_uuid=execution_uuid)
    module13_signal_engine.main(execution_uuid=execution_uuid)

    logging.info("Daily research refresh completed")


if __name__ == "__main__":
    main()
