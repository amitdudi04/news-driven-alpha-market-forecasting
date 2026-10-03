"""Research configuration for the News-Driven Alpha project.

These settings define the public research experiment and simulated-strategy rule.
They are not live-capital limits.
"""

LONG_PROBABILITY_THRESHOLD = 0.55
SHORT_PROBABILITY_THRESHOLD = 0.45
MAX_ABS_POSITION = 1.0
TRANSACTION_COST = 0.001  # 10 bps per unit of turnover

MIN_DIRECTION_TRAIN_OBSERVATIONS = 60
MIN_OOS_REPORTING_OBSERVATIONS = 30
MIN_GARCH_TRAIN_OBSERVATIONS = 60

RANDOM_STATE = 42
