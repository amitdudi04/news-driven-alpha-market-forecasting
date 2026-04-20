import os
import sys
import logging
import pandas as pd
import joblib

# Import pipeline modules
import module1_news
import module2_sentiment
import module3_market
import module4_features
import module12_inference
import module13_signal_engine

# ==========================================
# MASTER SCRIPT: DAILY ORCHESTRATION PIPELINE
# ==========================================
# Objective: Execute the full data-to-prediction pipeline sequentially 
# and generate the institutional trading signal for the next market open.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline():
    logging.info("==================================================")
    logging.info("       NEWS-DRIVEN ALPHA: DAILY ORCHESTRATION     ")
    logging.info("==================================================")
    
    # Step 1: News Data
    logging.info("\n--- STEP 1: Fetching Latest News (GDELT) ---")
    try:
        module1_news.main()
    except Exception as e:
        logging.error(f"Module 1 Failed: {e}")
        return False
        
    # Step 2: Sentiment
    logging.info("\n--- STEP 2: Generating Sentiment Signals (FinBERT) ---")
    try:
        module2_sentiment.main()
    except Exception as e:
        logging.error(f"Module 2 Failed: {e}")
        return False
        
    # Step 3: Market Data
    logging.info("\n--- STEP 3: Updating Market Data (yfinance) ---")
    try:
        module3_market.main()
    except Exception as e:
        logging.error(f"Module 3 Failed: {e}")
        return False
        
    # Step 4: Feature Engineering
    logging.info("\n--- STEP 4: Engineering Features & Aligning Time-Series ---")
    try:
        module4_features.main()
    except Exception as e:
        logging.error(f"Module 4 Failed: {e}")
        return False
        
    return True
        
def generate_predictions_and_signals():
    logging.info("\n--- STEP 5: Executing Live Inference Engine ---")
    try:
        module12_inference.main()
    except Exception as e:
        logging.error(f"Inference Engine Failed: {e}")
        return False
        
    logging.info("\n--- STEP 6: Generating Trading Signal & Logging ---")
    try:
        module13_signal_engine.main()
    except Exception as e:
        logging.error(f"Signal Engine Failed: {e}")
        return False
        
    return True

def main():
    success = run_pipeline()
    if success:
        generate_predictions_and_signals()
    else:
        logging.error("Pipeline failed. Prediction sequence aborted.")

if __name__ == "__main__":
    main()
