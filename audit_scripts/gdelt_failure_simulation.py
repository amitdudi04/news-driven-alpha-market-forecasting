import sys
import os
import logging
import pandas as pd
from unittest import mock

# Add parent directory to path to import module1
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import module1_news_extraction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_empty_response():
    logging.info("--- SIMULATING: Empty Response ---")
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"articles": []}
    mock_response.content = b'{"articles": []}'
    
    with mock.patch('requests.get', return_value=mock_response):
        try:
            module1_news_extraction.get_latest_news("test")
            logging.error("FAIL: Silent empty return permitted.")
            sys.exit(1)
        except RuntimeError as e:
            if "Zero articles extracted" in str(e):
                logging.info(f"PASS: Caught empty return -> {e}")
            else:
                logging.error(f"FAIL: Wrong error raised -> {e}")
                sys.exit(1)

def test_malformed_json():
    logging.info("--- SIMULATING: Malformed JSON ---")
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = b'BAD JSON'
    
    # Python json decoder error mock
    import json
    def mock_json():
        raise json.JSONDecodeError("Expecting value", "", 0)
        
    mock_response.json = mock_json
    
    with mock.patch('requests.get', return_value=mock_response):
        try:
            module1_news_extraction.get_latest_news("test")
            logging.error("FAIL: Malformed JSON parsed.")
            sys.exit(1)
        except RuntimeError as e:
            if "Malformed JSON" in str(e):
                logging.info(f"PASS: Caught Malformed JSON -> {e}")
            else:
                logging.error(f"FAIL: Wrong error raised -> {e}")
                sys.exit(1)

def test_schema_drift():
    logging.info("--- SIMULATING: Schema Drift ---")
    mock_response = mock.Mock()
    mock_response.status_code = 200
    # Missing 'domain'
    mock_response.json.return_value = {"articles": [{"seendate": "20260101T000000Z", "title": "Test"}]}
    mock_response.content = b'{}'
    
    with mock.patch('requests.get', return_value=mock_response):
        try:
            module1_news_extraction.get_latest_news("test")
            logging.error("FAIL: Schema drift permitted.")
            sys.exit(1)
        except RuntimeError as e:
            if "Schema Drift Detected" in str(e):
                logging.info(f"PASS: Caught Schema Drift -> {e}")
            else:
                logging.error(f"FAIL: Wrong error raised -> {e}")
                sys.exit(1)

def main():
    logging.info("Executing GDELT Failure Simulation Suite...")
    test_empty_response()
    test_malformed_json()
    test_schema_drift()
    logging.info("ALL SIMULATIONS PASSED. SAFE MODE ESCALATION IS SECURE.")

if __name__ == "__main__":
    main()
