# src/chatbot_eval_pkg/loader.py
import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_test_cases_from_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Loads all JSON test case files from a specified directory.

    Args:
        directory (str): The path to the directory containing JSON test files.

    Returns:
        List[Dict[str, Any]]: A list of loaded test cases (as dictionaries).
                               Returns an empty list if the directory doesn't exist
                               or contains no valid JSON files.
    """
    test_cases = []
    if not os.path.isdir(directory):
        logger.error(f"Test data directory not found: {directory}")
        return test_cases

    logger.info(f"Loading test cases from directory: {directory}")
    for filename in os.listdir(directory):
        if filename.lower().endswith(".json"):
            file_path = os.path.join(directory, filename)
            logger.debug(f"Attempting to load test case file: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Basic validation (F1.1, F1.2, F1.3)
                    if "id" not in data:
                        logger.warning(f"Skipping file {filename}: Missing required field 'id'.")
                        continue
                    if "input" not in data and "messages" not in data:
                         logger.warning(f"Skipping file {filename} (id: {data.get('id')}): Missing required field 'input' or 'messages'.")
                         continue
                    if "metrics" not in data or not isinstance(data["metrics"], list):
                         logger.warning(f"Skipping file {filename} (id: {data.get('id')}): Missing or invalid 'metrics' field (must be a list).")
                         continue

                    # Add file path for reference during testing/reporting
                    data['_file_path'] = file_path
                    test_cases.append(data)
                    logger.debug(f"Successfully loaded test case '{data.get('id')}' from {filename}")

            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from file {file_path}: {e}")
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")

    logger.info(f"Loaded {len(test_cases)} test cases.")
    return test_cases

def load_test_case_from_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads a single JSON test case file.

    Args:
        file_path (str): The path to the JSON test file.

    Returns:
        Optional[Dict[str, Any]]: The loaded test case (as a dictionary) or None if loading fails.
    """
    logger.debug(f"Attempting to load single test case file: {file_path}")
    if not os.path.isfile(file_path):
        logger.error(f"Test case file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Basic validation (F1.1, F1.2, F1.3)
            if "id" not in data:
                logger.warning(f"Skipping file {file_path}: Missing required field 'id'.")
                return None
            if "input" not in data and "messages" not in data:
                 logger.warning(f"Skipping file {file_path} (id: {data.get('id')}): Missing required field 'input' or 'messages'.")
                 return None
            if "metrics" not in data or not isinstance(data["metrics"], list):
                 logger.warning(f"Skipping file {file_path} (id: {data.get('id')}): Missing or invalid 'metrics' field (must be a list).")
                 return None

            # Add file path for reference
            data['_file_path'] = file_path
            logger.debug(f"Successfully loaded test case '{data.get('id')}' from {file_path}")
            return data

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return None
