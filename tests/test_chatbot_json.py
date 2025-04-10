# tests/test_chatbot_json.py
import pytest
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Import from the new package structure
from chatbot_eval_pkg.loader import load_test_cases_from_directory
from chatbot_eval_pkg.client import get_chatbot_client, BaseChatbotClient
from chatbot_eval_pkg.evaluator import evaluate_test_case
from chatbot_eval_pkg.types import EvaluationResult, MetricResultDetail # Import result types

logger = logging.getLogger(__name__)

# --- Configuration ---
# Get configuration from environment variables (ensure these are set)
TEST_DATA_DIR = os.getenv("TEST_DATA_DIR", "test_data") # Default to 'test_data' relative to root
CHATBOT_API_ENDPOINT = os.getenv("CHATBOT_API_ENDPOINT")
CHATBOT_API_KEY = os.getenv("CHATBOT_API_KEY")
# Optional: Configure evaluation model via env var
DEEPEVAL_EVALUATION_MODEL = os.getenv("DEEPEVAL_EVALUATION_MODEL")
# Optional: Configure async mode via env var (convert string to bool)
DEEPEVAL_RUN_ASYNC_STR = os.getenv("DEEPEVAL_RUN_ASYNC", "true").lower()
DEEPEVAL_RUN_ASYNC = DEEPEVAL_RUN_ASYNC_STR in ['true', '1', 't', 'y', 'yes']


# --- Test Setup ---

# Load test cases once for the entire test session
logger.info(f"Pytest: Loading test cases from directory: {TEST_DATA_DIR}")
if not os.path.isdir(TEST_DATA_DIR):
     logger.warning(f"Test data directory '{TEST_DATA_DIR}' not found. Skipping tests.")
     ALL_TEST_CASES = []
     TEST_CASE_IDS = []
else:
    ALL_TEST_CASES = load_test_cases_from_directory(TEST_DATA_DIR)
    logger.info(f"Pytest: Found {len(ALL_TEST_CASES)} test cases.")
    # Create IDs for pytest parametrization
    TEST_CASE_IDS = [
        f"{os.path.basename(tc.get('_file_path', 'unknown'))}[{tc.get('id', 'no_id')}]"
        for tc in ALL_TEST_CASES
    ]

# Fixture to provide a ChatbotClient instance
@pytest.fixture(scope="session")
def chatbot_client_session() -> BaseChatbotClient:
    """Pytest fixture to create and tear down a ChatbotClient session."""
    logger.info("Pytest: Setting up ChatbotClient for session...")
    if not CHATBOT_API_ENDPOINT:
        pytest.skip("CHATBOT_API_ENDPOINT environment variable not set. Skipping tests that require client.")

    # Use the factory function with config from environment variables
    client_config = {
        "type": "http", # Assuming HTTP client for now
        "api_endpoint": CHATBOT_API_ENDPOINT,
        "api_key": CHATBOT_API_KEY
    }
    client = get_chatbot_client(client_config)
    yield client # Provide the client to the tests
    logger.info("Pytest: Tearing down ChatbotClient session...")
    client.close_session()


# --- Test Function ---

# Skip all tests in this file if no test cases were loaded
pytestmark = pytest.mark.skipif(not ALL_TEST_CASES, reason=f"No test cases found in {TEST_DATA_DIR}")

@pytest.mark.parametrize("test_case_data", ALL_TEST_CASES, ids=TEST_CASE_IDS)
def test_chatbot_evaluation(test_case_data: dict, chatbot_client_session: BaseChatbotClient):
    """
    Runs a single chatbot evaluation test case using pytest.
    """
    test_id = test_case_data.get("id", "unknown_id")
    file_path = test_case_data.get("_file_path", "unknown_file")
    logger.info(f"Pytest: Running evaluation for test case ID: {test_id} from file: {os.path.basename(file_path)}")

    # Perform the evaluation using the refactored evaluator function
    # Pass evaluation model config and async setting
    result: EvaluationResult = evaluate_test_case(
        test_case_data=test_case_data,
        chatbot_client=chatbot_client_session,
        global_model_config=DEEPEVAL_EVALUATION_MODEL, # Can be None
        run_async=DEEPEVAL_RUN_ASYNC
    )

    # --- Assertion for Pytest Pass/Fail ---
    # Assert that the evaluation was successful using the EvaluationResult dataclass.
    error_msg = result.error or "No error message provided."
    metrics_summary = "\nMetrics Results:\n"
    for mr in result.metrics_results:
        score_str = f"{mr.score:.4f}" if mr.score is not None else "N/A"
        metrics_summary += (
            f"  - Metric: {mr.metric}, "
            f"Score: {score_str}, "
            f"Threshold: {mr.threshold}, "
            f"Success: {mr.success}, "
            f"Reason: {mr.reason}\n"
        )

    assert result.success, \
        f"Test Case Failed: ID='{result.id}', File='{os.path.basename(result.file_path or 'N/A')}'\n" \
        f"Chatbot Response: {result.chatbot_response}\n" \
        f"Failure Reason: {error_msg}\n" \
        f"{metrics_summary}"

    logger.info(f"Pytest: Finished evaluation for test case ID: {result.id}. Success: {result.success}")

# Instructions for running (update these if needed):
# 1. Ensure required environment variables are set (e.g., in a .env file):
#    CHATBOT_API_ENDPOINT, CHATBOT_API_KEY (optional), TEST_DATA_DIR (optional)
#    OPENAI_API_KEY (if using default deepeval models)
# 2. Install dependencies: pip install -e ".[test]" (or uv pip install -e ".[test]")
# 3. Navigate to the project root directory.
# 4. Run pytest: pytest tests/
# 5. For HTML report: pytest tests/ --html=report.html --self-contained-html
