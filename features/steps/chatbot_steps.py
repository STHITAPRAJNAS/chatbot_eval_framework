# features/steps/chatbot_steps.py
from behave import *
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Import from the new package structure
from chatbot_eval_pkg.loader import load_test_case_from_file
from chatbot_eval_pkg.client import get_chatbot_client, BaseChatbotClient
from chatbot_eval_pkg.evaluator import evaluate_test_case
from chatbot_eval_pkg.types import EvaluationResult # Import result type

logger = logging.getLogger(__name__)

# --- Configuration (Read from Environment) ---
# It's generally better to handle client setup/teardown and config
# in environment.py, but we'll read basic config here for simplicity.
CHATBOT_API_ENDPOINT = os.getenv("CHATBOT_API_ENDPOINT")
CHATBOT_API_KEY = os.getenv("CHATBOT_API_KEY")
DEEPEVAL_EVALUATION_MODEL = os.getenv("DEEPEVAL_EVALUATION_MODEL")
DEEPEVAL_RUN_ASYNC_STR = os.getenv("DEEPEVAL_RUN_ASYNC", "true").lower()
DEEPEVAL_RUN_ASYNC = DEEPEVAL_RUN_ASYNC_STR in ['true', '1', 't', 'y', 'yes']

# Use behave's context object to store data between steps
# and manage resources like the chatbot client.

@given('a chatbot evaluation test case defined in "{file_path}"')
def step_impl_load_test_case(context, file_path):
    """
    Loads the specified JSON test case file.
    """
    logger.info(f"Behave: Loading test case from: {file_path}")
    # Assuming behave is run from the project root where test_data exists
    full_path = os.path.abspath(file_path)
    context.test_case_data = load_test_case_from_file(full_path)
    assert context.test_case_data is not None, f"Failed to load test case file: {full_path}"
    context.test_case_id = context.test_case_data.get("id", os.path.basename(file_path))
    logger.info(f"Behave: Loaded test case ID: {context.test_case_id}")

@when('the evaluation is performed for this test case')
def step_impl_perform_evaluation(context):
    """
    Initializes the client (if not already done via environment.py) and runs the evaluation.
    """
    assert hasattr(context, 'test_case_data'), "Test case data not loaded in context."

    # --- Client Initialization ---
    # Best Practice: Move client setup/teardown to features/environment.py
    # This ensures the client is created once per feature/scenario and cleaned up properly.
    # If using environment.py, this block would likely be removed or simplified.
    if not hasattr(context, 'chatbot_client'):
        logger.warning("Behave: ChatbotClient not found in context. Initializing here (consider using environment.py).")
        if not CHATBOT_API_ENDPOINT:
             raise ValueError("CHATBOT_API_ENDPOINT environment variable not set. Cannot initialize client.")

        client_config = {
            "type": "http",
            "api_endpoint": CHATBOT_API_ENDPOINT,
            "api_key": CHATBOT_API_KEY
        }
        context.chatbot_client = get_chatbot_client(client_config)
        # Add a cleanup function to context if not using environment.py
        def cleanup_client():
             if hasattr(context, 'chatbot_client') and context.chatbot_client:
                 logger.info("Behave: Cleaning up ChatbotClient from step context...")
                 context.chatbot_client.close_session()
        context.add_cleanup(cleanup_client)


    # --- Perform Evaluation ---
    logger.info(f"Behave: Performing evaluation for test case ID: {context.test_case_id}")
    # Ensure the client exists on the context before proceeding
    if not hasattr(context, 'chatbot_client') or not context.chatbot_client:
         raise RuntimeError("Chatbot client is not available on the context.")

    # Call the refactored evaluator
    context.evaluation_result = evaluate_test_case(
        test_case_data=context.test_case_data,
        chatbot_client=context.chatbot_client,
        global_model_config=DEEPEVAL_EVALUATION_MODEL,
        run_async=DEEPEVAL_RUN_ASYNC
    )
    assert isinstance(context.evaluation_result, EvaluationResult), "Evaluation function did not return an EvaluationResult object."
    logger.info(f"Behave: Evaluation finished for test case ID: {context.test_case_id}")


@then('the evaluation result should be successful')
def step_impl_check_success(context):
    """
    Asserts that the evaluation result indicates success using the EvaluationResult object.
    """
    assert hasattr(context, 'evaluation_result'), "Evaluation result not found in context."
    result: EvaluationResult = context.evaluation_result # Type hint for clarity

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
        f"Behave Scenario Failed: Test Case ID='{result.id}'\n" \
        f"Failure Reason: {error_msg}\n" \
        f"{metrics_summary}"

    logger.info(f"Behave: Test case ID: {result.id} - PASSED")

# --- Recommended: features/environment.py ---
# Create a file named `environment.py` inside the `features/` directory
# for robust setup and teardown.
#
# Example (features/environment.py):
#
# import os
# import logging
# from dotenv import load_dotenv
# from chatbot_eval_pkg.client import get_chatbot_client, BaseChatbotClient
#
# logger = logging.getLogger(__name__)
# load_dotenv()
#
# def before_all(context):
#     # Runs once before all features
#     context.config.setup_logging() # Optional: configure behave logging
#     logger.info("Behave: Global setup (before_all)")
#     # Read config needed for client setup
#     context.chatbot_api_endpoint = os.getenv("CHATBOT_API_ENDPOINT")
#     context.chatbot_api_key = os.getenv("CHATBOT_API_KEY")
#
# def before_feature(context, feature):
#     # Runs before each feature file
#     logger.info(f"Behave: Setting up client for Feature: {feature.name}")
#     if not context.chatbot_api_endpoint:
#         logger.error("CHATBOT_API_ENDPOINT not set. Skipping feature.")
#         feature.skip("CHATBOT_API_ENDPOINT environment variable not set.")
#         return
#
#     client_config = {
#         "type": "http",
#         "api_endpoint": context.chatbot_api_endpoint,
#         "api_key": context.chatbot_api_key
#     }
#     context.chatbot_client = get_chatbot_client(client_config)
#
# def after_feature(context, feature):
#     # Runs after each feature file
#     if hasattr(context, 'chatbot_client') and context.chatbot_client:
#         logger.info(f"Behave: Tearing down client after Feature: {feature.name}")
#         context.chatbot_client.close_session()
#         delattr(context, 'chatbot_client') # Clean up context
#
# # You can also use before_scenario/after_scenario for finer control

# To run these tests:
# 1. Ensure required environment variables are set (e.g., in a .env file).
# 2. Install dependencies: pip install -e ".[test]" (or uv pip install -e ".[test]")
# 3. Navigate to the project root directory.
# 4. Run behave: behave features/
