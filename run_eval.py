# run_eval.py
import argparse
import logging
import time
import os
import sys
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file first
load_dotenv()

# Import from the new package structure
from chatbot_eval_pkg.loader import load_test_cases_from_directory
from chatbot_eval_pkg.client import get_chatbot_client, BaseChatbotClient
from chatbot_eval_pkg.evaluator import evaluate_test_case
from chatbot_eval_pkg.types import EvaluationResult # Import result type

# --- Basic Logging Setup ---
# Configure logging (can be made more sophisticated)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Log to stdout
)
logger = logging.getLogger(__name__) # Get logger for this module

# --- Configuration from Environment ---
TEST_DATA_DIR_DEFAULT = "test_data"
CHATBOT_API_ENDPOINT = os.getenv("CHATBOT_API_ENDPOINT")
CHATBOT_API_KEY = os.getenv("CHATBOT_API_KEY")
DEEPEVAL_EVALUATION_MODEL = os.getenv("DEEPEVAL_EVALUATION_MODEL")
DEEPEVAL_RUN_ASYNC_STR = os.getenv("DEEPEVAL_RUN_ASYNC", "true").lower()
DEEPEVAL_RUN_ASYNC = DEEPEVAL_RUN_ASYNC_STR in ['true', '1', 't', 'y', 'yes']

def run_all_evaluations(
    test_dir: str,
    client: BaseChatbotClient,
    global_model_config: Optional[str] = None,
    run_async: bool = True
) -> List[EvaluationResult]:
    """
    Loads test cases from a directory and runs evaluations using the provided client.

    Args:
        test_dir: The directory containing JSON test case files.
        client: An initialized chatbot client instance.
        global_model_config: Optional configuration for the default evaluation model.
        run_async: Whether to run DeepEval asynchronously.

    Returns:
        A list of EvaluationResult objects.
    """
    all_results: List[EvaluationResult] = []

    # 1. Load Test Cases
    test_cases = load_test_cases_from_directory(test_dir)
    if not test_cases:
        logger.warning(f"No test cases found or loaded from directory: {test_dir}")
        return []

    # 2. Run Evaluation for Each Test Case
    total_tests = len(test_cases)
    logger.info(f"Starting evaluation run for {total_tests} test cases...")
    for i, test_case in enumerate(test_cases):
        test_id = test_case.get("id", f"unknown_{i+1}")
        file_path = test_case.get("_file_path", "N/A")
        logger.info(f"--- Running test {i+1}/{total_tests}: ID = {test_id} ({os.path.basename(file_path)}) ---")
        try:
            result = evaluate_test_case(
                test_case_data=test_case,
                chatbot_client=client,
                global_model_config=global_model_config,
                run_async=run_async
            )
            all_results.append(result)
            logger.info(f"--- Finished test {i+1}/{total_tests}: ID = {test_id}, Success = {result.success} ---")
        except Exception as e:
             logger.error(f"--- CRITICAL ERROR during evaluation for test {i+1}/{total_tests} (ID: {test_id}) ---")
             logger.exception(e)
             # Create a minimal error result object
             error_result = EvaluationResult(
                 id=test_id,
                 success=False,
                 duration=0.0, # Duration might be inaccurate here
                 file_path=file_path,
                 error=f"Critical evaluation error: {e}",
                 test_case_details=test_case
             )
             all_results.append(error_result)

    return all_results

def print_summary_report(results: List[EvaluationResult], total_duration: float):
    """
    Prints a summary report of the evaluation results to the console.

    Args:
        results: The list of EvaluationResult objects.
        total_duration: The total time taken for the run.
    """
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.success)
    failed_tests = total_tests - passed_tests

    print("\n" + "="*60)
    print("Evaluation Summary Report")
    print("="*60)
    print(f"Total Tests Run: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Total Duration: {total_duration:.2f} seconds")
    print("="*60)

    if failed_tests > 0:
        print("\nFailed Test Details:")
        print("-"*60)
        for result in results:
            if not result.success:
                print(f"\nTest ID: {result.id}")
                print(f"  File: {os.path.basename(result.file_path or 'N/A')}")
                print(f"  Status: FAILED")
                print(f"  Reason: {result.error or 'No specific error message.'}")
                # Print metric-specific failures
                if result.metrics_results:
                    failed_metrics_details = [
                        f"{m.metric} (Score: {m.score:.3f if m.score is not None else 'N/A'}, Threshold: {m.threshold}, Reason: {m.reason or 'N/A'})"
                        for m in result.metrics_results if not m.success
                    ]
                    if failed_metrics_details:
                        print("  Failed Metrics:")
                        for detail in failed_metrics_details:
                            print(f"    - {detail}")
                print("-"*20)
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeepEval chatbot evaluations from JSON test cases using chatbot-eval-pkg.")
    parser.add_argument(
        "--test-dir",
        type=str,
        default=os.getenv("TEST_DATA_DIR", TEST_DATA_DIR_DEFAULT),
        help=f"Directory containing JSON test case files (default: reads from TEST_DATA_DIR env var or '{TEST_DATA_DIR_DEFAULT}')",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=CHATBOT_API_ENDPOINT,
        help="Chatbot API endpoint URL (overrides CHATBOT_API_ENDPOINT env var)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=CHATBOT_API_KEY,
        help="Chatbot API key (overrides CHATBOT_API_KEY env var)",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=DEEPEVAL_EVALUATION_MODEL,
        help="Evaluation model name (e.g., 'gpt-4', overrides DEEPEVAL_EVALUATION_MODEL env var)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Run DeepEval evaluations synchronously (overrides DEEPEVAL_RUN_ASYNC env var)",
    )

    args = parser.parse_args()

    # Determine final config, prioritizing CLI args over environment variables
    final_endpoint = args.endpoint
    final_api_key = args.api_key
    final_eval_model = args.eval_model
    final_run_async = not args.sync if args.sync else DEEPEVAL_RUN_ASYNC # CLI --sync overrides env var

    if not final_endpoint:
        logger.critical("Chatbot API endpoint is required. Set CHATBOT_API_ENDPOINT environment variable or use --endpoint argument.")
        sys.exit(1)

    logger.info(f"Using Test Directory: {args.test_dir}")
    logger.info(f"Using API Endpoint: {final_endpoint}")
    logger.info(f"Using Evaluation Model: {final_eval_model or 'DeepEval Default'}")
    logger.info(f"DeepEval Async Mode: {final_run_async}")

    chatbot_client = None
    try:
        # Initialize Chatbot Client using the factory
        client_config = {
            "type": "http", # Assuming HTTP client
            "api_endpoint": final_endpoint,
            "api_key": final_api_key
        }
        chatbot_client = get_chatbot_client(client_config)

        # Run evaluations
        start_time = time.time()
        evaluation_results = run_all_evaluations(
            test_dir=args.test_dir,
            client=chatbot_client,
            global_model_config=final_eval_model,
            run_async=final_run_async
        )
        end_time = time.time()

        total_time = end_time - start_time
        print_summary_report(evaluation_results, total_time)

        # Exit with status code 1 if any tests failed, 0 otherwise
        if any(not r.success for r in evaluation_results):
            logger.info("Evaluation run finished with failures.")
            sys.exit(1)
        else:
            logger.info("Evaluation run finished successfully.")
            sys.exit(0)

    except Exception as e:
         logger.critical(f"An unexpected error occurred during the run: {e}", exc_info=True)
         sys.exit(2) # Different exit code for critical errors
    finally:
        # Ensure client session is closed if it was initialized
        if chatbot_client:
            chatbot_client.close_session()
