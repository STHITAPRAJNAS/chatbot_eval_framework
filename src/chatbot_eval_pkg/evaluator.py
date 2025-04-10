# src/chatbot_eval_pkg/evaluator.py
import os
import logging
import time
from typing import Dict, Any, List, Optional, Union

# DeepEval imports
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, ConversationalTestCase, Message
from deepeval.metrics import BaseMetric
from deepeval.dataset import EvaluationDataset # For potential future use

# Package imports
from .client import BaseChatbotClient # Use the abstract base class
from .metrics import create_metrics
from .types import EvaluationResult, MetricResultDetail # Use the result dataclass

logger = logging.getLogger(__name__)

def evaluate_test_case(
    test_case_data: Dict[str, Any],
    chatbot_client: BaseChatbotClient,
    global_model_config: Optional[Union[str, Dict[str, Any]]] = None,
    run_async: bool = True
) -> EvaluationResult:
    """
    Evaluates a single test case against the chatbot using DeepEval.

    Args:
        test_case_data: The dictionary representing the loaded JSON test case.
                        Must include 'id', ('input' or 'messages'), and 'metrics'.
                        Can optionally include 'expected_output', 'context',
                        'retrieval_context'. '_file_path' is added by the loader.
        chatbot_client: An instance of a BaseChatbotClient subclass to interact
                        with the target chatbot.
        global_model_config: Optional configuration for a default evaluation model
                             passed to create_metrics.
        run_async: Whether DeepEval should run evaluations asynchronously.

    Returns:
        An EvaluationResult object containing the detailed results.
    """
    start_time = time.time()
    test_id = test_case_data.get("id", "unknown_id")
    file_path = test_case_data.get("_file_path", "unknown_file")
    logger.info(f"Starting evaluation for test case ID: {test_id} from file: {os.path.basename(file_path)}")

    # Initialize result object
    result = EvaluationResult(
        id=test_id,
        success=False, # Default to False
        duration=0.0,
        file_path=file_path,
        test_case_details=test_case_data,
    )

    try:
        # 1. Instantiate Metrics
        metric_configs = test_case_data.get("metrics", [])
        if not metric_configs:
            result.error = "No metrics defined in the test case."
            logger.error(f"Test case {test_id}: No metrics defined.")
            result.duration = time.time() - start_time
            return result

        # Pass global config and async flag to metric creation
        metrics: List[BaseMetric] = create_metrics(
            metric_configs=metric_configs,
            global_model_config=global_model_config,
            run_async=run_async
        )
        if not metrics:
             result.error = "Failed to instantiate any metrics from the configuration."
             logger.error(f"Test case {test_id}: Failed to instantiate metrics.")
             result.duration = time.time() - start_time
             return result

        # 2. Determine Test Case Type and Get Chatbot Response
        is_conversational = "messages" in test_case_data
        actual_output: Optional[str] = None
        retrieval_context_extracted: Optional[List[str]] = None
        user_input: Optional[str] = None # Define user_input here

        if is_conversational:
            messages_data = test_case_data["messages"]
            if not messages_data or not isinstance(messages_data, list):
                 result.error = "Invalid 'messages' format in conversational test case."
                 logger.error(f"Test case {test_id}: Invalid 'messages' format.")
                 result.duration = time.time() - start_time
                 return result

            last_user_message = next((m for m in reversed(messages_data) if m.get('role') == 'user'), None)
            if not last_user_message:
                 result.error = "Conversational test case 'messages' does not end with a 'user' role."
                 logger.error(f"Test case {test_id}: No final user message found.")
                 result.duration = time.time() - start_time
                 return result

            user_input = last_user_message["content"]
            conversation_history = messages_data[:-1] # History excludes the last user message
            actual_output, retrieval_context_extracted = chatbot_client.get_response(user_input, conversation_history)

        else: # Single-turn
            user_input = test_case_data.get("input")
            if not user_input:
                 result.error = "Missing 'input' field in single-turn test case."
                 logger.error(f"Test case {test_id}: Missing 'input' field.")
                 result.duration = time.time() - start_time
                 return result
            actual_output, retrieval_context_extracted = chatbot_client.get_response(user_input)

        result.chatbot_response = actual_output
        result.retrieval_context_extracted = retrieval_context_extracted

        if actual_output is None:
            result.error = "Failed to get response from chatbot client."
            logger.error(f"Test case {test_id}: Chatbot client call failed.")
            result.duration = time.time() - start_time
            return result

        # 3. Create DeepEval Test Case Object
        # Note: retrieval_context_extracted from the *actual* response might be needed
        # by some metrics (e.g., faithfulness against actual context). DeepEval's standard
        # metrics often use the 'retrieval_context' field (ground truth) provided here.
        # If actual context needs evaluation, custom metrics or adjustments might be needed.
        shared_params = {
            "actual_output": actual_output,
            "expected_output": test_case_data.get("expected_output"),
            "context": test_case_data.get("context"),
            "retrieval_context": test_case_data.get("retrieval_context"),
            "id": test_id,
            # Pass extracted context if a metric needs it (e.g., custom metric)
            # This might require mapping it to a specific parameter name in the metric
            # "actual_retrieval_context": retrieval_context_extracted, # Example
        }

        deepeval_test_case: Optional[Union[LLMTestCase, ConversationalTestCase]] = None
        if is_conversational:
             deepeval_messages = [Message(role=m["role"], content=m["content"]) for m in messages_data]
             # Add the actual assistant response to the messages list for evaluation context
             deepeval_messages.append(Message(role="assistant", content=actual_output))
             deepeval_test_case = ConversationalTestCase(
                 messages=deepeval_messages,
                 # Pass other relevant fields if ConversationalTestCase accepts them directly
                 # Check DeepEval documentation for ConversationalTestCase parameters
                 **{k: v for k, v in shared_params.items() if k in ConversationalTestCase.__annotations__ or k in ['id']} # Filter params
             )
             # Ensure actual_output is explicitly set if needed by metrics accessing it directly
             deepeval_test_case.actual_output = actual_output

        else: # Single-turn
            if user_input is None: # Should not happen due to earlier check, but satisfy type checker
                 raise ValueError("User input is None for single-turn case, this should not happen.")
            deepeval_test_case = LLMTestCase(
                input=user_input,
                **{k: v for k, v in shared_params.items() if k in LLMTestCase.__annotations__ or k in ['id']} # Filter params
            )


        # 4. Run Evaluation
        logger.info(f"Test case {test_id}: Running deepeval.evaluate() {'async' if run_async else 'sync'}...")
        evaluation_results_list = evaluate(
            test_cases=[deepeval_test_case], # evaluate expects a list
            metrics=metrics,
            print_results=False, # We handle reporting
            run_async=run_async
        )
        logger.info(f"Test case {test_id}: deepeval.evaluate() finished.")

        # 5. Process Results
        if not evaluation_results_list:
             result.error = "DeepEval evaluate() returned no results."
             logger.error(f"Test case {test_id}: evaluate() returned empty list.")
             result.duration = time.time() - start_time
             return result

        # Get the result object for our single test case
        eval_result_obj = evaluation_results_list[0]

        result.success = eval_result_obj.success # Use DeepEval's overall success flag

        # Store detailed metric results in the dataclass structure
        detailed_metrics = []
        for metric_result in eval_result_obj.metrics_results:
            detail = MetricResultDetail(
                metric=metric_result.metric, # Name of the metric class/instance
                score=metric_result.score,
                threshold=metric_result.threshold,
                success=metric_result.success,
                reason=metric_result.reason,
                error=metric_result.error,
            )
            detailed_metrics.append(detail)
            log_level = logging.INFO if detail.success else logging.WARNING
            logger.log(log_level, f"Test case {test_id} - Metric '{detail.metric}': Score={detail.score:.4f if detail.score is not None else 'N/A'}, Threshold={detail.threshold}, Success={detail.success}, Reason={detail.reason}")

        result.metrics_results = detailed_metrics

        if not result.success:
            failed_metrics = [m.metric for m in detailed_metrics if not m.success]
            result.error = f"One or more metrics failed: {', '.join(failed_metrics)}"
            logger.warning(f"Test case {test_id}: Evaluation failed. Failed metrics: {', '.join(failed_metrics)}")
        else:
            logger.info(f"Test case {test_id}: Evaluation successful.")


    except Exception as e:
        logger.exception(f"Test case {test_id}: Unexpected error during evaluation: {e}")
        result.error = f"Unexpected evaluation error: {str(e)}"
        result.success = False # Ensure failure on exception

    finally:
        result.duration = time.time() - start_time
        logger.info(f"Test case {test_id}: Evaluation completed in {result.duration:.2f} seconds. Overall Success: {result.success}")

    return result
