# src/chatbot_eval_pkg/metrics.py
import logging
import inspect
from typing import List, Dict, Any, Optional, Type

# Import specific model types if needed for type hinting or instantiation logic
# Example: from deepeval.models import OpenAIModel, BedrockModel
# For now, rely on deepeval handling model strings/instances passed to metrics
import deepeval.metrics
from deepeval.metrics import BaseMetric

logger = logging.getLogger(__name__)

# --- Default Metric Mapping ---
# Users can potentially extend or override this mapping programmatically
DEFAULT_METRIC_NAME_TO_CLASS: Dict[str, Type[BaseMetric]] = {
    # Standard Metrics
    "AnswerRelevancy": deepeval.metrics.AnswerRelevancyMetric,
    "Faithfulness": deepeval.metrics.FaithfulnessMetric,
    "ContextualPrecision": deepeval.metrics.ContextualPrecisionMetric,
    "ContextualRecall": deepeval.metrics.ContextualRecallMetric,
    "ContextualRelevancy": deepeval.metrics.ContextualRelevancyMetric,
    "Hallucination": deepeval.metrics.HallucinationMetric,
    "Bias": deepeval.metrics.BiasMetric,
    "Toxicity": deepeval.metrics.ToxicityMetric,
    "Summarization": deepeval.metrics.SummarizationMetric,

    # GEval based Metrics
    "GEval": deepeval.metrics.GEval, # Requires 'criteria'

    # Execution Metrics
    "Latency": deepeval.metrics.LatencyMetric,
    "Cost": deepeval.metrics.CostMetric,

    # Add more metrics from deepeval.metrics as needed...
    # "KnowledgeRetention": deepeval.metrics.KnowledgeRetentionMetric, # Example
}

def _instantiate_evaluation_model(model_config: Union[str, Dict[str, Any]]) -> Optional[Any]:
    """
    Attempts to instantiate an evaluation model based on config.
    Note: This is a basic implementation. DeepEval metrics often handle model
          instantiation internally based on strings or environment variables.
          Passing the model name string or config dict might be sufficient.
          This helper is more for complex cases or future expansion.
    """
    if isinstance(model_config, str):
        # Assume it's a model name string (e.g., "gpt-4", "claude-2")
        # DeepEval metrics usually handle this directly.
        logger.info(f"Using model name string for evaluation: {model_config}")
        return model_config # Return the string itself
    elif isinstance(model_config, dict):
        model_type = model_config.get("type", "openai").lower() # Default to openai
        model_name = model_config.get("model")
        if not model_name:
            logger.error("Model configuration dict missing 'model' name.")
            return None

        logger.info(f"Attempting to configure evaluation model: type='{model_type}', name='{model_name}'")
        # Example for specific instantiation if needed in the future:
        # try:
        #     if model_type == "bedrock":
        #         from deepeval.models import BedrockModel
        #         region = model_config.get("aws_region_name") # Get region from config
        #         if not region:
        #             logger.warning("Bedrock model config missing 'aws_region_name', using default.")
        #         return BedrockModel(model=model_name, region_name=region)
        #     elif model_type == "openai":
        #          from deepeval.models import OpenAIModel
        #          return OpenAIModel(model=model_name)
        #     # Add other model types (Azure, Cohere, etc.)
        #     else:
        #          logger.warning(f"Unsupported model type '{model_type}' for direct instantiation. Passing name string.")
        #          return model_name
        # except ImportError as e:
        #      logger.error(f"Failed to import model class for type '{model_type}': {e}. Ensure necessary extras are installed.")
        #      return None
        # except Exception as e:
        #      logger.error(f"Failed to instantiate model '{model_name}' (type: {model_type}): {e}")
        #      return None
        # For now, just return the name string, as metrics handle it well
        return model_name
    else:
        logger.warning(f"Invalid model configuration type: {type(model_config)}. Expected str or dict.")
        return None


def create_metrics(
    metric_configs: List[Dict[str, Any]],
    global_model_config: Optional[Union[str, Dict[str, Any]]] = None,
    run_async: bool = True,
    metric_mapping: Optional[Dict[str, Type[BaseMetric]]] = None
) -> List[BaseMetric]:
    """
    Instantiates DeepEval metric objects based on configurations.

    Args:
        metric_configs: A list of dictionaries, where each dict defines a metric
                        from the JSON test case (e.g., {'name': 'AnswerRelevancy',
                        'threshold': 0.7, 'model': 'gpt-4', 'criteria': '...', ...}).
        global_model_config: Optional configuration for a model to be used by default
                             for all metrics (can be a model name string or a dict
                             like {'type': 'bedrock', 'model': 'claude-v2', 'aws_region_name': 'us-east-1'}).
                             Metric-specific 'model' config overrides this.
        run_async: Whether to configure metrics to run asynchronously (default: True).
        metric_mapping: Optional dictionary mapping metric names (from JSON) to
                        DeepEval BaseMetric subclasses. Defaults to internal mapping.

    Returns:
        A list of instantiated DeepEval metric objects.
    """
    metrics = []
    active_metric_mapping = metric_mapping or DEFAULT_METRIC_NAME_TO_CLASS

    # Instantiate global model configuration if provided (might just be a string)
    global_evaluation_model = _instantiate_evaluation_model(global_model_config) if global_model_config else None

    for config in metric_configs:
        metric_name = config.get("name")
        if not metric_name:
            logger.warning(f"Skipping metric config due to missing 'name': {config}")
            continue

        MetricClass = active_metric_mapping.get(metric_name)
        if not MetricClass:
            logger.warning(f"Metric name '{metric_name}' not found in mapping. Skipping.")
            continue

        try:
            # --- Prepare arguments for the specific MetricClass ---
            init_params = inspect.signature(MetricClass.__init__).parameters
            metric_args = {}

            # 1. Get parameters directly from the metric's JSON config
            #    (excluding 'name' which is used for lookup)
            for key, value in config.items():
                if key != "name" and key in init_params:
                    metric_args[key] = value
                elif key == "model": # Handle 'model' specifically for override
                    pass # Will be handled in step 3
                elif key not in init_params:
                     logger.debug(f"Parameter '{key}' from JSON config is not in {metric_name}.__init__, ignoring.")


            # 2. Add/Override global configurations if not already set by metric config
            #    Only add if the metric's __init__ accepts them
            if "async_mode" in init_params and "async_mode" not in metric_args:
                 metric_args["async_mode"] = run_async
            if "strict_mode" in init_params and "strict_mode" not in metric_args:
                 # Default strict_mode if needed, or get from a global config
                 pass # metric_args["strict_mode"] = False # Example

            # 3. Handle Model Configuration (Priority: metric > global > deepeval default)
            metric_specific_model_config = config.get("model")
            model_to_use = None
            if metric_specific_model_config:
                model_to_use = _instantiate_evaluation_model(metric_specific_model_config)
                logger.info(f"Using metric-specific model config for {metric_name}: {metric_specific_model_config}")
            elif global_evaluation_model:
                model_to_use = global_evaluation_model
                logger.info(f"Using global model config for {metric_name}: {global_model_config}")
            # Else: Let DeepEval handle its default model (usually requires OPENAI_API_KEY env var)

            if model_to_use and "model" in init_params:
                 metric_args["model"] = model_to_use
            elif model_to_use and "evaluation_model" in init_params: # Some metrics use different param name
                 metric_args["evaluation_model"] = model_to_use


            # 4. Set threshold if not provided in config and if accepted by metric
            if "threshold" in init_params and "threshold" not in metric_args:
                 default_threshold = 0.5 # Default threshold
                 metric_args["threshold"] = default_threshold
                 logger.debug(f"Using default threshold {default_threshold} for {metric_name}")


            # 5. Validate required arguments (like 'criteria' for GEval)
            #    This could be more sophisticated by checking signature for non-optional params
            if metric_name == "GEval" and "criteria" not in metric_args:
                 logger.error(f"GEval metric requires 'criteria' in its configuration. Skipping metric: {config}")
                 continue


            # --- Instantiate the metric ---
            logger.debug(f"Instantiating {metric_name} with args: {metric_args}")
            metric_instance = MetricClass(**metric_args)
            metrics.append(metric_instance)

        except TypeError as e:
             logger.error(f"TypeError instantiating metric '{metric_name}' with args {metric_args}. Check arguments and DeepEval version compatibility. Error: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error instantiating metric '{metric_name}': {e}") # Log stack trace

    logger.info(f"Successfully created {len(metrics)} metric instances.")
    return metrics
