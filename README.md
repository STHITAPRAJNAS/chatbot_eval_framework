# Chatbot Evaluation Package (`chatbot-eval-pkg`)

A reusable Python package for evaluating chatbot conversations using [DeepEval](https://github.com/confident-ai/deepeval), integrated with `pytest` and `behave`.

## Features

*   Evaluates chatbot responses based on JSON test case definitions.
*   Supports single-turn and multi-turn conversations.
*   Leverages DeepEval metrics for comprehensive evaluation (e.g., AnswerRelevancy, Faithfulness, Hallucination, custom GEval).
*   Integrates with `pytest` for standard test execution and reporting.
*   Integrates with `behave` for BDD-style testing.
*   Configurable via environment variables.
*   Flexible client for interacting with different chatbot APIs (HTTP endpoints, FastAPI `TestClient`).

## Installation

It's recommended to install this package in a virtual environment. You can use `pip` or `uv`.

1.  **Clone the repository (if not already done):**
    ```bash
    git clone <your-repo-url>
    cd chatbot-eval-framework
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Using venv
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.venv\Scripts\activate`

    # Or using uv (if installed)
    uv venv
    source .venv/bin/activate
    ```

3.  **Install the package in editable mode with test dependencies:**
    ```bash
    # Using pip
    pip install -e ".[test]"

    # Using uv
    uv pip install -e ".[test]"
    ```
    *   Use `.[test,aws]` if you need AWS Bedrock support (`boto3`).
    *   Use `.[dev]` to install all optional dependencies.

## Configuration

Configuration is primarily handled through environment variables. Create a `.env` file in the project root directory or set the variables directly in your environment.

**Required:**

*   `CHATBOT_API_ENDPOINT`: The URL of the chatbot API you want to test.

**Optional:**

*   `CHATBOT_API_KEY`: API key for the chatbot endpoint (if required).
*   `TEST_DATA_DIR`: Path to the directory containing JSON test case files (defaults to `./test_data`).
*   `OPENAI_API_KEY`: Required by DeepEval's default evaluation models if you don't configure a specific one.
*   `DEEPEVAL_EVALUATION_MODEL`: Specify a global evaluation model for DeepEval metrics (e.g., `gpt-4`, `claude-2`). Overrides DeepEval defaults. Can be overridden per-metric in JSON.
*   `DEEPEVAL_RUN_ASYNC`: Set to `false` to run DeepEval metrics synchronously (defaults to `true`).
*   `LOG_LEVEL`: Set the logging level (e.g., `DEBUG`, `INFO`, `WARNING`). Defaults to `INFO`.
*   `AWS_REGION_NAME`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`: Required if using Bedrock models for evaluation.

See `.env.example` for a template.

## JSON Test Case Structure

Place your test case definitions as `.json` files in the directory specified by `TEST_DATA_DIR`.

**Required Fields:**

*   `id` (str): A unique identifier for the test case.
*   `input` (str) OR `messages` (List[Dict]):
    *   Use `input` for single-turn tests.
    *   Use `messages` for multi-turn tests, following the format `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`. The last message *must* be from the `user`.
*   `metrics` (List[Dict]): A list of DeepEval metrics to apply. Each dictionary defines one metric:
    *   `name` (str): The name of the DeepEval metric (e.g., `AnswerRelevancy`, `Faithfulness`, `GEval`). See `src/chatbot_eval_pkg/metrics.py` for supported names.
    *   `threshold` (float): The minimum acceptable score for the metric (0.0 to 1.0). Defaults to 0.5 if omitted.
    *   **Metric-specific parameters:** Add any other parameters required by the specific DeepEval metric (e.g., `criteria` for `GEval`, `model` to override the global evaluation model).

**Optional Fields:**

*   `expected_output` (str): The ideal or expected response (used by some metrics).
*   `context` (List[str]): Context provided *to* the chatbot during the request (used by some metrics).
*   `retrieval_context` (List[str]): The ground truth context relevant to the input (used by RAG metrics like Faithfulness, Contextual Precision/Recall).

**Example (`test_data/sample_single_turn.json`):**

```json
{
  "id": "sample-st-001",
  "input": "What is the capital of France?",
  "expected_output": "Paris",
  "context": ["Geography facts"],
  "retrieval_context": ["Paris is the capital and most populous city of France."],
  "metrics": [
    {
      "name": "AnswerRelevancy",
      "threshold": 0.8
    },
    {
      "name": "Faithfulness",
      "threshold": 0.7
    },
    {
      "name": "GEval",
      "criteria": "Is the answer concise and accurate?",
      "threshold": 0.6,
      "model": "gpt-4" // Metric-specific model override
    }
  ]
}
```

## Usage

### 1. Command Line Interface (`run_eval.py`)

This script runs all test cases found in the specified directory and prints a summary report.

```bash
# Ensure environment variables are set (e.g., via .env)
python run_eval.py

# Override configuration via arguments:
python run_eval.py --test-dir path/to/your/tests --endpoint http://localhost:8000/chat --eval-model gpt-3.5-turbo --sync
```

The script exits with status code 0 if all tests pass, 1 if any test fails, and 2 for critical errors.

### 2. Pytest Integration

Run evaluations using `pytest`. Tests are defined in `tests/test_chatbot_json.py`.

```bash
# Ensure environment variables are set
pytest tests/

# Generate an HTML report
pytest tests/ --html=report.html --self-contained-html
```

### 3. Behave Integration

Run evaluations using `behave` for BDD testing. Features are defined in `features/` and steps in `features/steps/`.

```bash
# Ensure environment variables are set
behave features/
```
*Note: For robust client setup/teardown in `behave`, it's recommended to implement `before_feature`/`after_feature` hooks in `features/environment.py` (see comments in `features/steps/chatbot_steps.py` for an example).*

## Package Structure (`src/chatbot_eval_pkg/`)

*   `__init__.py`: Package initializer.
*   `client.py`: Defines `BaseChatbotClient` interface and implementations (`HttpClient`, `TestClientWrapper`). Includes `get_chatbot_client` factory.
*   `loader.py`: Functions to load test cases from JSON files (`load_test_case_from_file`, `load_test_cases_from_directory`).
*   `metrics.py`: Handles instantiation of DeepEval metrics based on JSON configuration (`create_metrics`). Includes default metric name mapping.
*   `types.py`: Defines dataclasses for structured results (`EvaluationResult`, `MetricResultDetail`).
*   `evaluator.py`: Contains the core evaluation logic (`evaluate_test_case`) orchestrating client interaction and DeepEval execution.

## Extending the Package

*   **Supporting New Chatbot APIs:** Subclass `BaseChatbotClient` and implement the `get_response` and `close_session` methods for your specific API interaction logic. Update the `get_chatbot_client` factory or pass your custom client instance directly.
*   **Adding Custom Metrics:** If you have custom DeepEval metrics, you can provide your own `metric_mapping` dictionary to `create_metrics` or `evaluate_test_case` (if exposed).
*   **Modifying Payload/Response Handling:** Adjust the payload creation and response parsing logic within the relevant client implementation (`HttpClient` or your custom client) if your API differs significantly.
