# src/chatbot_eval_pkg/client.py
import requests
import json
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Define an interface for different client types (HTTP, TestClient, etc.)
class BaseChatbotClient(ABC):
    """Abstract base class for chatbot clients."""
    @abstractmethod
    def get_response(self, user_input: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Tuple[Optional[str], Optional[List[str]]]:
        """Sends input and retrieves response and context."""
        pass

    @abstractmethod
    def close_session(self):
        """Closes any underlying connections or sessions."""
        pass

# Implementation for standard HTTP/HTTPS endpoints
class HttpClient(BaseChatbotClient):
    """
    Handles communication with a target chatbot API via HTTP/HTTPS.
    """
    def __init__(self, api_endpoint: str, api_key: Optional[str] = None, timeout: int = 30):
        """
        Initializes the HttpClient.

        Args:
            api_endpoint (str): The URL of the chatbot API endpoint.
            api_key (Optional[str]): Optional API key for authentication.
            timeout (int): Request timeout in seconds.
        """
        if not api_endpoint:
            raise ValueError("Chatbot API endpoint must be provided.")
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session() # Use a session for potential connection pooling
        self.session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})
        if self.api_key:
            # Defaulting to Bearer token, users might need to customize headers
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        logger.info(f"HttpClient initialized for endpoint: {self.api_endpoint}")

    def get_response(self, user_input: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Sends input to the chatbot API and retrieves the response.

        Args:
            user_input (str): The latest user input.
            conversation_history (Optional[List[Dict[str, str]]]): Conversation history.

        Returns:
            Tuple[Optional[str], Optional[List[str]]]: Chatbot response text and optional retrieval context.
        """
        # Flexible payload structure - common patterns
        payload = {
            "input": user_input, # Common key
            "query": user_input, # Another common key
            "prompt": user_input, # Yet another
            "messages": conversation_history + [{"role": "user", "content": user_input}] if conversation_history else [{"role": "user", "content": user_input}],
            "history": conversation_history,
        }
        # Note: A real API likely only uses one of these patterns.
        # Users might need to configure which payload structure to use or subclass this client.
        # For simplicity now, we send common variations. A more robust solution
        # would involve configuration or specific client implementations per API type.

        logger.debug(f"Sending request to {self.api_endpoint} with payload containing user input: {user_input}")

        try:
            response = self.session.post(self.api_endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            response_data = response.json()
            logger.debug(f"Received response: {response_data}")

            # --- Extract Chatbot Response Text ---
            # Flexible extraction - try common keys
            possible_keys = ['response', 'answer', 'text', 'output', 'completion', 'content']
            chatbot_response = None
            if isinstance(response_data, dict):
                for key in possible_keys:
                    if key in response_data:
                        chatbot_response = response_data[key]
                        break
                # Handle nested structures like OpenAI's format
                if not chatbot_response and 'choices' in response_data and isinstance(response_data['choices'], list) and response_data['choices']:
                    choice = response_data['choices'][0]
                    if isinstance(choice, dict) and 'message' in choice and isinstance(choice['message'], dict) and 'content' in choice['message']:
                         chatbot_response = choice['message']['content']
                    elif isinstance(choice, dict) and 'text' in choice:
                         chatbot_response = choice['text']

            elif isinstance(response_data, str): # Handle plain text response
                chatbot_response = response_data

            if chatbot_response is None:
                 logger.error(f"Could not find chatbot response text in API response: {response_data}")
                 return None, None

            # --- Extract Retrieval Context (Optional) ---
            retrieval_context = None
            if isinstance(response_data, dict):
                # Try common keys for context
                context_keys = ['retrieved_context', 'context', 'sources', 'documents']
                for key in context_keys:
                    if key in response_data and isinstance(response_data[key], list):
                        retrieval_context = response_data[key]
                        logger.info(f"Extracted retrieval context from key '{key}': {len(retrieval_context)} items.")
                        break

            return str(chatbot_response), retrieval_context

        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with chatbot API at {self.api_endpoint}: {e}")
            return None, None
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON response from chatbot API: {e}")
            logger.error(f"Response text: {response.text[:500]}...") # Log snippet
            return None, None
        except Exception as e:
            logger.exception(f"An unexpected error occurred during chatbot interaction: {e}") # Log stack trace
            return None, None

    def close_session(self):
        """Closes the underlying requests session."""
        self.session.close()
        logger.info("HttpClient session closed.")


# Implementation for FastAPI TestClient or similar objects
class TestClientWrapper(BaseChatbotClient):
    """
    Wraps a test client object (like FastAPI's TestClient) to conform to the BaseChatbotClient interface.
    """
    def __init__(self, test_client: Any, endpoint: str = "/"):
        """
        Initializes the TestClientWrapper.

        Args:
            test_client: The test client instance (e.g., FastAPI TestClient).
                         Must have a 'post' method compatible with requests.post.
            endpoint (str): The relative path on the test client to post to.
        """
        if not hasattr(test_client, 'post'):
            raise TypeError("Provided test_client object must have a 'post' method.")
        self.test_client = test_client
        self.endpoint = endpoint
        logger.info(f"TestClientWrapper initialized for endpoint: {self.endpoint}")

    def get_response(self, user_input: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Sends input via the test client and retrieves the response.
        Adapts the payload and response extraction logic similarly to HttpClient.
        """
        # Adapt payload structure based on the expected input of the test application
        payload = {
            "input": user_input,
            "history": conversation_history
            # Adjust as needed for the specific test application's API
        }
        logger.debug(f"Sending request via TestClient to {self.endpoint} with payload containing user input: {user_input}")

        try:
            # Use the test client's post method
            response = self.test_client.post(self.endpoint, json=payload)
            response.raise_for_status() # TestClient often raises exceptions directly on failure

            response_data = response.json()
            logger.debug(f"Received response via TestClient: {response_data}")

            # --- Extract Chatbot Response Text (similar logic to HttpClient) ---
            possible_keys = ['response', 'answer', 'text', 'output', 'completion', 'content']
            chatbot_response = None
            if isinstance(response_data, dict):
                for key in possible_keys:
                    if key in response_data:
                        chatbot_response = response_data[key]
                        break
            elif isinstance(response_data, str):
                 chatbot_response = response_data

            if chatbot_response is None:
                 logger.error(f"Could not find chatbot response text in TestClient response: {response_data}")
                 return None, None

            # --- Extract Retrieval Context (similar logic to HttpClient) ---
            retrieval_context = None
            if isinstance(response_data, dict):
                context_keys = ['retrieved_context', 'context', 'sources', 'documents']
                for key in context_keys:
                    if key in response_data and isinstance(response_data[key], list):
                        retrieval_context = response_data[key]
                        logger.info(f"Extracted retrieval context from key '{key}': {len(retrieval_context)} items.")
                        break

            return str(chatbot_response), retrieval_context

        except Exception as e:
            # Catch potential exceptions from the test client or response processing
            logger.exception(f"An error occurred using the TestClientWrapper: {e}")
            return None, None

    def close_session(self):
        """No explicit session closing needed for most test clients."""
        logger.debug("TestClientWrapper close_session called (typically no-op).")
        pass

# Factory function or class to create the appropriate client
def get_chatbot_client(config: Union[Dict[str, Any], Any]) -> BaseChatbotClient:
    """
    Factory function to create a chatbot client based on configuration.

    Args:
        config: Either a dictionary with keys like 'type', 'api_endpoint', 'api_key'
                or a pre-configured client object (e.g., FastAPI TestClient).

    Returns:
        An instance of a BaseChatbotClient subclass.

    Raises:
        ValueError: If the configuration is invalid.
    """
    if isinstance(config, dict):
        client_type = config.get("type", "http").lower()
        if client_type == "http":
            return HttpClient(
                api_endpoint=config.get("api_endpoint"),
                api_key=config.get("api_key"),
                timeout=config.get("timeout", 30)
            )
        # Add other types like 'test_client' if needed, though passing the object directly is simpler
        else:
            raise ValueError(f"Unsupported client type in config: {client_type}")
    elif hasattr(config, 'post'): # Check if it looks like a test client
        # Assume it's a pre-configured test client object
        # The user needs to ensure it has the necessary methods (post, json(), raise_for_status())
        # We might need an 'endpoint' configuration parameter here too if not root '/'
        logger.info("Received pre-configured client object, wrapping with TestClientWrapper.")
        return TestClientWrapper(test_client=config)
    else:
        raise ValueError("Invalid chatbot client configuration provided. Expected dict or compatible client object.")
