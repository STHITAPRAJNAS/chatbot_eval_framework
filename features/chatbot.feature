# chatbot_eval_framework/features/chatbot.feature
Feature: Chatbot Evaluation using JSON Test Cases

  Scenario Outline: Evaluate chatbot response based on a JSON test case definition
    Given a chatbot evaluation test case defined in "<TestCaseFile>"
    When the evaluation is performed for this test case
    Then the evaluation result should be successful

    Examples: Test Cases from Directory
      | TestCaseFile                         |
      | test_data/sample_single_turn.json    |
      | test_data/sample_multi_turn.json     |
      # Add more rows for other specific JSON files you want to run via Behave

