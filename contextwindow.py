import tiktoken
import openai

class ContextWindow:

    def __init__(self, high_level_overview, default_data_policy="pure", model_name="gpt-3.5-turbo", max_tokens=2048):
        """
        Initializes the ContextWindow class.
        
        Args:
            high_level_overview (str): High level explanation of the conversation.
            default_data_policy (str, optional): Default data handling policy. Defaults to "pure".
            model_name (str, optional): Name of the OpenAI model being used. Defaults to "gpt-3.5-turbo".
            max_tokens (int, optional): Maximum tokens for a message. Defaults to 2048.
        """
        self.high_level_overview = high_level_overview
        self.default_data_policy = default_data_policy
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.messages = []
        self.enhanced_messages = []
        self.total_tokens = 0
        self.tokens_sent = 0  # Counter for tokens sent to the OpenAI API

    def add_message(self, user_content, data=None, data_policy=None, function_info=None, function_policy="include"):
        """
        Adds a new user message to the context window.

        Args:
            user_content (str): The natural language message from the user.
            data (str, optional): Associated data. Defaults to None.
            data_policy (str, optional): Data handling policy. If None, uses default. Defaults to None.
            function_info (dict, optional): Information about the associated function. Defaults to None.
            function_policy (str, optional): Policy regarding function information (include/don't include). Defaults to "include".
        """
        pass

    def generate_message_summary(self, message_index):
        """
        Generates a summary for the given message based on its data policy.

        Args:
            message_index (int): Index of the message in the enhanced_messages list.
        """
        pass

    def update_messages_list(self):
        """
        Updates the messages list based on the current enhanced_messages. Ensures the conversation stays within the token limit.
        """
        pass

    def visualize_context_window(self):
        """
        Provides a representation of the current messages and enhanced_messages for visualization.
        """
        pass

    def edit_message(self, message_index, new_data_policy=None, new_function_info=None):
        """
        Allows for editing a specific message's data policy or function info.

        Args:
            message_index (int): Index of the message to edit.
            new_data_policy (str, optional): New data handling policy. Defaults to None.
            new_function_info (dict, optional): New function information. Defaults to None.
        """
        pass

    def _get_current_messages_for_api(self):
        """
        Returns the current messages list ready to be sent to the OpenAI API.

        Returns:
            list: List of messages to be sent to the API.
        """
        pass

    def _token_count(self, message):
        """
        Computes the token count of a given message for the model specified in the class instance.

        Args:
            message (str): The input message for which tokens need to be counted.

        Returns:
            int: The number of tokens in the message.
        """
        encoder = tiktoken.encoding_for_model(self.model_name)
        return len(encoder.encode(message))

    def _create_function_info(self, name, description, parameter_list):
        """
        Create a function info dictionary for API function definition.
        """
        function_info = {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }

        for param_name, param_type, param_description, is_required in parameter_list:
            param_info = {
                "type": param_type,
                "description": param_description,
            }

            if param_type == "array":
                param_info["items"] = {"type": "string"}

            function_info["parameters"]["properties"][param_name] = param_info

            if is_required:
                function_info["parameters"]["required"].append(param_name)

        return function_info

    def _call_openai(self, messages, function_info=None, max_retries=1, wait_time=1, examples=[]):
        """
        Communicates with OpenAI API for results.
        """
        self.tokens_sent += self.total_tokens
        functions = [function_info] if function_info else []

        for example in examples:
            example_message = {
                "role": "user",
                "content": f"Document: {example['document']}. Fields: {json.dumps(example['fields'])}."
            }
            messages.append(example_message)

        retries = 0
        while retries < max_retries:
            try:
                if functions:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=messages,
                        functions=functions,
                        temperature=0
                    )
                else:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0
                    )

                if function_info:
                    function_call = response.choices[0].message.get('function_call', {})
                    if function_call:
                        arguments = function_call['arguments']
                        if isinstance(arguments, str):
                            return json.loads(arguments)
                        else:
                            return arguments
                else:
                    return response.choices[0].message.get('content', "")
            except json.JSONDecodeError:
                retries += 1
                time.sleep(wait_time)

        raise Exception("Maximum retries reached. API issues.")