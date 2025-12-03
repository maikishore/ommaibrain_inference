'''from openai import OpenAI

client = OpenAI(
    base_url="https://maibrain--vllm-llama-3-1-8b-serve.modal.run/v1",
    api_key="not-needed"   # Modal endpoint does not require auth
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B",  # same model you served
    messages=[{"role": "user", "content": "What is  Nuclear Fusion?"}],
    max_tokens=80,
)

#print(response)

def get_formatted_text_from_chat_completion(completion_response):
    """
    Extracts and formats the content from a ChatCompletion object or dictionary,
    handling both subscriptable dictionaries and objects with attributes (like Pydantic models).

    Args:
        completion_response (Union[dict, object]): The response data structure.

    Returns:
        str: The extracted content, formatted with newlines preserved.
    """
    content = ""
    try:
        # Try accessing attributes using dot notation (for objects)
        # We assume the standard structure: response.choices[0].message.content
        choice = completion_response.choices[0]
        content = choice.message.content
    except (AttributeError, TypeError, IndexError):
        try:
            # Fallback to dictionary key access (for dicts)
            choice = completion_response['choices'][0]
            content = choice['message']['content']
        except (KeyError, IndexError, TypeError) as e_dict:
            print(f"Error accessing attributes/keys: {e_dict}")
            return "Could not extract content due to a parsing error."

    # Perform minor cleanup on the content string
    if content:
        formatted_content = content.replace("nn.'", "\n")
        formatted_content = formatted_content.replace("?'", "")
        # Remove any leading/trailing whitespace left by cleanup or the original string
        formatted_content = formatted_content.strip() 
        return formatted_content
    
    return "Content field was empty."

# --- Example Usage (Conceptual) ---
# When using this function in your environment, you would pass the variable
# that holds the result of your SDK call directly:

# Example of how you might call it in your actual application code:
# my_completion_object = ChatCompletion(...) # <-- This is the object causing the error
# formatted_output = get_formatted_text_from_chat_completion(my_completion_object)
# print(formatted_output)

# The function above should now correctly process your specific object type.
print(get_formatted_text_from_chat_completion(response))
'''
from openai import OpenAI

client = OpenAI(
    base_url="https://maibrain--vllm-meta-llama-3-8b-serve.modal.run/v1",
    api_key="none"
)

response = client.completions.create(
    model="meta-llama/Meta-Llama-3-8B",
    prompt="What is Nuclear Fusion?",
    max_tokens=100,
)

print(response.choices[0].text)
