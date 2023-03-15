import openai
import json
import datetime

def query_completion(prompt: str, engine: str = 'text-davinci-003', temperature: float = 0.5, max_tokens: int = 1500, top_p: int = 1, frequency_penalty: int = 0.5, presence_penalty: int = 0.2) -> object:
    """
    Function for querying GPT-3.
    """
    estimated_prompt_tokens = int(len(prompt.split()) * 1.6)
    estimated_answer_tokens = 2049 - estimated_prompt_tokens
    response = openai.Completion.create(
    engine=engine,
    prompt=prompt,
    temperature=temperature,
    max_tokens=min(4096-estimated_prompt_tokens, max_tokens),
    top_p=top_p,
    frequency_penalty=frequency_penalty,
    presence_penalty=presence_penalty
    )
    return response
    
def lambda_handler(event, context):
    '''Provide an event that contains the following keys:
      - prompt: text of an open ai prompt
    '''
    
    openai.api_key = os.environ["OPENAI_API_KEY"]
    
    print("Init:")
    print(datetime.datetime.now())
    print("Event:")
    print(event)

    body = json.loads(event['body'])
    prompt = body['prompt']
        
    max_tokens = 1500
    
    response = query_completion(prompt)
    response_text = response['choices'][0]['text'].strip()

    response = {
        "statusCode": 200,
        "headers": {'Content-Type': 'application/json'},
        "body": response_text
    }
    
    return response
