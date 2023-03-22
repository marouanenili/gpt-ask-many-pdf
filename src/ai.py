"AI (LLM) adapter"
# TODO: replace with ai_bricks.ai_openai

BUTCHER_EMBEDDINGS = None # this should be None, as it cuts the embedding vector to n first values (for debugging)

import openai
import tiktoken
import time

def get_token_count(text, retry_delay=5):
    c = 0
    while c < 5:
        c += 1
        try:
            encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
            tokens = encoder.encode(text)
            return len(tokens)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    raise Exception("Failed to get token count")



def use_key(api_key):
    openai.api_key = api_key

def complete(prompt, temperature=0.0,messages=[],retry_delay=5):
    kwargs = dict(
        model = 'gpt-3.5-turbo',
        max_tokens = 4000 - get_token_count(prompt),
        temperature = temperature,
        messages = messages,
        n = 1,
    )
    c = 0
    while c < 5:
        c += 1
        try:
            resp = openai.ChatCompletion.create(**kwargs)
            out = {}
            out['text'] = resp['choices'][0]['message']['content']
            out['usage'] = resp['usage']
            return out
        except openai.error.APIError as e:
            print(f"Error: {e}")
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    raise Exception("Failed to complete")

def old_complete(prompt, temperature=0.0):
	kwargs = dict(
		model='text-davinci-003',
		max_tokens=4000 - get_token_count(prompt),
		temperature=temperature,
		prompt=prompt,
		n=1,
	)
	resp = openai.Completion.create(**kwargs)
	out = {}
	out['text'] = resp['choices'][0]['text']
	out['usage'] = resp['usage']
	return out


def embedding(text):
    resp = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002",
    )
    out = {}
    out['vector'] = list(resp['data'][0]['embedding'][:BUTCHER_EMBEDDINGS])
    out['usage']  = dict(resp['usage'])
    return out

