import openai


import os
import pprint
  
# Get the list of user's
# environment variables
env_var = os.environ
  
# Print the list of user's
# environment variables

openai.api_key = os.environ['TUSHA_OPENAI_API_KEY']

print(f'openai.api_key = {openai.api_key}')

# Set up the model and prompt
model_engine = 'gpt-3.5-turbo'# "text-davinci-003"

cause = 'smoking'
effect = 'cancer'
prompt = f"can {cause} increase {effect}.answer yes or no"
prompt = f"can {cause} increase {effect}. why?"
# prompt = 'why?'

# Generate a response
completion = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)

response = completion.choices[0].text
print(response)