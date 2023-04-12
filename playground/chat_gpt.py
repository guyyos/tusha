import openai

# Set up the OpenAI API client
openai.api_key = "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"

# Set up the model and prompt
model_engine = "text-davinci-003"

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