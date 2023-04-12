import tiktoken
import openai

# Set up the OpenAI API client
openai.api_key = "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"

# Set up the model and prompt
model_engine = "text-davinci-003"
# 4096 tokens limit for gpt-3.5-turbo-0301 (Both input and output tokens count) . see https://platform.openai.com/docs/guides/chat/introduction
model_engine = 'gpt-3.5-turbo'
import json

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


def chat_response(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
    return response, response.choices[0].message.content


chat_prompts = {}
system_content = "You are a data scientist helper. Your name is Tusha. You are an expert at data science and bayesian statistics."
# main menu
chat_main_options = {'load':'load a dataset file','save':'save a model','comp':'compute or run the model','stats':'calculate statistics on the dataset','plot':'plot the features'}

def prompt_main_menu(user_query):
    main_options_str = {f'* {op}\n' for op in chat_main_options.values()}

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": """these are defined as my options:
            {main_option_str}
            ."""},
        {"role": "user", "content":
         f"""which option is the last query about? do not add explanations! (if no option is similar then just say 'not sure'.):
            query: upload my data
            option: load a dataset file
            query: create a graph of weight vs height
            option: plot the features
            query: {user_query}
            option:
            """}
    ]

# rephrase option


def prompt_rephrase(user_query):
    return [{"role": "system", "content": system_content},
            {"role": "user", "content": f"""{user_query}"""}
            ]

# suggest causal network


def prompt_suggest_causal_net(features_list):
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"""suggest a causal network:
        features: rain,season,wet,sprinkler,slippery
        causal network: season->rain, season->sprinkler, sprinkler->wet, wet->slippery,rain->wet
        features: {features_list}
        causal network:
        """}
    ]

def find_response_as_option(options_dict,response_content):

    parts = response_content.split('Option: ')
    if len(parts)>1:
        response_content = parts[1]

    find_ops = [(op,len(op_text)) for op,op_text in options_dict.items() if op_text in response_content]
    find_ops = sorted(find_ops,key=lambda x:-x[1])

    if len(find_ops)>0:
        return find_ops[0][0]

    return None


def infer_chat_query(user_query):
    messages = prompt_main_menu(user_query)
    num_tokens = num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    print(f'infer_chat_query:num_tokens = {num_tokens}')
    print(json.dumps(messages,sort_keys=True, indent=4))

    if num_tokens>4050:
        print(f'ERROR: num_tokens = {num_tokens} over 4050. message will probably be truncated')

    response, response_content = chat_response(messages)
    print(f'infer_chat_query: response_content= {response_content}')

    option = find_response_as_option(chat_main_options, response_content)
    print(f'infer_chat_query: response_content= {response_content}, option = {option}')

    if not option:
        messages = prompt_rephrase(user_query)
        response, response_content = chat_response(messages)

    return response_content
