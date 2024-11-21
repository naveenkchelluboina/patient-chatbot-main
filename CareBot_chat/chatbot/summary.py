import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY,  model="gpt-4o-mini")

def summarize_conversation(conversation):
    """
    Summarizes a conversation using OpenAI GPT models.

    Args:
    conversation (list of tuples): A list of (speaker, message) pairs representing the conversation.

    Returns:
    str: A summarized version of the conversation.
    """
    # Prepare the conversation text for input
    full_conversation = ""
    for speaker, message in conversation:
        full_conversation += f"{speaker}: {message}\n"

    # Prompt OpenAI to summarize the conversation
    prompt = f"Summarize the following conversation:\n\n{full_conversation}\n\nSummary:"
    
    response = llm(prompt)

    # response = openai.Completion.create(
    #     engine="",  # You can use "gpt-3.5-turbo" for chat-based completion as well
    #     prompt=prompt,
    #     max_tokens=150,
    #     temperature=0.5

    # )
    summary = response.content
    print(summary)
    return summary