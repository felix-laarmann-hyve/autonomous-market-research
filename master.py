from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, create_structured_chat_agent
from langchain.agents import AgentExecutor
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler

import streamlit as st
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

st.set_page_config(page_title="HYVE AutoInno Pipeline", page_icon="bee",)

hide_st_style = """
    <style>    
    footer {visibility: hidden;}    
    </style>
"""

st.markdown(hide_st_style, unsafe_allow_html=True)

st_callback = StreamlitCallbackHandler(st.container())




from langchain.schema import HumanMessage, AIMessage, SystemMessage  # Import message classes



from langchain.globals import set_debug

# set_debug(True)

class MyCustomHandler(BaseCallbackHandler):

    def on_chain_start(self, serialized, inputs, **kwargs):
        st.info("chain started")


llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

search = TavilySearchResults()
tools = [search]
prompt = hub.pull("hwchase17/openai-functions-agent")
research_agent = create_tool_calling_agent(llm, tools, prompt)

# tools = [TavilySearchResults(max_results=1)]
# prompt = hub.pull("hwchase17/structured-chat-agent")
# research_agent = create_structured_chat_agent(llm, tools, prompt)


agent_executor = AgentExecutor(agent=research_agent, tools=tools, verbose=True, handle_parsing_errors=True)



chat_history = [
    SystemMessage(content="You are an AI specialized in innovation processes. Your task is to guide through the entire innovation process from research to concept creation."),
    SystemMessage(content="You are equipped with a search tool."),
    SystemMessage(content="You conclude every task by suggesting a next step. For this consider the following process."),
    SystemMessage(content="Start with 1.1 researching trends in the given field and format the results as a table. Then proceed with 1.2 researching products and competitors in the given field and format the results as a table. Finally, for this step, 1.3 research painpoints for each of the relevant products and format the results as a table. Use product reviews and Reddit for this research."),
    SystemMessage(content="Next, 2.1 evaluate the painpoints by impact. 2.2 for the top 5 painpoints, define a How Might We Statement."),
    SystemMessage(content="Then, 3.1 dive into ideation and suggest 10 ideas for each painpoint using ideation methods like SCAMPER, TRIZ, and analogy thinking. 3.2 Evaluate the ideas by potential and effort and format the results in a table."),
    SystemMessage(content="For the top idea, 4.1 create a persona. 4.2 Create a concept description mentioning the target group, value, and idea description. 4.3 Suggest a roadmap for next steps including advice on how to prototype and test the idea."),
    SystemMessage(content="Then, 5.1 generate a briefing for another AI system to generate a web application that prototypes the core functionality of the top idea. format the briefing as code"),
    SystemMessage(content="Walk through this process step by step, first planning the action, then executing it, and always finish your answer by suggesting a next step!"),
    HumanMessage(content="You are my innovation expert. I will give you a topic and you will conduct the entire innovation process from user, trend and tech research to ideation and concept creation."),
    AIMessage(content="Now provide your topic so I can dive into the research!")
]

message_history = ChatMessageHistory()
message_history.messages = chat_history


agent_with_chat_history = RunnableWithMessageHistory(    
    agent_executor,        
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)




st.title('🐝 HYVE AutoInno Pipeline')

with st.form('my_form'):
    initial_input = st.text_area('Enter your Innovation Challenge:', 'next generation of tooth brushes')
    iterations = st.number_input(label = 'iterations', min_value=0, max_value=None, value=3, step=1)
    submitted = st.form_submit_button('Submit')

if submitted:
    # Define a global variable for result
    result = agent_with_chat_history.invoke(
        {"chat_history": chat_history, "input": initial_input},
        config={"configurable": {"session_id": "<foo>"},"callbacks":[st_callback]}
    )

    # st.info(result.get('output'))
    with st.chat_message("assistant"):
        st.write(result.get('output'))


    counter = 0

    while counter < iterations:
        output = result.get('output')
        if isinstance(output, str):
            new_input = output
        else:
            new_input = str(output)
        # Invoke the agent with the updated result
        result = agent_with_chat_history.invoke(
            # {"input": "suggest and execute a next steps based on your processes and the previous outcome: " + new_input},
            {"input": "execute the next step based on your processes and the previous outcome! Finish with suggesting a following next step. This needs to be different from your previous job to avoid getting stuck in a loop."},
            config={"configurable": {"session_id": "<foo>"},"callbacks":[st_callback]}
        )
        with st.chat_message("assistant"):
            st.write(result.get('output'))
        counter += 1

    st.success("Innovation project executed!")

    # Print final result
    # print(result)



