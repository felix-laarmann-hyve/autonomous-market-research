import os
import toml

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import HumanMessage, AIMessage, SystemMessage  # Import message classes
from langchain_core.callbacks import BaseCallbackHandler  # Ensure this import is included

import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

class MyCustomHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        st.info("chain started")

st.set_page_config(page_title="HYVE Auto-Market-Research Pipeline", page_icon="bee")

hide_st_style = """
    <style>    
    footer {visibility: hidden;}    
    </style>
"""

st.markdown(hide_st_style, unsafe_allow_html=True)

st_callback = StreamlitCallbackHandler(st.container())

# Load config and set environment variable for Tavily API key
os.environ['TAVILY_API_KEY'] = st.secrets["Tavily"]["api_key"]
os.environ['LANGCHAIN_TRACING_V2'] = st.secrets["Langchain"]["LANGCHAIN_TRACING_V2"]
os.environ['LANGCHAIN_API_KEY'] = st.secrets["Langchain"]["LANGCHAIN_API_KEY"]
os.environ['LANGCHAIN_ENDPOINT'] = st.secrets["Langchain"]["LANGCHAIN_ENDPOINT"]
os.environ['LANGCHAIN_PROJECT'] = st.secrets["Langchain"]["LANGCHAIN_PROJECT"]

# Session state handling for API key
if 'api_key_submitted' not in st.session_state:
    st.session_state['api_key_submitted'] = False

if not st.session_state['api_key_submitted']:
    api_key = st.text_input("Enter your OpenAI API Key:", type="password", value="")
    submit_key = st.button("Submit API Key")

    if submit_key and api_key:
        st.session_state['api_key'] = api_key
        st.session_state['api_key_submitted'] = True
        st.rerun()  # Rerun the app to reflect the state change

if 'api_key' in st.session_state:

    # Initialize LLM and agents
    llm = ChatOpenAI(api_key=st.session_state['api_key'], model="gpt-4o-mini", temperature=0)
    search = TavilySearchResults()
    tools = [search]
    prompt = hub.pull("hwchase17/openai-functions-agent")
    research_agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=research_agent, tools=tools, verbose=True, handle_parsing_errors=True)

    default_system_messages = [
    "You are a capable Innovation Consultant specialized in business models and competitive analysis for products and services. Your task is to guide through the entire competitive analysis process. The user inputs a specific business area, product, or service related to a specific industry or market for which a competitive analysis should be conducted.",
    "Competitive categories: Based on the specified area, develop a detailed, well-structured overview of competitive categories within the target market and segment the market accordingly. Create a clear structure of the identified competition categories with brief descriptions of each category. This structure will serve as a foundation for in-depth competitive analysis, helping identify and segment competitors effectively. Provide sources.",
    "Summarize the competitive categories as a table.",
    "Market analysis: Conduct a deep market analysis for each competition category within the target market to understand current dynamics, forecasted growth, and competitive positioning, summarizing the key trends, growth rates, and market shares, and providing insights for strategic positioning. Provide sources.",
    "Summarize the market analysis as a detailed table. Ensure all information is included in the table for this step.",
    "Name all the competitive categories.",
    "Most relevant categories: Take the three most relevant competitive categories. Explain the rationale behind the decision and provide sources. For the following steps of this analysis, you will focus only on these top three competitive categories.",
    "Insights for most relevant categories: For each category, search for current market reports and studies, extracting relevant data and insights. Provide at least three bullet points per category. Provide sources.",
    "Identification of competitors: Identify and profile the primary competitors most likely to influence the market landscape within each previously defined competition category. List the main competitors per category, including a brief company description and links to their websites. Within each category, list the top 3-5 competitors. Provide a rationale for each selection based on factors such as market share, brand recognition, relevance to the category, and strategic initiatives.",
    "Most influential category: Let's deep dive into the most influential Category. Provide rationale for the choice. For each competitor, include a detailed description covering: core business (primary products or services offered), market position (e.g., market leader, emerging player), and unique selling proposition (USP: key differentiators or strengths). Provide sources.",
    "Description of competitors: For each competitor, include a brief description covering: core business (primary products or services offered), market position (e.g., market leader, emerging player), and unique selling proposition (USP: key differentiators or strengths). Provide the official website URL for each competitor to facilitate direct access for further research and further sources.",
    "Products and services: now, deliver a concise summary of the key products and services provided by each identified competitor. For each competitor, provide a brief overview of their main products and services, focusing on product lines (key products offered, including flagship items or best-sellers), service offerings (core services provided, highlighting any unique or specialized services), and target market (primary customer segments or industries served).",
    "Strategy and business model: Conduct a comprehensive and detailed analysis of each identified competitor, focusing on business model, market strategy, recent developments, market share, introduction of new products or services, investment in R&D, customer base characteristics and financial performance. Provide sources.",
    "Competitors' activities: Evaluate all the competitors' activities. Search online for each competitor's PR and communications (press releases, crisis communications), regulatory activities (patent filings, compliance with regulations), sales strategies (partnership announcements), and financial performance (earnings reports, market share estimates, investment activities). Provide sources.",
    "Competitive advantage: Examine the distinctive features and advantages that all competitors promote to differentiate themselves in the market.",
    "USP analysis: Analyze the unique selling propositions (USPs) of the key competitors.",
    "SWOT Analysis: Conduct a detailed SWOT analysis for each key competitor to gain insights into their strategic positioning and potential vulnerabilities, identifying strengths, weaknesses, opportunities, and threats.",
    "Summary: Summarize the findings and provide an overview of the market structure, competitive situation, and potential market entry barriers.",
    "Recommendations for leveraging opportunities and addressing challenges: Based on the comprehensive market and competitive analysis, develop actionable strategic recommendations to leverage market opportunities, and address potential challenges.",
    "Recommendations for competitive advantages: Derive strategic recommendations on the analysis to achieve competitive advantages.",
    "Executive Summary: Write a comprehensive report including all results summarized for strategic decision-making. Explain the rationale of choosing the key categories. Go through market landscape, key competitors' activities, competitive advantages, and strategic recommendations summarized from the last steps. The recommendation should focus on another company that wants to have more competitive advantage in the landscape. Format in tables where possible. Provide sources."
]


    
    if 'system_messages' not in st.session_state:
        st.session_state['system_messages'] = default_system_messages
        st.session_state['current_prompt_index'] = 0  # Initialize prompt index
        st.session_state['initial_input_added'] = False  # Track if user input was added to the first message
        st.session_state['chat_history'] = []  # Initialize chat history

    chat_history = [
        SystemMessage(content=msg) for msg in st.session_state['system_messages']
    ]
    chat_history.extend([
        HumanMessage(content="You are my innovation expert. I will give you a topic and you will conduct the entire innovation process from user, trend and tech research to ideation and concept creation."),
        AIMessage(content="Now provide your topic so I can dive into the research!")
    ])

    message_history = ChatMessageHistory()
    message_history.messages = chat_history

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    st.title('🐝 HYVE Auto-Market-Research Pipeline')

    # Toggle to show/hide the message editing menu
    if 'edit_menu' not in st.session_state:
        st.session_state['edit_menu'] = False

    if st.button("Setup agent"):
        st.session_state['edit_menu'] = not st.session_state['edit_menu']

    if st.session_state['edit_menu']:
        st.subheader("Edit System Messages")

        for i, message in enumerate(st.session_state['system_messages']):
            st.session_state['system_messages'][i] = st.text_area(f"Message {i + 1}", message, key=f"msg_{i}")

        if st.button("Add New Message"):
            st.session_state['system_messages'].append("")
            st.rerun()  # Rerun the app to reflect the state change

        if st.button("Remove Last Message"):
            if st.session_state['system_messages']:
                st.session_state['system_messages'].pop()
                st.rerun()  # Rerun the app to reflect the state change

        if st.button("Update Messages"):
            st.success("System messages updated successfully!")     
            st.session_state['edit_menu'] = False
            st.rerun()  # Rerun the app to reflect the state change

    # Form to enter the initial challenge
    with st.form('my_form'):
        initial_input = st.text_area('Enter your research field:', 'digital payment solutions')
        submitted = st.form_submit_button('Submit')

    if submitted:
        st.session_state['current_prompt_index'] = 0  # Reset prompt index on new submission
        st.session_state['initial_input'] = initial_input  # Store the initial input
        st.session_state['process_started'] = True  # Indicate that the process should start
        st.session_state['initial_input_added'] = False  # Reset the flag to ensure user input is added only once
        st.session_state['chat_history'] = []  # Reset chat history on new input
        st.rerun()  # Rerun the app to reflect the new input

    # Process prompts one by one if the process has started
    if st.session_state.get('process_started', False):

        # Add the user input to the first system message only once
        if not st.session_state['initial_input_added']:
            st.session_state['system_messages'][0] += f" {st.session_state['initial_input']}"
            st.session_state['initial_input_added'] = True

        message_history = ChatMessageHistory()

        for current_prompt in st.session_state['system_messages']:
            message_history.messages.append(SystemMessage(content=current_prompt))
            
            st.info(f"Executing: {current_prompt}")

            agent_with_chat_history = RunnableWithMessageHistory(
                agent_executor,
                lambda session_id: message_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )

            result = agent_with_chat_history.invoke(
                {"chat_history": message_history.messages, "input": current_prompt},
                config={"configurable": {"session_id": "<foo>"},"callbacks":[st_callback]}
            )

            with st.chat_message("assistant"):
                st.write(result.get('output'))

        st.success("Market research executed!")

else:
    st.warning("Please enter your OpenAI API Key to proceed.")