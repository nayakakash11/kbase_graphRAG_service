import os

import streamlit as st
from streamlit.logger import get_logger
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.graphs import Neo4jGraph
from dotenv import load_dotenv
from utils import (
    extract_title_and_question,
)


# >>>> initialise - environment <<<< 


openai_api_key = 0 #Enter your OpenAI API key here

url = 0 #Neo4j URL
username = 0 #Enter your Neo4j username
password = 0 #Enter your Neo4j password
database = 0 #Enter your Neo4j database name
ollama_base_url = 0 #Enter your ollama base url (local url)
embedding_model_name = 0 #Enter your embedding model name
llm_name = 0 #Enter your LLM model name

config = {
    "ollama_base_url": ollama_base_url,
    "openai_api_key": openai_api_key
}


# >>>> initialise - services <<<< 

logger = get_logger(__name__)

neo4j_graph = Neo4jGraph(url=url, username=username, password=password, database=database)

# Import these after setting up environment variables
from chains import (
    load_embedding_model,
    load_llm,
    configure_llm_only_chain,
    configure_qa_rag_chain,
    configure_qa_structure_rag_chain,
    create_vector_index,
    setup_schema_constraints,
    import_sample_data,
)

embeddings, dimension = load_embedding_model(
    embedding_model_name, 
    config=config, 
    logger=logger
)

llm = load_llm(llm_name, logger=logger, config=config)

# llm_chain: LLM only response
llm_chain = configure_llm_only_chain(llm)

# Set up Neo4j schema and constraints
setup_schema_constraints(neo4j_graph)

# Create vector index and import sample data if needed
create_vector_index(neo4j_graph, dimension, embeddings)

# Import some sample data (questions/answers) if none exists
import_sample_data(neo4j_graph, embeddings)

# rag_chain: KG augmented response
rag_chain = configure_qa_structure_rag_chain(
    llm, embeddings, embeddings_store_url=url, username=username, password=password
)


# >>>> Class definition - StreamHander <<<< 

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# >>>> Streamlit UI <<<<

styl = f"""
<style>
    /* not great support for :has yet (hello FireFox), but using it for now */
    .element-container:has([aria-label="Select RAG mode"]) {{
      position: fixed;
      bottom: 33px;
      background: white;
      z-index: 101;
    }}
    .stChatFloatingInputContainer {{
        bottom: 20px;
    }}

    /* Generate question text area */
    textarea[aria-label="Description"] {{
        height: 200px;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)
st.image("/Users/adityaraj/Aditya/IIIT/3rd Year/6th Sem/Cloud Computing/Project/genai-stack/images/qna-logo.png", width=160) 

# >>>> UI interations <<<<

def chat_input():
    user_input = st.chat_input("What service questions can I help you resolve today?")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            st.caption(f"RAG: {name}")
            stream_handler = StreamHandler(st.empty())

            # Call chain to generate answers
            result = output_function(
                {"question": user_input, "chat_history": []}, callbacks=[stream_handler]
            )["answer"]

            output = result

            st.session_state[f"user_input"].append(user_input)
            st.session_state[f"generated"].append(output)
            st.session_state[f"rag_mode"].append(name)


def display_chat():
    # Session state
    if "generated" not in st.session_state:
        st.session_state[f"generated"] = []

    if "user_input" not in st.session_state:
        st.session_state[f"user_input"] = []

    if "rag_mode" not in st.session_state:
        st.session_state[f"rag_mode"] = []

    if st.session_state[f"generated"]:
        size = len(st.session_state[f"generated"])
        # Display only the last three exchanges
        for i in range(max(size - 3, 0), size):
            with st.chat_message("user"):
                st.write(st.session_state[f"user_input"][i])

            with st.chat_message("assistant"):
                st.caption(f"RAG: {st.session_state[f'rag_mode'][i]}")
                st.write(st.session_state[f"generated"][i])

        with st.expander("Not finding what you're looking for?"):
            st.write(
                "Automatically generate a draft for an internal ticket to our support team."
            )
            st.button(
                "Generate ticket",
                type="primary",
                key="show_ticket",
                on_click=open_sidebar,
            )
        with st.container():
            st.write("&nbsp;")


def mode_select() -> str:
    options = ["Disabled", "Enabled"]
    return st.radio("Select RAG mode", options, horizontal=True)

# >>>>> switch on/off RAG mode

name = mode_select()
if name == "LLM only" or name == "Disabled":
    output_function = llm_chain
elif name == "Vector + Graph" or name == "Enabled":
    output_function = rag_chain


def generate_ticket():
    # Get high ranked questions
    records = neo4j_graph.query(
        "MATCH (q:Question) RETURN q.title AS title, q.body AS body ORDER BY q.score DESC LIMIT 3"
    )
    questions = []
    for i, question in enumerate(records, start=1):
        questions.append((question["title"], question["body"]))
    # Ask LLM to generate new question in the same style
    questions_prompt = ""
    for i, question in enumerate(questions, start=1):
        questions_prompt += f"{i}. {question[0]}\n"
        questions_prompt += f"{question[1]}\n\n"
        questions_prompt += "----\n\n"

    gen_system_template = f"""
    You're an expert in formulating high quality questions. 
    Can you formulate a question in the same style, detail and tone as the following example questions?
    {questions_prompt}
    ---

    Don't make anything up, only use information in the following question.
    Return a title for the question, and the question post itself.

    Return example:
    ---
    Title: How do I use the Neo4j Python driver?
    Question: I'm trying to connect to Neo4j using the Python driver, but I'm getting an error.
    ---
    """
    # we need jinja2 since the questions themselves contain curly braces
    system_prompt = SystemMessagePromptTemplate.from_template(
        gen_system_template, template_format="jinja2"
    )
    q_prompt = st.session_state[f"user_input"][-1]
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            SystemMessagePromptTemplate.from_template(
                """
                Respond in the following format or you will be unplugged.
                ---
                Title: New title
                Question: New question
                ---
                """
            ),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )
    llm_response = llm_chain(
        f"Here's the question to rewrite in the expected format: ```{q_prompt}```",
        [],
        chat_prompt,
    )
    new_title, new_question = extract_title_and_question(llm_response["answer"])
    return (new_title, new_question)


def open_sidebar():
    st.session_state.open_sidebar = True


def close_sidebar():
    st.session_state.open_sidebar = False


if not "open_sidebar" in st.session_state:
    st.session_state.open_sidebar = False
if st.session_state.open_sidebar:
    new_title, new_question = generate_ticket()
    with st.sidebar:
        st.title("Ticket draft")
        st.write("Auto generated draft ticket")
        st.text_input("Title", new_title)
        st.text_area("Description", new_question)
        st.button(
            "Submit to support team",
            type="primary",
            key="submit_ticket",
            on_click=close_sidebar,
        )

# >>>> UI: show chat <<<<
display_chat()
chat_input()
