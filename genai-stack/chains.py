from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import OllamaEmbeddings, SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import List, Any
import hashlib
import uuid
import json
import time

class BaseLogger:
    def info(self, message):
        print(f"INFO: {message}")
    
    def error(self, message):
        print(f"ERROR: {message}")

def load_embedding_model(embedding_model_name: str, logger=BaseLogger(), config={}):
    if embedding_model_name == "ollama":
        embeddings = OllamaEmbeddings(
            base_url=config["ollama_base_url"], model="llama2"
        )
        dimension = 4096
        logger.info("Embedding: Using Ollama")
    elif embedding_model_name == "openai":
        api_key = config.get("openai_api_key")
        if not api_key:
            logger.error("OpenAI API key not found!")
            # Fall back to sentence transformer if no API key
            embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2", cache_folder="./embedding_model"
            )
            dimension = 384
            logger.info("Embedding: Falling back to SentenceTransformer")
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            dimension = 1536
            logger.info("Embedding: Using OpenAI")
    else:
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2", cache_folder="./embedding_model"
        )
        dimension = 384
        logger.info("Embedding: Using SentenceTransformer")
    return embeddings, dimension


def load_llm(llm_name: str, logger=BaseLogger(), config={}):
    api_key = config.get("openai_api_key")
    
    if llm_name == "gpt-4":
        logger.info("LLM: Using GPT-4")
        return ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True, openai_api_key=api_key)
    elif llm_name == "gpt-3.5":
        logger.info("LLM: Using GPT-3.5")
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True, openai_api_key=api_key)
    elif len(llm_name):
        logger.info(f"LLM: Using Ollama: {llm_name}")
        return ChatOllama(
            temperature=0,
            base_url=config["ollama_base_url"],
            model=llm_name,
            streaming=True,
            top_k=10,
            top_p=0.3,
            num_ctx=3072,
        )
    logger.info("LLM: Using GPT-3.5")
    return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True, openai_api_key=api_key)


def configure_llm_only_chain(llm):
    # LLM only response
    template = """
    You are a helpful assistant that helps with answering general questions.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    def generate_llm_output(
        user_input: str, callbacks: List[Any], prompt=chat_prompt
    ) -> str:
        answer = llm(
            prompt.format_prompt(
                text=user_input,
            ).to_messages(),
            callbacks=callbacks,
        ).content
        return {"answer": answer}

    return generate_llm_output


def configure_qa_rag_chain(llm, embeddings, embeddings_store_url, username, password):
    # RAG response
    general_system_template = """ 
    Use the following pieces of context to answer the question at the end.
    The context contains question-answer pairs and their links from Stackoverflow.
    You should prefer information from accepted or more upvoted answers.
    Make sure to rely on information from the answers and not on questions to provide accuate responses.
    When you find particular answer in the context useful, make sure to cite it in the answer using the link.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----
    {summaries}
    ----
    Each answer you generate should contain a section at the end of links to 
    Stackoverflow questions and answers you found useful, which are described under Source value.
    You can only use links to StackOverflow questions that are present in the context and always
    add links to the end of the answer in the style of citations.
    Generate concise answers with references sources section of links to 
    relevant StackOverflow questions only at the end of the answer.
    """
    general_user_template = "Question:```{question}```"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,
    )

    # Vector + Knowledge Graph response
    kg = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=embeddings_store_url,
        username=username,
        password=password,
        database='neo4j',  # neo4j by default
        index_name="stackoverflow",  # vector by default
        text_node_property="body",  # text by default
        retrieval_query="""
    WITH node AS question, score AS similarity
    CALL  { with question
        MATCH (question)<-[:ANSWERS]-(answer)
        WITH answer
        ORDER BY answer.is_accepted DESC, answer.score DESC
        WITH collect(answer)[..2] as answers
        RETURN reduce(str='', answer IN answers | str + 
                '\n### Answer (Accepted: '+ answer.is_accepted +
                ' Score: ' + answer.score+ '): '+  answer.body + '\n') as answerTexts
    } 
    RETURN '##Question: ' + question.title + '\n' + question.body + '\n' 
        + answerTexts AS text, similarity as score, {source: question.link} AS metadata
    ORDER BY similarity ASC // so that best answers are the last
    """,
    )

    kg_qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=kg.as_retriever(search_kwargs={"k": 2}),
        reduce_k_below_max_tokens=False,
        max_tokens_limit=3375,
    )
    return kg_qa

# Fixed function to configure structured RAG chain
def configure_qa_structure_rag_chain(llm, embeddings, embeddings_store_url, username, password):
    # RAG response based on vector search and retrieval of structured chunks
    
    general_system_template = """ 
    You are a customer service agent that helps a customer with answering questions about a service.
    Use the following context to answer the question at the end.
    Make sure not to make any changes to the context if possible when prepare answers so as to provide accuate responses.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----
    {summaries}
    ----
    At the end of each answer you should contain metadata for relevant document in the form of (source, page).
    For example, if context has `metadata`:(source:'docu_url', page:1), you should display ('doc_url',  1).
    """
    general_user_template = "Question:```{question}```"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,
    )

    # Vector + Knowledge Graph response
    kg = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=embeddings_store_url,
        username=username,
        password=password,
        database='neo4j',  # neo4j by default
        index_name="stackoverflow",  # vector by default
        node_label="Embedding",  # embedding node label
        embedding_node_property="value",  # embedding value property
        text_node_property="sentences",  # text by default
        retrieval_query="""
            WITH node AS answerEmb, score 
            ORDER BY score DESC LIMIT 10
            MATCH (answerEmb) <-[:HAS_EMBEDDING]- (answer) -[:HAS_PARENT*]-> (s:Section)
            WITH s, answer, score
            MATCH (d:Document) <-[*]- (s) <-[:HAS_PARENT*]- (chunk:Chunk)
            WITH d, s, answer, chunk, score ORDER BY d.url_hash, s.title, chunk.block_idx ASC
            // 3 - prepare results
            WITH d, s, collect(answer) AS answers, collect(chunk) AS chunks, max(score) AS maxScore
            RETURN {source: d.url, page: chunks[0].page_idx+1, matched_chunk_id: id(answers[0])} AS metadata, 
                reduce(text = "", x IN chunks | text + x.sentences + '.') AS text, maxScore AS score LIMIT 3;
    """,
    )

    kg_qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=kg.as_retriever(search_kwargs={"k": 25}),
        reduce_k_below_max_tokens=False,
        max_tokens_limit=7000,      # gpt-4
    )
    return kg_qa

# Function to extract title and question from text
def extract_title_and_question(text):
    lines = text.strip().split('\n')
    title = ""
    question = ""
    
    in_title = False
    in_question = False
    
    for line in lines:
        if line.startswith("Title:"):
            in_title = True
            in_question = False
            title = line[6:].strip()
        elif line.startswith("Question:"):
            in_title = False
            in_question = True
            question = line[9:].strip()
        elif in_title:
            title += " " + line.strip()
        elif in_question:
            question += " " + line.strip()
    
    return title, question

# Set up Neo4j schema and constraints
def setup_schema_constraints(neo4j_graph):
    # Create constraints for unique IDs
    constraints = [
        "CREATE CONSTRAINT question_id IF NOT EXISTS FOR (q:Question) REQUIRE q.id IS UNIQUE",
        "CREATE CONSTRAINT answer_id IF NOT EXISTS FOR (a:Answer) REQUIRE a.id IS UNIQUE",
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT embedding_id IF NOT EXISTS FOR (e:Embedding) REQUIRE e.id IS UNIQUE"
    ]
    
    for constraint in constraints:
        try:
            neo4j_graph.query(constraint)
        except Exception as e:
            print(f"Error creating constraint: {e}")

# Create vector index for embeddings
def create_vector_index(neo4j_graph, dimension, embeddings):
    # Check if index exists
    index_result = neo4j_graph.query(
        "SHOW INDEXES YIELD name, type WHERE name = 'stackoverflow' RETURN count(*) as count"
    )
    
    if index_result[0]["count"] == 0:
        # Create vector index if it doesn't exist
        try:
            neo4j_graph.query(
                f"""
                CREATE VECTOR INDEX stackoverflow IF NOT EXISTS
                FOR (e:Embedding) 
                ON e.value
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dimension},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """
            )
            print("Vector index created successfully")
        except Exception as e:
            print(f"Error creating vector index: {e}")
    else:
        print("Vector index already exists")

# Add some sample data if none exists
def import_sample_data(neo4j_graph, embeddings):
    # Check if we already have data
    result = neo4j_graph.query("MATCH (q:Question) RETURN count(q) as count")
    if result[0]["count"] > 0:
        print(f"Data already exists: {result[0]['count']} questions found")
        return
    
    # Sample questions and answers
    sample_data = [
        {
            "question": {
                "id": str(uuid.uuid4()),
                "title": "How to connect to Neo4j using Python?",
                "body": "I'm trying to connect to a Neo4j database using Python but facing some issues. What's the proper way to establish a connection?",
                "score": 25,
                "link": "https://stackoverflow.com/questions/12345/neo4j-python-connection"
            },
            "answers": [
                {
                    "id": str(uuid.uuid4()),
                    "body": "You can use the official Neo4j Python driver. First install it with pip: `pip install neo4j`. Then connect using: ```python\nfrom neo4j import GraphDatabase\n\ndriver = GraphDatabase.driver('neo4j://localhost:7687', auth=('neo4j', 'password'))\n```",
                    "score": 42,
                    "is_accepted": "true"
                },
                {
                    "id": str(uuid.uuid4()),
                    "body": "There's also py2neo which is an alternative client library: `pip install py2neo`",
                    "score": 15,
                    "is_accepted": "false"
                }
            ]
        },
        {
            "question": {
                "id": str(uuid.uuid4()),
                "title": "Creating a vector index in Neo4j",
                "body": "How do you create a vector index in Neo4j for semantic search? I'm trying to store embeddings but not sure about the proper syntax.",
                "score": 18,
                "link": "https://stackoverflow.com/questions/54321/neo4j-vector-index"
            },
            "answers": [
                {
                    "id": str(uuid.uuid4()),
                    "body": "For Neo4j 5.11+, you can create a vector index using: ```cypher\nCREATE VECTOR INDEX my_index FOR (n:Embedding) ON n.embedding OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}\n```",
                    "score": 32,
                    "is_accepted": "true"
                }
            ]
        }
    ]
    
    # Create a document to represent StackOverflow
    doc_id = str(uuid.uuid4())
    neo4j_graph.query(
        """
        CREATE (d:Document {
            id: $id, 
            title: 'StackOverflow Questions', 
            url: 'https://stackoverflow.com', 
            url_hash: 'stackoverflow'
        })
        """,
        {"id": doc_id}
    )
    
    # Create a section for Q&A
    section_id = str(uuid.uuid4())
    neo4j_graph.query(
        """
        MATCH (d:Document {id: $doc_id})
        CREATE (s:Section {
            id: $id,
            title: 'Questions and Answers'
        })
        CREATE (s)-[:PART_OF]->(d)
        """,
        {"doc_id": doc_id, "id": section_id}
    )
    
    # Add questions and answers
    for i, qa_pair in enumerate(sample_data):
        q = qa_pair["question"]
        
        # Create question node
        neo4j_graph.query(
            """
            MATCH (s:Section {id: $section_id})
            CREATE (q:Question {
                id: $id,
                title: $title,
                body: $body,
                score: $score,
                link: $link
            })
            CREATE (q)-[:HAS_PARENT]->(s)
            """,
            {
                "section_id": section_id,
                "id": q["id"],
                "title": q["title"],
                "body": q["body"],
                "score": q["score"],
                "link": q["link"]
            }
        )
        
        # Create embedding for question
        q_emb = embeddings.embed_query(q["title"] + " " + q["body"])
        q_emb_id = str(uuid.uuid4())
        
        neo4j_graph.query(
            """
            MATCH (q:Question {id: $q_id})
            CREATE (e:Embedding {
                id: $emb_id,
                value: $embedding
            })
            CREATE (q)-[:HAS_EMBEDDING]->(e)
            """,
            {
                "q_id": q["id"],
                "emb_id": q_emb_id,
                "embedding": q_emb
            }
        )
        
        # Create chunk for question (needed for structure model)
        chunk_id = str(uuid.uuid4())
        neo4j_graph.query(
            """
            MATCH (q:Question {id: $q_id})
            MATCH (s:Section)-[:PART_OF]->(d:Document)
            WHERE (q)-[:HAS_PARENT]->(s)
            CREATE (c:Chunk {
                id: $chunk_id,
                sentences: $sentences,
                block_idx: $block_idx,
                page_idx: $page_idx
            })
            CREATE (c)-[:HAS_PARENT]->(s)
            CREATE (c)-[:FROM_QUESTION]->(q)
            """,
            {
                "q_id": q["id"],
                "chunk_id": chunk_id,
                "sentences": q["title"] + ". " + q["body"],
                "block_idx": i,
                "page_idx": i
            }
        )
        
        # Add answers
        for j, answer in enumerate(qa_pair["answers"]):
            # Create answer node
            neo4j_graph.query(
                """
                MATCH (q:Question {id: $q_id})
                CREATE (a:Answer {
                    id: $id,
                    body: $body,
                    score: $score,
                    is_accepted: $is_accepted
                })
                CREATE (a)-[:ANSWERS]->(q)
                """,
                {
                    "q_id": q["id"],
                    "id": answer["id"],
                    "body": answer["body"],
                    "score": answer["score"],
                    "is_accepted": answer["is_accepted"]
                }
            )
            
            # Create embedding for answer
            a_emb = embeddings.embed_query(answer["body"])
            a_emb_id = str(uuid.uuid4())
            
            neo4j_graph.query(
                """
                MATCH (a:Answer {id: $a_id})
                CREATE (e:Embedding {
                    id: $emb_id,
                    value: $embedding
                })
                CREATE (a)-[:HAS_EMBEDDING]->(e)
                """,
                {
                    "a_id": answer["id"],
                    "emb_id": a_emb_id,
                    "embedding": a_emb
                }
            )
            
            # Create chunk for answer
            ans_chunk_id = str(uuid.uuid4())
            neo4j_graph.query(
                """
                MATCH (a:Answer {id: $a_id})
                MATCH (q:Question)<-[:ANSWERS]-(a)
                MATCH (s:Section)<-[:HAS_PARENT]-(q)
                CREATE (c:Chunk {
                    id: $chunk_id,
                    sentences: $sentences,
                    block_idx: $block_idx + $j,
                    page_idx: $page_idx
                })
                CREATE (c)-[:HAS_PARENT]->(s)
                CREATE (c)-[:FROM_ANSWER]->(a)
                """,
                {
                    "a_id": answer["id"],
                    "chunk_id": ans_chunk_id,
                    "sentences": answer["body"],
                    "block_idx": (i * 10) + j + 1,  # Ensure unique block indexes
                    "page_idx": i,
                    "j": j
                }
            )

    print(f"Imported {len(sample_data)} questions with their answers")

            # Utility function to get embedding for text
def get_text_embedding(text, embeddings):
                return embeddings.embed_query(text)
