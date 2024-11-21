from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Tuple, List, Optional
import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
llm=ChatOpenAI(temperature=0, model="gpt-4o-mini") 

NEO4J_URI= os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
AURA_INSTANCEID = os.environ.get("AURA_INSTANCEID")
AURA_INSTANCENAME= os.environ.get("AURA_INSTANCENAME")
graph = Neo4jGraph()

def entities(conversation, query):
    documents = [Document(page_content=conversation)]
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )
    # vector_index = Neo4jVector.from_existing_graph(
    # OpenAIEmbeddings(model="text-embedding-3-small"),
    # search_type="hybrid",
    # node_label="Document",
    # text_node_properties=["text"],
    # embedding_node_property="embedding"
    # )
    graph.query(
    "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

    # Extract entities from text
    class Entities(BaseModel):
        """Identifying information about entities."""

        names: List[str] = Field(
            ...,
            description="All the person, Doctors, medicines, frequency of medicines, hospital names, treatment names, treatment frequency that "
            "appear in the text are related to the patient in some way",
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are extracting Doctors name, medical conditions, organization names, medicine names and person entities from the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )

    entity_chain = prompt | llm.with_structured_output(Entities)
    response = entity_chain.invoke(query).names
    return response
