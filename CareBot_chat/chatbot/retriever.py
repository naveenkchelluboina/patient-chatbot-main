import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.documents import Document
from typing import Tuple, List, Optional
from langchain.chains import RetrievalQA

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0, api_key= OPENAI_API_KEY,   model="gpt-4o-mini",max_tokens=900)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

NEO4J_URI= os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
AURA_INSTANCEID = os.environ.get("AURA_INSTANCEID")
AURA_INSTANCENAME= os.environ.get("AURA_INSTANCENAME")

graph = Neo4jGraph()

def generate_docs_from_patient_data(patient_data: dict) -> List[Document]:
    """
    Converts patient data into a Document object suitable for use with vector indexing or document retrieval systems.
    """
    # Construct a summary from patient data
    page_content = (
        f"{patient_data['name']} has been diagnosed with {patient_data['condition']}. "
        f"Medications include {patient_data['medication']}. "
        f"The last appointment was on {patient_data['last_appointment'].strftime('%Y-%m-%d %H:%M:%S %Z')}, "
        f"and the next appointment is scheduled for {patient_data['next_appointment'].strftime('%Y-%m-%d %H:%M:%S %Z')}. "
        f"Doctor: {patient_data['doctor_name']}."
    )
    
    # Prepare the metadata
    metadata = {
        'name': patient_data['name'],
        'condition': patient_data['condition'],
        'medication': patient_data['medication'],
        'last_appointment': patient_data['last_appointment'],
        'next_appointment': patient_data['next_appointment'],
        'doctor_name': patient_data['doctor_name']
    }
    
    # Create Document object
    doc = Document(page_content=page_content, metadata=metadata)
    
    return [doc]

def retrieve_relevant_docs(patient_data, query):
    
    docs = generate_docs_from_patient_data(patient_data)

    metadata_field_info = [
        AttributeInfo(
            name="name",
            description="The patient's name",
            type="string",
        ),
        AttributeInfo(
            name="condition",
            description="The patient's diagnosed medical condition",
            type="string",
        ),
        AttributeInfo(
            name="medication",
            description="The medication prescribed to the patient",
            type="string",
        ),
        AttributeInfo(
            name="last_appointment",
            description="The date and time of the patient's last appointment",
            type="datetime",
        ),
        AttributeInfo(
            name="next_appointment",
            description="The date and time of the patient's next appointment",
            type="datetime",
        ),
        AttributeInfo(
            name="doctor_name",
            description="The name of the patient's doctor",
            type="string",
        ),
    ]
    document_content_description = "Medical summary of the patient, including condition, medications, and appointment details."
    vector_index = Neo4jVector.from_documents(docs, embeddings)
    retriever = SelfQueryRetriever.from_llm(
        llm, vector_index,document_content_description , metadata_field_info, verbose=True
    )

    prompt_rules = f"""
    You are a medical assistant bot for the patient {patient_data['name']}. You should only respond to health-related topics such as:
    - Patient Profile
    - General health and lifestyle inquiries
    - Questions about the patient’s medical condition, medication regimen, diet, etc, If the patient reports a new symptom, tell the patient that you will convey it to  Dr. {patient_data['doctor_name'] if 'doctor_name' in patient_data else 'your doctor'}.
    - Requests from the patient to their doctor such as medication changes
    
    Please follow these specific rules:
    - Ignore any unrelated, sensitive, or controversial topics.
    - Give the answer in a meaningful way, avoidding special charecters for no reason
    - If the patient requests an appointment date change of the next appointment, respond by saying, "I will convey your request to {patient_data['doctor_name'] if 'doctor_name' in patient_data else 'your doctor'}." Additionally, output a message saying, "- Patient {patient_data['name']} is requesting an appointment change." and log this request for review.
    - For any other questions that requires the doctors involvement, tell them to contact the doctor. Ask - "Do you want me to ask the doctor for an appointment?"
    
    Patient Query: "{query}"
    """
    # response = retriever.invoke(query)
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 'stuff' adds relevant content to the prompt for OpenAI to process
    retriever=retriever)
    response = qa_chain.run(prompt_rules)

    return response