from django.shortcuts import render
import os
from .models import Patient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from pydantic import BaseModel,  Field
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from typing import Tuple, List, Optional
from langchain_core.documents import Document
from langchain.output_parsers import StructuredOutputParser
from .retriever import retrieve_relevant_docs
from .entity_extractor import entities
from .summary import summarize_conversation

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
NEO4J_URI= os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
AURA_INSTANCEID = os.environ.get("AURA_INSTANCEID")
AURA_INSTANCENAME= os.environ.get("AURA_INSTANCENAME")


def fetch_patient_data(patient_id):
    try:
        patient = Patient.objects.get(id=patient_id)
        patient_data = {
            "name": f"{patient.first_name} {patient.last_name}",
            "condition": patient.medical_condition,
            "medication": patient.medication_regimen,
            "last_appointment": patient.last_appointment,
            "next_appointment": patient.next_appointment,
            "doctor_name": patient.doctor_name
        }
        return patient_data
    except Patient.DoesNotExist:
        return None

# testing 
# patient_id = 1  
# patient_data = fetch_patient_data(patient_id)
# query="When is my appointment?"
# relevant_docs = retrieve_relevant_docs(patient_data,query)
# print(relevant_docs)

def chat_view(request):
    patient_id = 1  # Assuming you're using a specific patient ID for now
    patient_data = fetch_patient_data(patient_id)
    patient = Patient.objects.first()
    
    conversation = request.session.get('conversation', [])  # Retrieve conversation from session

    if request.method == "POST":
        user_message = request.POST.get("message")
        bot_response = retrieve_relevant_docs(patient_data, user_message)

        # Append the new message and bot response to the conversation
        conversation.append((user_message, bot_response))
        request.session['conversation'] = conversation  # Save to session
        summary = summarize_conversation(conversation)
        print(summary)
        extracted_entities = entities(str(conversation), user_message)
        print(extracted_entities)
    context = {
        'patient': patient,
        'conversation': conversation,
    }
    return render(request, 'chat.html', context)
