# AI Chatbot using Neo4j and Langchain with Django Backend

Django application where a patient (user) can interact with an AI bot regarding their health and care plan. The AI bot is designed to handle health-related conversations and detect patient requests for changes to their treatment or appointments, while filtering out irrelevant or sensitive topics.
The focus of this project is to allow a patient to chat with an AI bot that can respond to health-related inquiries, provide answers about their care plan, and escalate specific requests(like changes in appointments or treatment protocols) to their doctor.

## Getting Started

Follow the instructions below for set up and run.

#### Prerequisites

- Python 3.10 installed.
- necessary Python packages installed from requirements.txt.

```python
python -m venv venv
./venv/Scripts/activate/
pip install -r requirements.txt
```

## Setup

1. Create a neo4j account - https://neo4j.com/product/auradb/
2. Create an instance.
3. Create a `.env` file in the root directory and add the following environment variables:

```bash
  OPENAI_API_KEY=your_openai_api_key
  NEO4J_URI=neo4j+s://<id>.databases.neo4j.io
  NEO4J_USERNAME=neo4j
  NEO4J_PASSWORD= "instance password"
  AURA_INSTANCEID="instance id"
  AURA_INSTANCENAME="instance name"
```

### Django setup

Create a database with the name written in `settings.py` - `rag_chatbot` in this case

```bash
DATABASES = {
    'default': {
        # 'ENGINE': 'django.db.backends.sqlite3',
        # 'NAME': BASE_DIR / 'db.sqlite3',
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'rag_chatbot',
        'USER': 'username',
        'PASSWORD': 'password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

run the commands -

```python
cd CareBot
python manage.py makemigrations
python manage.py migrate
```

Create superuser

```python
python manage.py createsuperuser
```

#### Create a patient

On `http://localhost:8000/admin`.
Add a Patient.

```python
python manage.py runserver
```

The chat will be opened at - `http://localhost:8000/chat/`

The code uses `langchain_openai`s ChatOpenAI and Neo4j RAG graph.

### File explanations

`CareBot/chatbot/summary.py/` contains code to generate summary of the conversation in real time. Prints on the console on the frontend. \
`CareBot/chatbot/entity_extractor.py/` contains code to extract entities and add to the knowledge graph in realtime. The entities after each query will also be displayed on console. \
`CareBot/chatbot/retriever.py/` contains code to to generate response for a users query using RAG.
