from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client.models import PointStruct
import random, time
from uuid import uuid4
import os

# --- ENV VARS ---
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
OPENAI_API_KEY= os.environ["OPENAI_API_KEY"]

# --- FASTAPI SETUP ---
app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://stacbot-fe.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (ok for dev, not ideal for sensitive APIs)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- MODELS & CLIENTS ---
OPENAI_MODEL = "gpt-4o-mini"
collection_name_openai = "stacA2_collection_1250_225_openAI"
collection_A2_name_offline_feedback = "stacA2_offline_feedback"
collection_m3_name_openai = "stacM3_collection_1250_150_openAI"
collection_m3_name_offline_feedback = "stacM3_offline_feedback"


chat_llm_openai = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0,
    top_p=1,
    max_tokens=750
)

openai_embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

qdrant_client = QdrantClient(
    url="https://58798795-252e-43e2-b815-b0d187fb45b2.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=os.getenv("QDRANT_API_KEY"),
)

# --- Request Schemas ---
class QueryRequest(BaseModel):
    query: str
    chatType: str

class FeedbackRequest(BaseModel):
    user_feedback: str
    previous_question: str = None
    previous_answer: str = None
    chatType: str

class OfflineFeedbackRequest(BaseModel):
    question_asked: str
    answer_received: str
    helpful_feedback: str
    feedbackType: str

# --- /chat endpoint ---
@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    user_query = request.query
    chatType = request.chatType
    
    try:
            # Embed new query
            query_vector = openai_embeddings.embed_query(user_query)

            if chatType.upper()== "A2" :  
                collection_name = collection_name_openai
                collection_feedback = collection_A2_name_offline_feedback
            elif chatType.upper()== "M3":
                collection_name = collection_m3_name_openai
                collection_feedback = collection_m3_name_offline_feedback
            
            print(f"Using collection: {collection_name} and feedback collection: {collection_feedback}")
            
            # Search in the offline feedback collection (limit=3)
            feedback_results = qdrant_client.query_points(
                collection_name=collection_feedback,
                query=query_vector,
                limit=3,
                with_payload=True,
                score_threshold=0.90
            )
            
            print(f"Matched Feedbacks: {feedback_results.points}")
            
            # Collect feedbacks above threshold
            matched_feedbacks = [
                p.payload.get("helpful_feedback")
                for p in feedback_results.points
                if p.payload.get("helpful_feedback")
            ]
         
            
            # If feedback exists, combine with query
            if matched_feedbacks:
                feedback_text = " ".join(matched_feedbacks)
                combined_query = f"{user_query}. {feedback_text}"
                query_vector = openai_embeddings.embed_query(combined_query)
            
            # Normal retrieval from main documents
            retrieved = qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=15,
                with_payload=True
            )

            context_list, docs = [], []
            for point in retrieved.points:
                content = point.payload.get("page_content")
                metadata = point.payload.get("metadata", {})
                if content:
                    context_list.append(content)
                    docs.append(Document(page_content=content, metadata=metadata))
            context_str = "\n\n".join(context_list)

            # Build the final prompt
            if matched_feedbacks:
                feedback_section = "Important Note:\nUsers previously gave the following helpful feedback for similar queries:\n"
                feedback_section += "\n".join([f"- {fb}" for fb in matched_feedbacks])
                feedback_section += "\n\nPlease make sure to include these points if relevant.\n"
            else:
                feedback_section = ""

            final_prompt = f"""
    You are a helpful assistant that answers questions based on the provided context.
    If the information is not found in the context, state that you don't know.
    Please answer in a detailed, numbered list format if possible.

    {feedback_section}

    Context:
    {context_str}

    Question: {user_query}

    Answer:
    """
            # Generate answer
            answer = (
                chat_llm_openai.invoke(final_prompt).content
                if context_str or matched_feedbacks
                else "I couldn't find relevant information in the documents or feedback."
            )

            return {
                "openai_answer": answer,
                "openai_sources": [
                    {"source": doc.metadata.get("source", "Unknown"), "content_excerpt": doc.page_content}
                    for doc in docs
                ],
                "matched_feedbacks": feedback_results.points  # expose all matched feedbacks
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during processing: {e}")


# --- /chat_feedback endpoint ---
@app.post("/chat_feedback")
async def chat_feedback_endpoint(request: FeedbackRequest):
    user_feedback = request.user_feedback
    previous_question = request.previous_question
    previous_answer = request.previous_answer
    chatType = request.chatType

    try:
       
        # Embed new query
        feedback_vector = openai_embeddings.embed_query(user_feedback)

        if chatType.upper()== "A2": 
            collection_name = collection_name_openai
            collection_feedback = collection_A2_name_offline_feedback
        elif chatType.upper()== "M3":
            collection_name = collection_m3_name_openai
            collection_feedback = collection_m3_name_offline_feedback
        
        print(f"Using collection: {collection_name} and feedback collection: {collection_feedback}")
        
        # Search in the offline feedback collection (limit=3)
        feedback_results = qdrant_client.query_points(
            collection_name=collection_feedback,
            query=feedback_vector,
            limit=3,
            with_payload=True,
            score_threshold=0.90
        )

        print(f"Matched Feedbacks: {feedback_results.points}")
        
        # Collect feedbacks above threshold
        matched_feedbacks = [
            p.payload.get("helpful_feedback")
            for p in feedback_results.points
            if p.payload.get("helpful_feedback")
        ]
        
        # If feedback exists, combine with query
        if matched_feedbacks:
            feedback_text = " ".join(matched_feedbacks)
            combined_query = f"{user_feedback}. {feedback_text}"
            feedback_vector = openai_embeddings.embed_query(combined_query)


        # Retrieve context from main collection
        retrieved_feedback = qdrant_client.query_points(
            collection_name=collection_name,
            query=feedback_vector,
            limit=10,
            with_payload=True
        )

        feedback_context = "\n\n".join(
            [p.payload.get("page_content") for p in retrieved_feedback.points if p.payload.get("page_content")]
        )

        # Step 4: Build previous section (if available)
        if previous_question and previous_answer:
            previous_section = f"Previously, you answered this question:\nQuestion: {previous_question}\nAnswer: {previous_answer}"
        else:
            previous_section = ""

        # Step 5: Add offline feedback injection if matched
        if matched_feedbacks:
            feedback_injection = "Important Note:\nUsers previously provided this helpful feedback for similar queries:\n"
            feedback_injection += "\n".join([f"- {fb}" for fb in matched_feedbacks])
            feedback_injection += "\n\nMake sure to incorporate this into your revised answer if relevant.\n"
        else:
            feedback_injection = ""

        # Step 6: Final prompt
        prompt = f"""
You are a helpful assistant.

{previous_section}

The user gave the following feedback:
"{user_feedback}"

{feedback_injection}

Use the context below to improve your answer if needed:

Context:
{feedback_context}

Provide a revised, detailed answer, explaining the points mentioned in the feedback.
"""

        updated_answer = chat_llm_openai.invoke(prompt).content

        return {
            "updated_answer": updated_answer,
            "feedback_context_sources": [
                {"source": p.payload.get("metadata", {}).get("source", "Unknown"),
                 "content_excerpt": p.payload.get("page_content")}
                for p in retrieved_feedback.points
            ],
            "matched_feedbacks": feedback_results.points  # exposing all matched offline feedbacks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during feedback processing: {e}")
    
    
# --- /save_offline_feedback endpoint ---
@app.post("/save_offline_feedback")
async def save_offline_feedback(feedback: dict):
    """
    Save offline feedback with retry mechanism to handle Qdrant rate limits.
    Expects JSON with: question_asked, answer_received, helpful_feedback
    """
    question_asked = feedback.get("question_asked")
    answer_received = feedback.get("answer_received")
    helpful_feedback = feedback.get("helpful_feedback")
    feedbackType = feedback.get("feedbackType")

    if not all([question_asked, helpful_feedback]):
        return {"status": "error", "detail": "All required fields must be filled."}

    if feedbackType.upper()== "A2": 
        collection_name_offline_feedback = collection_A2_name_offline_feedback
    elif feedbackType.upper()== "M3":
        collection_name_offline_feedback = collection_m3_name_offline_feedback
    
    # --- Config ---
    MAX_UPSERT_RETRIES = 10
    INITIAL_UPSERT_DELAY_SECONDS = 5
    MAX_UPSERT_DELAY_SECONDS = 60
    JITTER_RANGE = 2
    MAX_EMBEDDING_RETRIES = 6
    INITIAL_EMBEDDING_DELAY_SECONDS = 6
    MAX_EMBEDDING_DELAY_SECONDS = 120

    # Embed the feedback text
    text_to_embed = f"{question_asked}"
    embedding = None
    for attempt in range(MAX_EMBEDDING_RETRIES):
        try:
            embedding = openai_embeddings.embed_documents([text_to_embed])[0]
            break
        except Exception as e:
            wait_time = min(INITIAL_EMBEDDING_DELAY_SECONDS * (2 ** attempt) + random.uniform(0, JITTER_RANGE),
                            MAX_EMBEDDING_DELAY_SECONDS)
            print(f"Embedding error (Attempt {attempt+1}/{MAX_EMBEDDING_RETRIES}): {e}. Waiting {wait_time:.2f}s")
            time.sleep(wait_time)
    if embedding is None:
        return {"status": "error", "detail": "Failed to generate embedding for feedback."}

    # Prepare point
    point = PointStruct(
        id=str(uuid4()),  # let Qdrant generate an ID
        vector=embedding,
        payload={
            "question_asked": question_asked,
            "answer_received": answer_received,
            "helpful_feedback": helpful_feedback
        }
    )

    # --- Upsert with retries ---
    for attempt in range(MAX_UPSERT_RETRIES):
        try:
            qdrant_client.upsert(
                collection_name=collection_name_offline_feedback,
                points=[point],
                wait=True
            )
            return {"status": "success", "message": "Feedback saved successfully!"}
        except Exception as e:
            delay = min(INITIAL_UPSERT_DELAY_SECONDS * (2 ** attempt) + random.uniform(0, JITTER_RANGE),
                        MAX_UPSERT_DELAY_SECONDS)
            print(f"Upsert error (Attempt {attempt+1}/{MAX_UPSERT_RETRIES}): {e}. Waiting {delay:.2f}s")
            time.sleep(delay)

    return {"status": "error", "detail": "Failed to upsert feedback after multiple retries."}
