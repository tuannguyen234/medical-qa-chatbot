import streamlit as st
import streamlit as st
import sqlite3
from google import genai as genai2
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from PIL import Image
import faiss
import pickle
import io
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit UI
st.title("🧠 Multi-RAG System for Medical Q&A")

# User input (text or image)
query_text = None
uploaded_image = st.file_uploader("📷 Upload an image or enter text below:", type=["jpg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write(f"🔍 Extracted Text: **{query_text}**")

query_text_input = st.text_input("💬 Or enter a medical question:")
@st.cache_resource
def embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name="TuanNM171284/TuanNM171284-HaLong-embedding-medical")
    return embeddings

@st.cache_resource
def get_conversational_chain_for_text():

    prompt_template = """
    Bạn là một trợ lý y tế chuyên sâu, sử dụng hệ thống Multi-RAG để tìm kiếm và truy xuất thông tin từ cơ sở dữ liệu y khoa bao gồm văn bản và hình ảnh về triệu chứng, bệnh tật, và phương pháp điều trị. Bạn chỉ dựa vào dữ liệu trong hệ thống để trả lời. Nếu không tìm thấy thông tin phù hợp, hãy trả lời rằng câu hỏi không liên quan hoặc không có trong hệ thống\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """


    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain
# @st.cache_resource
def user_input(user_question):
    # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # embeddings = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")
    # embeddings = HuggingFaceEmbeddings(model_name="TuanNM171284/TuanNM171284-HaLong-embedding-medical")
    new_db = FAISS.load_local(r"D:\code\RAG\faiss_VN_sbert", embedding_model(), allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question,1)
    print("-------------")
    print(type(docs))
    print(docs)
    chain = get_conversational_chain_for_text()

    if user_question is not None:
        response = chain(
            {"input_documents":docs, "question": user_question}
            , return_only_outputs=True)

        st.markdown(f"**Kết quả:**\n\n{response['output_text']}")

# @st.cache_resource
def user_input_images(query):
    # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # embeddings = HuggingFaceEmbeddings(model_name="TuanNM171284/TuanNM171284-HaLong-embedding-medical")
    
    # Load FAISS index
    vector_store = FAISS.load_local(r"D:\code\RAG\faiss_index_image", embedding_model(), allow_dangerous_deserialization=True)
    
    # Get top K results
    results = vector_store.similarity_search(query)
    print(type(results[0]))
    print(results[0])
    if results:
        matched_descriptions = [r.page_content for r in results]  # Extract descriptions
        # Connect to SQLite & retrieve matching image paths
        conn = sqlite3.connect('image_database.db')
        cursor = conn.cursor()
        
        placeholders = ",".join(["?"] * len(matched_descriptions))  # Create ?,?,? for SQL query
        cursor.execute(f"SELECT image_path FROM images WHERE description IN ({placeholders})", matched_descriptions)
        
        image_paths = [row[0] for row in cursor.fetchall()]  # Fetch all results
        conn.close()

        # return image_paths if image_paths else None
        prompt_template = """
        dựa vào câu hỏi người dùng trả loại đường dẫn ảnh có liên quan không được trùng lập và chỉ trả lại đường dẫn ảnh cách nhau bởi dấu \n hoặc dòng mớimới không kèm thông tin gì khác
        """
        client = genai2.Client(api_key="AIzaSyC60YzCQ4IndhZd2_qdVn5a1dzUzZ56kxI")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[f'{prompt_template} + câu hỏi của người dùng:{query} + đường dẫn ảnh:{image_paths}']
    )
        list_images = [i for i in response.text.split('\n') if i.strip()]
        # Hiển thị từng ảnh trong danh sách
        for image in list_images:
            st.image(image, caption=image, use_container_width=True)
if __name__ == "__main__":
    if query_text_input != '':
        user_input(query_text_input)
        user_input_images(query_text_input)
        print("Query Text Input:", query_text_input)
        hello = "Hello, World!"
        st.write(hello)
        print("Query Text Input:", query_text_input)
