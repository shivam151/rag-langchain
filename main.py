# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import Chroma
# import os

# os.environ["GOOGLE_API_KEY"] = "AIzaSyACTF_3hRHXKaDxweKQwQFEa8BLGmuFdic"  

# try:
#     loader = PyPDFLoader("./Cardio-Sample-Report.pdf") 
#     docs = loader.load()
    
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     splits = text_splitter.split_documents(docs)
    
#     print(f"Loaded {len(docs)} documents and split into {len(splits)} chunks")
    
# except Exception as e:
#     print(f"Error loading documents: {e}")
#     raise

# try:
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
#     # Create Chroma vector store from documents
#     vectorstore = Chroma.from_documents(
#         documents=splits,
#         embedding=embeddings,
#         persist_directory="./chroma_db" 
#     )
    
#     print("Created Chroma vector store")
    
# except Exception as e:
#     print(f"Error creating Chroma vector store: {e}")
#     raise

# # 4. Create a retriever from the Chroma vector store
# retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
# print(retriever,"retriever")

# # 5. Initialize Gemini LLM
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# # 6. Create prompt template
# template = """Answer the question based on the following context. 
# If you don't know the answer, just say that you don't know.
# Keep the answer concise and to the point.

# Context: {context}

# Question: {question}

# Answer:"""
# prompt = ChatPromptTemplate.from_template(template)

# # 7. Create RAG chain
# def rag_query(question):
#     # Retrieve relevant documents
#     docs = retriever.get_relevant_documents(question)
#     context = "\n\n".join([doc.page_content for doc in docs])
    
#     # Create the chain
#     chain = (
#         {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
    
#     # Run the chain
#     response = chain.invoke({"context": context, "question": question})
#     return response

# # 8. Query the system
# try:
#     questions = [
#         "What is the main topic of the document?",
#         "What are the key findings?",
#         "Summarize the document in 3 bullet points",
#         "what is the name in the report"
#     ]
#     for question in questions:
#         print(f"\nQuestion: {question}")
#         response = rag_query(question)
#         print(f"Response: {response}")
        
# except Exception as e:
#     print(f"Error querying the system: {e}")
#     raise



# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import FAISS  # Changed from Chroma to FAISS
# import os

# os.environ["GOOGLE_API_KEY"] = "AIzaSyACTF_3hRHXKaDxweKQwQFEa8BLGmuFdic"  

# try:
#     loader = PyPDFLoader("./Cardio-Sample-Report.pdf") 
#     docs = loader.load()
    
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     splits = text_splitter.split_documents(docs)
    
#     print(f"Loaded {len(docs)} documents and split into {len(splits)} chunks")
    
# except Exception as e:
#     print(f"Error loading documents: {e}")
#     raise

# try:
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
#     # Create FAISS vector store from documents
#     vectorstore = FAISS.from_documents(
#         documents=splits,
#         embedding=embeddings
#     )
    
#     # Save the FAISS index locally
#     vectorstore.save_local("faiss_index")
#     print("Created FAISS vector store and saved locally")
    
# except Exception as e:
#     print(f"Error creating FAISS vector store: {e}")
#     raise

# # 4. Create a retriever from the FAISS vector store
# retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
# print(retriever,"retriever")

# # 5. Initialize Gemini LLM
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# # 6. Create prompt template
# template = """Answer the question based on the following context. 
# If you don't know the answer, just say that you don't know.
# Keep the answer concise and to the point.

# Context: {context}

# Question: {question}

# Answer:"""
# prompt = ChatPromptTemplate.from_template(template)

# # 7. Create RAG chain
# def rag_query(question):
#     # Retrieve relevant documents
#     docs = retriever.get_relevant_documents(question)
#     context = "\n\n".join([doc.page_content for doc in docs])
    
#     # Create the chain
#     chain = (
#         {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
    
#     # Run the chain
#     response = chain.invoke({"context": context, "question": question})
#     return response

# # 8. Query the system
# try:
#     questions = [
#         "What is the main topic of the document?",
#         "What are the key findings?",
#         "Summarize the document in 3 bullet points",
#         "what is the name in the report"
#     ]
#     for question in questions:
#         print(f"\nQuestion: {question}")
#         response = rag_query(question)
#         print(f"Response: {response}")
        
# except Exception as e:
#     print(f"Error querying the system: {e}")
#     raise





