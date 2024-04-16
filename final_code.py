import os
import tempfile
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import CompositeElement, Table
from langchain.schema import Document
import streamlit as st
import cohere
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import uuid
import cohere
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chat_models import cohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.retrievers import MultiVectorRetriever
from langchain.prompts import ChatPromptTemplate

import uuid

from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma

from langchain.schema.runnable import RunnablePassthrough


from langchain.schema.messages import HumanMessage, SystemMessage
import os

import base64
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import pathlib
from langchain_core.tracers.context import tracing_v2_enabled
from PIL import Image
import pandas as pd
import random
import string




os.environ['OPENAI_API_KEY'] = ""
os.environ['GOOGLE_API_KEY'] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Research_assistant"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ""  # Update to your API key


st.set_page_config(page_title="InfoGenie",layout="wide")
st.write("""
    <style>
        footer {visibility: hidden;}
        @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600;700&display=swap');
        body {
            font-family: 'Open Sans', sans-serif;
            background-color: #212529;
            color: #FFFFFF;
        }
    </style>
""", unsafe_allow_html=True)
 

st.header('InfoGenie',divider='rainbow')
st.write("Empower your market research journey with Info Genie: Your intuitive partner for analyzing, summarizing, and answering your specific questions on capital market documents")
bullet_point = "◇"



with st.sidebar:
    label = "Enter URL"
    web_url = st.text_input(label,label_visibility='visible',value='https://www.capitalmarket.com/')
    if 'input_keys' not in st.session_state:
        st.session_state.input_keys= []
    if st.button("Add new URL"):
        st.session_state.input_keys.append(random.choice(string.ascii_uppercase)+str(random.randint(0,999999)))
    input_values = []
    for input_key in st.session_state.input_keys:
        label = "Enter URL"
        input_value = st.text_input(label,label_visibility='visible',value='https://www.capitalmarket.com/', key=input_key)
        input_values.append(input_value)

    uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)  


# Use unstructured to extract text, tables and images data from files.
def add_document(file):  
    if file.type != 'application/pdf':
        raise ValueError("Unsupported file type. Only PDFs are allowed.")

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_file_path = os.path.join(temp_dir, file.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.read())

        elements = partition_pdf(
            filename=temp_file_path,
            chunking_strategy="by_title", 
            infer_table_structure=True,
            extract_images_in_pdf=True,
            strategy='hi_res',
            max_characters=3200,
            new_after_n_chars=3000,
            combine_text_under_n_chars=2200,
            image_output_dir_path="./" # Use images_folder path
        )
        return elements

 

if uploaded_files:
    element_list=[]
    st.title("Extraction process:-")
    st.write(f"{bullet_point} Extraction process started")
    for file in uploaded_files: 
        elements=add_document(file) 
        for ele in elements:
            element_list.append(ele)
    st.write(f"{bullet_point} Extraction process completed")

    # category_counts = {}
    # for element in element_list:
    #     category = str(type(element))
    #     if category in category_counts:
    #         category_counts[category] += 1
    #     else:
    #         category_counts[category] = 1

    # st.header('Below is summary of category count')
    # st.write(category_counts)

    summary_prompt = """
        Summarize the following {element_type}:
        {element}
        """

    summary_chain = LLMChain(llm=ChatOpenAI(model="gpt-3.5-turbo-0125", max_tokens=1024),prompt=PromptTemplate.from_template(summary_prompt))

    #Segregate extracted data into texts and tables.Generate summary of texts and tables data
    def generate_summary(element_list):

        text_elements = []
        table_elements = []
        text_summaries = []
        table_summaries = []

        for element in element_list:
            if 'CompositeElement' in repr(element):
                text_elements.append(element.text)
                summary = summary_chain.run({'element_type': 'text', 'element': element})
                text_summaries.append(summary)

            elif 'Table' in repr(element):
                table_elements.append(element.text)
                summary = summary_chain.run({'element_type': 'table', 'element': element})
                table_summaries.append(summary)        
        
        tables = [i for i in table_elements]
        texts = [i for i in text_elements]  

        return texts,tables,text_summaries,table_summaries
    
    st.title("Summary generation process:-")
    st.write(f"{bullet_point} Summary generation process started")
    texts,tables,text_summaries,table_summaries=generate_summary(element_list)

    if "texts" not in st.session_state or "tables" not in st.session_state:
        # Create session state variables
        with st.spinner("Generating Text & Table summaries....."):
            texts, tables, text_summaries, table_summaries = generate_summary(element_list)
        st.session_state["texts"] = texts
        st.session_state["tables"] = tables
        st.session_state["text_summaries"] = text_summaries
        st.session_state["table_summaries"] = table_summaries
    else:
        # Use already populated session state variables
        texts = st.session_state["texts"]
        tables = st.session_state["tables"]
        text_summaries = st.session_state["text_summaries"]
        table_summaries = st.session_state["table_summaries"]
    st.write(f"{bullet_point} \t\tText & Table summaries generation completed")    
       
  

    if 'vectorstore' not in st.session_state:
        vectorstore = Chroma(collection_name="aberd133",embedding_function=OpenAIEmbeddings(openai_api_key="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"))
        st.session_state["vectorstore"] = vectorstore
    else:
        vectorstore = st.session_state["vectorstore"]

    # st.write(tables)
    # st.write(table_summaries)

    # st.write(texts)
    # st.write(text_summaries)

    #Encode images
    def encode_image(image_path):
     with open(image_path, "rb") as f:
         return base64.b64encode(f.read()).decode('utf-8')

    #Get Images summary. by default images are stored in figures folder in local.
    def summarize_image(encoded_image):
        prompt = [
            SystemMessage(content="You are an assistant tasked with summarizing images for retrieval. \
                These summaries will be embedded and used to retrieve the raw image. \
                Give a concise summary of the image that is well optimized for retrieval."),
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": "Describe the contents of this image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    },
                },
            ])
        ]
        response = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024).invoke(prompt)
        # response = ChatGoogleGenerativeAI(model="gemini-pro-vision",google_api_key="AIzaSyC5bAj2EV6itXLJa_tUwTOKIBocLaGaR2w",convert_system_message_to_human=True).invoke(prompt)
        return response.content
    
    image_elements = []
    image_summaries = []

    if 'image_elements' not in st.session_state:
        with st.spinner("Generating Images summaries......"):
            for i in os.listdir('./figures'):
                if i.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join('./figures', i)
                    encoded_image = encode_image(image_path)
                    image_elements.append(encoded_image)
                    summary = summarize_image(encoded_image)
                    image_summaries.append(summary)
        st.session_state["image_elements"] = image_elements
        st.session_state["image_summaries"] = image_summaries
    else:
        image_elements = st.session_state["image_elements"]  
        image_summaries = st.session_state["image_summaries"]  

    st.write(f"{bullet_point} \t\tImage summaries generation completed") 
    st.write(f"{bullet_point} Summary generation process completed")   
   
    #Create MultiVectorRetreiver to store orignal documents with summary
    def multi_vector_retreiver(vectorstore,texts,tables,text_summaries,table_summaries,image_elements,image_summaries):
       
        store = InMemoryStore()
        id_key = "id"

        # Create MultiVectorRetriever
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )

        # Add texts
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(text_summaries)
        ]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, texts)))

        # Add tables
        table_ids = [str(uuid.uuid4()) for _ in tables]
        summary_tables = [
            Document(page_content=s, metadata={id_key: table_ids[i]})
            for i, s in enumerate(table_summaries)
        ]  

        retriever.vectorstore.add_documents(summary_tables)
        retriever.docstore.mset(list(zip(table_ids, tables)))

        # Add Images
        image_ids = [str(uuid.uuid4()) for _ in image_elements]
        summary_images = [
        Document(page_content=s, metadata={id_key: image_ids[i]})
        for i, s in enumerate(image_summaries)
        ]
        retriever.vectorstore.add_documents(summary_images)
        retriever.docstore.mset(list(zip(image_ids, image_summaries)))

        return retriever    

   
    # vectorstore = Chroma(collection_name="aberd133",embedding_function=OpenAIEmbeddings(openai_api_key="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"))
    # retriever=multi_vector_retreiver(vectorstore,texts,tables,text_summaries,table_summaries,image_elements,image_summaries)

     # Create Vectore store
    if 'vectorstore' not in st.session_state:
        vectorstore = Chroma(collection_name="aberd140",embedding_function=OpenAIEmbeddings(openai_api_key="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"))
        st.session_state["vectorstore"] = vectorstore
    else:
        vectorstore = st.session_state["vectorstore"]

    # Create retreiver
    if 'retriever' not in st.session_state:
        with st.spinner("Creating Multi vector retreiver"):
            retriever=multi_vector_retreiver(vectorstore,texts,tables,text_summaries,table_summaries,image_elements,image_summaries)
        st.session_state["retriever"] = retriever
    else:
        retriever = st.session_state["retriever"]
    st.write(f"{bullet_point} Multi vector retreiver is created")  
    

    # RAG part
    template = """You are AI research assistant.Answer the question based only on the following context, which can include Images,text and tables::
    If table data, analyse the data properly and provide the answer.
    Provide the answer in clear,proper format(add bulleting format if needed). 
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # LLM
    model = ChatOpenAI(temperature=0)
    # model=ChatGoogleGenerativeAI(model="gemini-pro",google_api_key="AIzaSyC5bAj2EV6itXLJa_tUwTOKIBocLaGaR2w")

    # RAG pipeline
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    question = st.text_input('Enter a question') 

    if(question):
        response=chain.invoke(question)
        # search_results = vectorstore.similarity_search(question)
        # response=search_results[0]
        # metadata_id = response.metadata['id']
        # st.write(metadata_id)
        # source_data = retriever.docstore.mget(metadata_id)
        # st.write(type(source_data))
        st.write('Response from InfoGenie:')
        with st.container(height=300):
            st.write(response)