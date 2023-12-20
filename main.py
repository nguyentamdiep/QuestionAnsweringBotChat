import streamlit as st
from openai import OpenAI
#OPENAI_API_KEY=""
#client = OpenAI(api_key=OPENAI_API_KEY)
client = OpenAI()
st.title("Hỏi đáp về covid-19")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def GPT(prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content


from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import TextLoader
#from langchain.vectorstores import FAISS
import pandas as pd
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
embeddings = HuggingFaceEmbeddings()
#db = FAISS.load_local("covid_faiss_index_1", embeddings)
new_db = Chroma(persist_directory="qna_covid_db", embedding_function=embeddings)
def Question_Answer(prompt):
    # # if (prompt == ""):
    # #     ans = "no content"
    if prompt[:8] == "--no-llm" and prompt[8]==' ':
        prompt = prompt[9:]
        print(prompt)
        results_with_scores = new_db.similarity_search_with_score(prompt, 4)
        print(results_with_scores)
        #docs[0][0].page_content
        ans = ""
        for i in range(len(results_with_scores)):
            ans += results_with_scores[i][0].page_content + '\n\n'
        //print(ans)
    else:
        results_with_scores = new_db.similarity_search_with_score(prompt, 4)
        context = results_with_scores[0][0].page_content + " " + results_with_scores[1][0].page_content + results_with_scores[2][0].page_content + results_with_scores[3][0].page_content
        question = prompt
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Trả lời câu hỏi dưới đây chỉ dựa trên thông tin đã cho, không dựa trên hiểu biết sẵn có của bạn, nếu không có câu trả lời phù hợp thì nói tôi không biết"},
                {"role": "user", "content": "Thông tin đã cho: " + context + "\n" + "Câu hỏi: " + question +"\n" + "Câu trả lời là: "}
            ],
            max_tokens=150
        )
        ans = response.choices[0].message.content
    return ans
    # docs = new_db.similarity_search_with_score(prompt, 4)
    # print(docs)


# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = Question_Answer(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})