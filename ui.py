import streamlit as st
import requests
from langchain_core.messages import HumanMessage, AIMessage
# streamlit run RAGApp.py --server.enableXsrfProtection=false

def process_docs(docs):
    url = 'http://127.0.0.1:5000/upload_docs'
    
    files={doc.name:doc.read() for doc in docs}

    response = requests.post(url,files=files)
    return response.text

def send_prompt(prompt):

    url = 'http://127.0.0.1:5000/get_answer'

    body={"text":prompt}
    response = requests.get(url, json=body)
    return response.json()
    # return response.text
    

def generate_response(prompt):
    
    """Sends the user's prompt to the backend API and displays the full response.
    The full response includes the generated answer, source docs, and source pages.
    The response is displayed in a Streamlit chat message component.
    """
    with st.chat_message("ai"):
        message_placeholder = st.empty()
        full_response=""
    
    response=send_prompt(prompt)
    full_response=f'''{response['answer']} \n
                       Source docs: {response['docs']} \n      
                       Source pages: {response['pages']}'''
    
    message_placeholder.markdown(full_response)

    return full_response

def main():
    """
    Configures and displays the Streamlit UI for the chatbot.

    Sets page title, icon, and chat history state. 
    Allows user to upload docs and send for processing.
    Displays chat history and handles sending user message 
    to backend and displaying response.
    """
    st.set_page_config("Jarvis", page_icon="🤖")
    st.header("Jarvis 🤖")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.subheader("Your docs")
        docs=st.file_uploader("Upload your docs 📚", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                response=process_docs(docs)
                st.markdown(f"{response}")

    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
                if isinstance(message, HumanMessage):
                    with st.chat_message("human"):
                        st.markdown(message.content)
                else:
                    with st.chat_message("ai"):
                        st.markdown(message.content)

    user_prompt=st.text_input(label=" ", placeholder="your message",max_chars=200)
    if user_prompt and user_prompt!="":

        st.session_state.chat_history.append(HumanMessage(content=user_prompt))
        with st.chat_message("human"):
            st.markdown(user_prompt)

        with st.spinner("Thinking"):
            full_response=generate_response(user_prompt)
            st.session_state.chat_history.append(AIMessage(content=full_response))
            

if __name__ == "__main__":
    main()





