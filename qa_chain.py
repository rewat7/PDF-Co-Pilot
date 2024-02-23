from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

"""
contextualize_q_chain takes in an LLM and returns an object (Lanagchain runnable chain) that can 
contextualize a question based on a chat history. It uses a prompt template
to format the question and chat history, gets the message history from 
Redis for the given session ID, runs the prompt through the LLM to get the
contextualized question, and returns a RunnableWithMessageHistory that can be
used to run this contextualization.
"""
def contextualize_q_chain(llm):

    system_message = """Given a chat history and the latest user question
    which might reference context in the chat history, formulate a standalone question
    which can be understood without the chat history. Do NOT answer the question,
    just reformulate it if needed and otherwise return it as is.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_message),
            MessagesPlaceholder("chat_history"),
            ("human","{question}")
        ]
    )

    def get_message_history(session_id: str):
        return RedisChatMessageHistory(session_id, url="redis://localhost:55000")

    chain= prompt | llm | StrOutputParser()
    contextualize_q_chain = RunnableWithMessageHistory(chain,
                                                       get_message_history,
                                                       input_messages_key="question",
                                                       history_messages_key="chat_history")
    
    return contextualize_q_chain


def get_qa_chain(retriever):

    """
    This FUNCTION defines a question answering pipeline that takes in a question 
    and retrieved context documents, formats the context documents, prompts an 
    AI assistant (LLM) to answer the question using the context, and parses the output.
    The qa_chain combines a retriever to get context documents, a formatter to join the documents, 
    a prompt template to pose the question with context to the AI, calling the AI (LLM), and parsing the output.
    """
    llm=ChatOpenAI(model="gpt-3.5-turbo-0125",temperature=0.5)

    qa_system_prompt = '''
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question
    If you don't know the answer, just say that you don't know.
    {context}
    '''

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{question}")
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    context_q_chain=contextualize_q_chain(llm)

    chain=(
        RunnablePassthrough(context=lambda x: format_docs(x["sources"]))
        |qa_prompt
        |llm
        |StrOutputParser()
    )
    
    qa_chain = (
                RunnableParallel(question=RunnablePassthrough(),
                                 context=context_q_chain|retriever)            
                .assign(output=chain)
               )
    
    return qa_chain