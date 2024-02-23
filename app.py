import uuid,os
from flask import Flask, jsonify, request
from qa_chain import get_qa_chain
from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, AIMessage

app = Flask(__name__)

# Generate a unique session ID using the uuid module
# Configure the session ID in a config dict

# Initialize an OpenAIEmbeddings object to generate embeddings
# Initialize a Pinecone client
# Get the Pinecone index name from environment variable 
# Create a PineconeVectorStore to store embeddings in Pinecone
# Initialize empty list to store doc IDs
# Create the file folder if it doesn't exist
# Get the file folder path from environment variable
ses_id=uuid.uuid4()
config={"configurable": {"session_id": str(ses_id)}}

embeddings=OpenAIEmbeddings(model="text-embedding-ada-002")
pc=Pinecone()
index_name=os.environ["PINECODE_INDEX"]
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
doc_ids = []
if not os.path.exists(os.environ["FILE_FOLDER"]):
    os.mkdir(os.environ["FILE_FOLDER"])
file_folder=os.environ["FILE_FOLDER"]


def cleanup():
    """Deletes all documents from the Pinecone index.

    Cleans up the Pinecone index after all documents have been processed, 
    removing any documents that were previously added.
    """
    print("cleaning up")
    index = pc.Index(index_name)
    index.delete(doc_ids)


@app.route('/upload_docs',methods=["POST"])
def initialize():

    """Uploads PDF documents to the server from the request.
            
            This route handles uploading PDF documents sent in the request. 
            It saves each file to the configured file folder, loads them using 
            the PyPDFLoader, splits the text into chunks, indexes them in Pinecone
            using the configured index name, and saves the doc ids.
            
            After processing, it deletes the uploaded files to clean up space.
            
            Returns a success message if uploaded and processed correctly.
            """
    
    try:
        for name, file in request.files.items():
            print(os.path.join(file_folder,name))
            with open(os.path.join(file_folder,name), 'wb') as f:
                f.write(file.read())
        
        loader=DirectoryLoader(file_folder,loader_cls=PyPDFLoader)
        documents = loader.load()
        print("loaded documents")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        ids=vectorstore.add_documents(docs)
        doc_ids=ids

        for file in os.listdir(file_folder):
            os.remove(os.path.join(file_folder,file))

        return 'Document Processed!'
    
    except Exception as e:
        print(e)
        return "Some error occured"

@app.route('/get_answer',methods=["GET","POST"])
def use_object():

    """
    Handles incoming requests by parsing the prompt, initializing a 
    vector database, running the the qa_chain and formatting the response.
    """
    
    try:
        prompt=request.json['text']

        db=vectorstore.as_retriever(search_type="mmr")

        qa_chain = get_qa_chain(db)

        answer=qa_chain.invoke({"question":prompt},config=config)

        pages=[]
        docs=[]
        for doc in answer['context']:
            pages.append(doc.metadata['page'])
            source=doc.metadata['source'].split("/")[-1]
            docs.append(source)

        response=jsonify({'answer':answer['output'],
                          "docs":docs, 
                          'pages': pages})
        
        response.headers['Content-Type'] = 'application/json'

        return response
    
    except Exception as e:
        print(e)
        return "Some error occured"

if __name__ == '__main__':
    app.run(debug=True)
    app.teardown_appcontext(cleanup)
