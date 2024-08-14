from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain import HuggingFacePipeline, PromptTemplate,LLMChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import login

# Use your token to log in
login(token='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')

from langchain_huggingface import HuggingFaceEmbeddings




loader = PyPDFLoader("./fundamental-_rights.pdf")

doc = loader.load_and_split()



from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch

class CustomOutputParser(StrOutputParser):
    def parse(self, response: str):
        # Custom logic to extract only the relevant part of the response
        return response.split('[INST]')[-1]  # Adjust based on actual response format


class LLMRAGModel:
    def __init__(self, llm_name="google/gemma-2b-it", retriever_name="sentence-transformers/all-MiniLM-L6-v2"):

        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto", torch_dtype=torch.float16)
        
        self.llmPipeline = pipeline("text-generation",
            model=self.model,
            tokenizer= self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            max_new_tokens = 100,
            do_sample=True,
            top_k=30,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        self.llm = HuggingFacePipeline(pipeline = self.llmPipeline, model_kwargs = {'temperature':0.7,'max_length': 5, 'top_k' :50})
    
    def getPromptFromTemplate(self):
        system_prompt="""Analyze the user's query to understand the specific information they are seeking.
Retrieve relevant sections from the provided content of the Constitution of Pakistan that match the user's query.
Generate a coherent and concise response based on the retrieved information.
If multiple sections are relevant, summarize the key points from each relevant section.
If no relevant information is found in the provided content, respond with:
"The requested information is not available in the provided content of the Constitution of Pakistan."""
        
        B_INST , E_INST = "[INST]","[/INST]"
        B_SYS,E_SYS="<<SYS>>\n", "\n<<SYS>>\n\n"
        system_prompt1=B_SYS+system_prompt+E_SYS
        instruction="""
        History: {history} \n
        Context: {context} \n
        User: {question}"""
        prompt_template=B_INST + system_prompt1 + instruction + E_INST
    
        prompt=PromptTemplate(input_variables=["history","question","context"],template=prompt_template)
        return prompt
    
    def buildRetrieval(self, model_name="sentence-transformers/all-MiniLM-L6-v2", text_files = doc):
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
        text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=128, separator=".")
        texts = text_splitter.split_documents(doc)
        db = FAISS.from_documents(texts, embeddings)
    
        retriever = db.as_retriever()
        return retriever
    
    # can create new chain for each user
    def getnewChain(self):
        prompt = self.getPromptFromTemplate()  # Get a prompt template for the user
        memory = ConversationBufferMemory(input_key="question", memory_key="history", max_len=5)  # Initialize conversation memory
        retriever = self.buildRetrieval()  # Build the retriever using the specified method
        llm_chain = LLMChain(prompt=prompt, llm=self.llm, memory=memory, verbose=True, output_parser=CustomOutputParser())  # Create an LLM chain
        parser = StrOutputParser()  # Initialize a string output parser
        rag_chain = ({  # Define the RAG (Retrieval-Augmented Generation) chain
            "context": retriever, 
            "question": RunnablePassthrough()} | llm_chain
        )
        return rag_chain, retriever  # Return the RAG chain and retriever
    
    
    
model=LLMRAGModel()
