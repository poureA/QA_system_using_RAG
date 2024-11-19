# Import necessaries
import torch
from transformers import pipeline, AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders.pdf import PDFPlumberLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA

class QA_from_docs:
    """Class to build a Question-Answering system from a PDF document using Retrieval-Augmented Generation (RAG)."""

    def __init__(self, document_path, max_words):
        """
        Initializes the class with the document path and max words.

        Args:
            document_path (str): Path to the PDF document.
            max_words (int): Maximum token length for outputs or splits.
        """
        self.document_path = document_path  # Path to the document to process
        self.max_words = max_words  # Maximum number of tokens to use

    def __str__(self):
        """
        String representation of the class instance.
        Returns:
            str: Details about the instance.
        """
        return f"QA_from_docs(document_path={self.document_path}, max_words={self.max_words})"

    def Embedding(self):
        """
        Loads a pre-trained embedding model for document vectorization.

        Returns:
            HuggingFaceEmbeddings: Embedding model for creating document vectors.
        """
        model_path = "sentence-transformers/all-mpnet-base-v2"  # Path to the embedding model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
        return HuggingFaceEmbeddings(model_name=model_path, model_kwargs={'device': device})

    def Model(self):
        """
        Loads a pre-trained generative language model for answering questions.

        Returns:
            pipeline: Hugging Face text-to-text generation pipeline.
        """
        model_path = "google/flan-t5-xl"  # Hugging Face model path
        tokenizer = AutoTokenizer.from_pretrained(model_path, truncation=True, padding_side="right")  # Load tokenizer
        return pipeline(task='text2text-generation',  # Create a text generation pipeline
                        model=model_path,
                        tokenizer=tokenizer,
                        max_new_tokens=self.max_words,  # Maximum tokens for generated answers
                        device_map='auto',  # Automatically map to available devices
                        model_kwargs={"load_in_8bit": False, "temperature": 0.95})  # Configure generation parameters

    def Splitter(self):
        """
        Loads and splits the document into smaller chunks for efficient processing.

        Returns:
            list: List of smaller text chunks.
        """
        try:
            loader = PDFPlumberLoader(self.document_path)  # Load the PDF document
            texts = loader.load()  # Extract text from the PDF
            textsplitter = TokenTextSplitter(chunk_size=self.max_words, chunk_overlap=10)  # Define chunking strategy
            return textsplitter.split_documents(texts)  # Split the document into smaller parts
        except Exception as e:
            # Handle errors, such as unsupported file format or missing document
            raise RuntimeError("Error splitting documents. Check the document format.") from e

    def Make_vectordb(self, persist_directory="/content/chromadb"):
        """
        Creates or loads a persistent vector database for document retrieval.

        Args:
            persist_directory (str): Directory to save or load the vector database.

        Returns:
            Chroma: Vector database instance.
        """
        return Chroma.from_documents(documents=self.Splitter(),  # Use split documents
                                     embedding=self.Embedding(),  # Use embedding model for vectorization
                                     persist_directory=persist_directory)  # Specify storage directory

    def QA(self, question):
        """
        Answers a question using the RAG approach.

        Args:
            question (str): The question to be answered.

        Returns:
            str: Generated answer to the question.
        """
        llm = HuggingFacePipeline(pipeline=self.Model())  # Load the language model pipeline
        retriever = self.Make_vectordb().as_retriever(search_kwargs={'k': 1})  # Retrieve the most relevant document
        RAG = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)  # Combine retrieval and generation
        result = RAG({'query': question})  # Pass the question to the RAG chain
        return result['result']  # Return the generated answer


# Usage example
sample = QA_from_docs('/content/DV 2026 Plain Language Instructions and FAQs.pdf', 200)  # Initialize with PDF path and max tokens
print(sample.QA('Why do natives of certain countries not qualify for the DV program?'))  # Ask a question and print the answer