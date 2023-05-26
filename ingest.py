"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import YoutubeLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.vectorstores.faiss import FAISS
import os


def ingest_docs():
    """Get documents from web pages."""
    # Youtube Loader
    # TeaParty Intro Part 1: "What Is TeaParty? How Does it Work?"
    loader1 = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=pjIyjJMCkZk", add_video_info=True)
    documents1 = loader1.load()

    # Tea Party Intro Part 2. "What Is Tea? How Do I Use It?"
    loader2 = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=xuj3erpSjs4&t", add_video_info=True)
    documents2 = loader2.load()

    # Adding New Chains to Tea
    loader3 = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=PvDzNwTVU9M", add_video_info=True)
    documents3 = loader3.load()

    # Demo: Radiant to PartyChain Trade Using tea
    loader4 = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=_ixv-BG4Bak", add_video_info=True)
    documents4 = loader4.load()

    # Steps to Propose a New Chain Integration in TeaParty
    loader5 = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=iantmMnrcx4", add_video_info=True)
    documents5 = loader5.load()

    # NFT's and Chill with TeaParty
    loader6 = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=DTNsZcnURjw", add_video_info=True)
    documents6 = loader6.load()


    # Markdown Loader
    loader = UnstructuredMarkdownLoader("learnme/faq.md")
    document7 = loader.load()

    loader = UnstructuredMarkdownLoader("learnme/whyneedteaparty.md")
    document8 = loader.load()


    # Combine the documents from both sources
    raw_documents = documents1 + documents2 + documents3 + documents4 + documents5 + documents6 + document7 + document8



    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

if __name__ == "__main__":
    ingest_docs()

