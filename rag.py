# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ipython==9.8.0",
#     "langchain==1.1.3",
#     "langchain-chroma==1.0.0",
#     "langchain-community==0.4.1",
#     "langchain-docling==2.0.0",
#     "langchain-huggingface==1.1.0",
#     "langchain-openai==1.1.1",
#     "langchain-text-splitters==1.0.0",
#     "openai==2.9.0",
#     "python-dotenv==1.2.1",
#     "sentence-transformers==5.1.2",
#     "wget==3.2",
# ]
# ///

import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import wget 

    # OpenAI SDK to use OpenRouter
    from openai import OpenAI

    # Load the env variables
    from dotenv import load_dotenv
    load_dotenv()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # What is RAG
    Retrieval-Augmented Generation (RAG) is a machine-learning technique that integrates information retrieval with generative AI to produce accurate and context-aware responses.

    **REFERENCE--**[IBM Skill Network](https://author-ide.skills.network/render?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJtZF9pbnN0cnVjdGlvbnNfdXJsIjoiaHR0cHM6Ly9jZi1jb3Vyc2VzLWRhdGEuczMudXMuY2xvdWQtb2JqZWN0LXN0b3JhZ2UuYXBwZG9tYWluLmNsb3VkL0ltR2tkaE1TZDNpVkZlS2hoSmVkWHcvUmVhZGluZyUyMC0lMjBXaGF0JTIwaXMlMjBSQUctdjEubWQ_dD0xNzQ2MTI4ODUzIiwidG9vbF90eXBlIjoiaW5zdHJ1Y3Rpb25hbC1sYWIiLCJhdGxhc19maWxlX2lkIjozMDM5MTEsImFkbWluIjpmYWxzZSwiaWF0IjoxNzU3Njk3NTExfQ.bMtyovXKhOxlCwRqSGxcW_L_MN4llY1yHbBOGyKbNGY)


    The steps in the RAG process are as follows:

    1.	Gather Sources: Start with sources like office documents, company policies, or any other relevant information that may provide context for the user's future prompt.
    2.	Embed Sources: Pass the gathered information through an embedding model. The embedding model converts each chunk of text into a vector representation, which is essentially a fixed-length column of numbers.
    3.	Store Vectors: Store the embedded source vectors in a vector store — a specialized database optimized for storing and manipulating vector data.
    4.	Obtain a User's Prompt: Receive a prompt from the user.
    5.	Embed the User's Prompt: Embed the user's prompt using the same embedding model used for the source documents. This produces a prompt embedding, which is a vector of numbers equal in length to the vectors representing the source embeddings.
    6.	Retrieve Relevant Data: Pass the prompt embedding to the retriever. The retriever also accesses the vector store to find and pull relevant source embeddings (vectors) that match the prompt embedding. The retriever's output is the retrieved text.
    7.	Create an Augmented Prompt: Combine the retrieved text with the user's original prompt to form an augmented prompt.
    8.	Obtain a Response: Feed the augmented prompt into a large language model (LLM), which processes it and produces a response.
    """)
    return


@app.cell(hide_code=True)
def _():
    api = mo.ui.text(
        placeholder="API-KEY...", label="Openrouter API", kind="password"
    )
    api
    return (api,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Pre-processing

    ### Load the document

    The document, which is provided in a TXT format, outlines some company policies and serves as an example data set for the project.

    This is the `load` step in `Indexing`.<br>
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/MPdUH7bXpHR5muZztZfOQg.png" width="50%" alt="split"/>
    """)
    return


@app.cell
def _():
    run_btn = mo.ui.run_button(label="Download file")
    run_btn
    return (run_btn,)


@app.cell(hide_code=True)
def _(run_btn):
    filename: str = 'companyPolicies.txt'
    def _run_download() -> None:
        _url: str = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'
        wget.download(_url, out=filename)
        print('file downloaded')

    if run_btn.value:
        _run_download()

    with open(filename, 'r') as _file:
        contents: str = _file.read()
    # Display the downloaded contents
    mo.md(f""" ## FILE - {filename}
    {contents}
    """)
    return (filename,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Splitting the document into chunks
    In this step, you are splitting the document into chunks, which is basically the `split` process in `Indexing`.
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/0JFmAV5e_mejAXvCilgHWg.png" width="50%" alt="split"/>

    The code in the IBM for "TextLoader" is old and does not exits. It is now moved to langchain community. I tried the Docling loader for this, but does not handle .txt files

    **REFERENCE:**

     - [Langchain Docling](https://docs.langchain.com/oss/python/integrations/document_loaders/docling)
     - [PyPi Langchain Community](https://pypi.org/project/langchain-community/)
     - [Text Splitters](https://docs.langchain.com/oss/python/integrations/splitters)
    """)
    return


@app.cell
def _(filename: str):
    ## Docling loader
    # from langchain_docling import DoclingLoader -- does not load .txt files
    from langchain_community.document_loaders import TextLoader

    # Load the text
    loader = TextLoader(file_path=filename)
    documents = loader.load()
    documents
    return (documents,)


@app.cell
def _(documents):
    ## Text splitters
    from langchain_text_splitters import CharacterTextSplitter

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    texts
    return (texts,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Embedding and storing
    This step is the `embed` and `store` processes in `Indexing`. <br>
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/u_oJz3v2cSR_lr0YvU6PaA.png" width="50%" alt="split"/>

    I am using the HuggingFaceEmbeddings that was also in the IBM exercise.

    **REFERENCE**

     - [Langchain HuggingFaceEmbeddings](https://docs.langchain.com/oss/python/integrations/text_embedding/huggingfacehub)
     - [Langchain ChromaDB](https://reference.langchain.com/python/integrations/langchain_chroma/#langchain_chroma.Chroma.from_documents)
    """)
    return


@app.cell
def _(texts):
    # Embedding
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    # Vector DB
    from langchain_chroma import Chroma

    # Creating the vectorDB
    embeddings   = HuggingFaceEmbeddings()
    vector_store = Chroma.from_documents(documents=texts,
                                      embedding=embeddings,
                                  
                                      persist_directory="./chromaDB_test")  # store the embedding in vector_store using Chromadb
    mo.md('document ingested')
    return (vector_store,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### LLM model construction

    Okay starting from this, the IBM assignment is pretty old. The Langchain functions used are no longer supported.

    The rest is from the Langchain official RAG tutorial.

    **REFERENCE**

     - [Langchain Agentic RAG](https://docs.langchain.com/oss/python/langchain/rag)
    """)
    return


@app.cell
def _(api):
    # Using OpenAI SDK to call OpenRouter models
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="openai/gpt-oss-20b:free",
        api_key=api.value,
        base_url="https://openrouter.ai/api/v1",
    )
    return (llm,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Retrieval and Generation

    Following the Langchain documentation, I will implement the Agent that has access to tools - in this case the vector DB we created before.
    """)
    return


@app.cell
def _(vector_store):
    ## Context tool
    ### From - https://docs.langchain.com/oss/python/langchain/rag#rag-agents

    from langchain.tools import tool

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""

        retrieved_docs = vector_store.similarity_search(query, k=2)
    
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    return (retrieve_context,)


@app.cell
def _(llm, retrieve_context):
    ## Agent
    ### From - https://docs.langchain.com/oss/python/langchain/rag#rag-agents

    from langchain.agents import create_agent

    tools = [retrieve_context]

    # If desired, specify custom instructions
    prompt = (
        "You have access to a tool that retrieves context from a blog post. "
        "Use the tool to help answer user queries."
    )
    agent = create_agent(llm, tools, system_prompt=prompt)
    return agent, create_agent


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Generation

    Now we start asking questions to it. Let’s test this out. Currently, in this setup there are two calls to the LLM, one for the tool call and another for the generation of answer.
    """)
    return


@app.cell(disabled=True, hide_code=True)
def _(agent):
    ## Lets ask the agent question now

    query = (
        "What is the company mobile policy?\n\n"
        "Once you get the answer, summarize this document."
            )

    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()
    return


@app.cell
def _(agent):
    ## Lets ask the agent question now
    query1 = ("What is the company mobile policy?")
    result = agent.invoke(
        {
            "messages":
            [
                {
                    "role": "user",
                    "content": query1
                }
            ]
        }
    )
    mo.md(result["messages"][-1].content)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can wrap the vector search, so that there is only one single call to LLM.
    """)
    return


@app.cell
def _(create_agent, llm, vector_store):
    from langchain.agents.middleware import dynamic_prompt, ModelRequest

    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        """Inject context into state messages."""
        last_query = request.state["messages"][-1].text
        retrieved_docs = vector_store.similarity_search(last_query)

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        system_message = (
            "You are a helpful assistant. Use the following context in your response:"
            f"\n\n{docs_content}"
        )

        return system_message


    new_agent = create_agent(llm, tools=[], middleware=[prompt_with_context])

    return (new_agent,)


@app.cell
def _(new_agent):
    query = "What is company mobile policy?"
    new_result = new_agent.invoke(
        {"messages": [{"role": "user", "content": query}]})
    mo.md(new_result["messages"][-1].content)
    return


if __name__ == "__main__":
    app.run()
