import validators
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# streamlit app
st.set_page_config(page_title="LangChain: Summarize text from youtube or website url")
st.title("LangChain: Summarize text from youtube or website url")
st.subheader('Summarize URL')

# get the groq api key and the url
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key", value="",type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# Gemma model using groq api
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

prompt_template ="""
Provide summary of the content in 300 words
Content:{text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the content from YT or website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information on left")
    elif not validators.url(generic_url):
        st.error("Please enter a valid url(either a youtube video or website url)")

    else:
        try:
            with st.spinner("Waiting..."):
                # loading the website or youtube url data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                   headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"})
                    
                docs = loader.load()
                    # chain for summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception: {e}")