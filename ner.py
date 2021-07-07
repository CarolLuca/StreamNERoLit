#Core Pkgs
import streamlit as st

#NLP Pkgs
import spacy_streamlit
import spacy
nlp = spacy.load("C:\\Users\\Carol Luca\\Desktop\\Code\\RACAI\\Versions\\Phase 2\\Version 1.2\\output\\model-best")

#Web Scraping Pkgs
from bs4 import BeautifulSoup
from urllib.request import urlopen

@st.cache
def get_text(raw_url):
    page = urlopen(raw_url)
    soup = BeautifulSoup(page)
    fetched_text = " ".join(map(lambda p:p.text, soup.find_all('p')))
    return fetched_text


def main():
    """A Simple NLP App with Spacy-Streamlit"""
    st.title("Named Entity Recognition")

    menu = ["NER (preferred)", "NER for URL"]
    choice = st.sidebar.radio("Pick a choice", menu)


    if choice == "NER (preferred)":
        raw_text = st.text_area("Enter Text","")
        if raw_text != "":
            docx = nlp(raw_text)
            spacy_streamlit.visualize_ner(docx, labels = nlp.get_pipe('ner').labels)

    elif choice == "NER for URL":
        raw_url = st.text_input("Enter URL","")
        st.write("Example: https://www.kosson.ro/restitutio/45-cultura-scrisa/407-scrisoarea-lui-neascu-1521")
        text_length = st.slider("Length to Preview", 50, 200)
        if raw_url != "":
            result = get_text(raw_url)
            len_of_full_text = len(result)
            len_of_short_text = round(len(result)/text_length)
            st.subheader("Text to be analyzed:")
            st.write(result[:len_of_short_text])
            preview_docx = nlp(result[:len_of_short_text])
            spacy_streamlit.visualize_ner(preview_docx, labels = nlp.get_pipe('ner').labels)

if __name__ == '__main__':
    main()