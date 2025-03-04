# prjoct for Resume matching in python using streamlit, cosine similarity algorithm
import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.title("Resume Keywords matching Tool")

# st.subheader("NLP Based Resume Matching")

# File uploader for Job Description and Resume
uploadedJD = st.file_uploader("Upload Job Description", type=["pdf", "txt"])
uploadedResume = st.file_uploader("Upload Resume as PDF", type="pdf")

click = st.button("Process")

# Function to extract text from uploaded PDF using pdfplumber
def extract_text_from_pdf(uploaded_file):
    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text() + '\n'
        return text
    return None

# Function to extract text from uploaded text file
def extract_text_from_txt(uploaded_file):
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        return text
    return None

# Function to compute similarity and find non-matching words
def get_similarity_and_non_matching_words(job_description, resume):
    content = [job_description, resume]
    cv = CountVectorizer()
    matrix = cv.fit_transform(content)
    similarity_matrix = cosine_similarity(matrix)
    match_percentage = similarity_matrix[0][1] * 100

    # Extract feature names (words)
    feature_names = cv.get_feature_names_out()
    
    # Get term frequencies for both documents
    job_desc_vector = matrix.toarray()[0]
    resume_vector = matrix.toarray()[1]

    # Identify non-matching words
    non_matching_words = {
        "job_description_only": [],
        "resume_only": []
    }

    for word, job_freq, resume_freq in zip(feature_names, job_desc_vector, resume_vector):
        if job_freq > 0 and resume_freq == 0:
            non_matching_words["job_description_only"].append(word)
        if resume_freq > 0 and job_freq == 0:
            non_matching_words["resume_only"].append(word)
    
    return match_percentage, non_matching_words

# Extract text from uploaded files
job_description = None
if uploadedJD is not None:
    if uploadedJD.type == "application/pdf":
        job_description = extract_text_from_pdf(uploadedJD)
    elif uploadedJD.type == "text/plain":
        job_description = extract_text_from_txt(uploadedJD)

resume = extract_text_from_pdf(uploadedResume) if uploadedResume else None

# Process and display results
if click:
    if job_description and resume:
        match_percentage, non_matching_words = get_similarity_and_non_matching_words(job_description, resume)
        match_percentage = round(match_percentage, 2)
        st.write(f"Match Percentage: {match_percentage}%")

        st.write("Words present in Job Description but not in Resume:")
        st.write(non_matching_words["job_description_only"])
        
         # Save non-matching words to a text file
        with open("non_matching_words_job_description.txt", "w") as file:
            file.write("\n".join(non_matching_words["job_description_only"]))
        
        # Provide a download link for the text file
        with open("non_matching_words_job_description.txt", "rb") as file:
            btn = st.download_button(
                label="Download Non-Matching Words from Job Description",
                data=file,
                file_name="non_matching_words_job_description.txt",
                mime="text/plain"
            )
    else:
        st.write("Please upload both the Job Description and the Resume.")
