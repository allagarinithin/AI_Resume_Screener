import streamlit as st
from groq import Groq
import PyPDF2
import io
import os
import time
import logging
from dotenv import load_dotenv
import uuid
import boto3
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()

# AWS Setup
dynamodb = boto3.resource(
    'dynamodb',
    region_name=os.getenv('AWS_REGION', 'ap-south-1'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

def upload_item_to_dynamodb(table_name, item):
    try:
        table = dynamodb.Table(table_name)
        table.put_item(Item=item)
    except ClientError as e:
        logging.error(f"DynamoDB Error: {e.response['Error']['Message']}")

def initialize_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Missing GROQ API Key!")
        return None
    return Groq(api_key=api_key)

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return ''.join(page.extract_text() or '' for page in pdf_reader.pages)
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None

def analyze_resume(client, resume_text, job_description):
    if not client:
        return None
    prompt = f"""
    Analyze the resume against the job description.
    Provide:
    - Match Score (0-100)
    - Key Qualifications Match
    - Missing Skills
    - Strengths
    - Areas for Improvement
    - Suggested Resume Improvements
    
    Resume:
    {resume_text}
    
    Job Description:
    {job_description}
    """
    
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are an expert resume analyzer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None

def main():
    st.title("📝 Resume Analyzer")
    client = initialize_groq_client()

    name = st.text_input("Name")
    email = st.text_input("Email")
    linkedin_profile = st.text_input("LinkedIn Profile")
    preferred_job_role = st.text_input("Preferred Job Role")
    
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=['pdf'])
    job_description = st.text_area("Paste Job Description", height=200)

    if uploaded_file and job_description:
        resume_text = extract_text_from_pdf(uploaded_file)
        if resume_text:
            st.text_area("Extracted Resume Text", resume_text, height=300)
            if st.button("Analyze Resume"):
                with st.spinner("Analyzing..."):
                    time.sleep(1)
                    analysis = analyze_resume(client, resume_text, job_description)
                    if analysis:
                        st.write(analysis)

                        # Save to DynamoDB
                        item = {
                            'id': str(uuid.uuid4()),
                            'name': name,
                            'email': email,
                            'linkedin_profile': linkedin_profile,
                            'preferred_job_role': preferred_job_role,
                            'resume_text': resume_text,
                            'analysis': analysis
                        }
                        upload_item_to_dynamodb('resume-analyzer', item)
                        
                        # Download Option
                        st.download_button("Download Analysis", data=analysis.encode(), file_name="analysis.txt", mime="text/plain")

if __name__ == "__main__":
    main()
