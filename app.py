import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import base64
from transformers import pipeline
from openai import OpenAI

pipe = pipeline("sentiment-analysis",
                model="finiteautomata/bertweet-base-sentiment-analysis")


def col_labels(df, column_name):
    label_count = df[column_name].value_counts()
    return label_count


def plots(labels):
    fig = px.pie(names=labels.index, values=labels.values)
    fig.update_layout(
        showlegend=True,
        autosize=False,
        width=500,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)


def generate_pdf_report(df, questions):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(200, 10, txt="Bennett University NAAC Survey Report",
             ln=True, align="C")
    pdf.ln(10)
    for question in questions:
        pdf.multi_cell(200, 10, txt=question)
        labels = col_labels(df, question)
        for label, count in labels.items():
            pdf.cell(200, 10, txt=f"{label}: {count}", ln=True)
        pdf.ln(5)
    pdf_file = "survey_report.pdf"
    pdf.output(pdf_file)
    return pdf_file


def chunking(arr):
    max_chunk_size = 1024
    chunks = []
    current_chunk = ""

    for recommendation in arr:
        if len(current_chunk) + len(arr) + len('. ') <= max_chunk_size:
            current_chunk += recommendation + '. '
        else:
            chunks.append(current_chunk[:-2])
            current_chunk = recommendation + '. '

    if current_chunk:
        chunks.append(current_chunk[:-2])

    return chunks


def generate_summary(text):
    input_chunks = chunking(text)
    output_chunks = []
    client = OpenAI()
    for chunk in input_chunks:
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=(
                f"Please give summary of:\n{chunk}. The summary given should be in bullet points.\n\nSummary:"),
            temperature=0.7,
            max_tokens=1024,
            n=1,
            stop=None
        )
        summary = response.choices[0].text.strip()
        output_chunks.append(summary)
    return " ".join(output_chunks)

st.set_page_config(
    page_title="Bennett University NAAC Survey Report",
    page_icon="📊",
    layout="wide"
)

hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)


st.markdown(
    "<h1 style='text-align: center; color: #008080;'>Bennett University NAAC Survey Report</h1>",
    unsafe_allow_html=True
)

file = st.file_uploader("Upload Response File", type=['csv'])

if file is not None:
    st.sidebar.info("File uploaded successfully! Proceed with the analysis.")

    df = pd.read_csv(file)

    questions = [
        'How much of the syllabus was covered in the class?',
        'How well did the teachers prepare for the classes?',
        'How well were the teachers able to communicate?',
        'The teacher\'s approach to teaching can best be described as',
        'Fairness of the internal evaluation process by the teachers.',
        'Was your performance in assignments discussed with you?',
        'The institute takes active interest in promoting internship, student exchange, field visit opportunities for students.',
        'The teaching and mentoring process in your institution facilitates you in cognitive, social and emotional growth.',
        'The institution provides multiple opportunities to learn and grow.',
        'Teachers inform you about your expected competencies, course outcomes and programme outcomes.',
        'Your mentor does a necessary follow-up with an assigned task to you.',
        'The teachers illustrate the concepts through examples and applications.',
        'The teachers identify your strengths and encourage you with providing right level of challenges.',
        'Teachers are able to identify your weaknesses and help you to overcome them.',
        'The institution makes effort to engage students in the monitoring, review and continuous quality improvement of the teaching learning process.',
        'The institute/ teachers use student centric methods, such as experiential learning, participative learning and problem solving methodologies for enhancing learning experiences.',
        'Teachers encourage you to participate in extracurricular activities.',
        'Efforts are made by the institute/ teachers to inculcate soft skills, life skills and employability skills to make you ready for the world of work.',
        'What percentage of teachers use ICT tools such as LCD projector, Multimedia, etc. while teaching.',
        'The overall quality of teaching-learning process in your institute is very good.',
    ]

    st.write("---")
    st.write("### Survey Questions Analysis")
    st.write("Below are the analysis results for each survey question:")

    pos_comments = []
    neg_comments = []
    neu_comments = []
    for data in df['Give three observation / suggestions to improve the overall teaching - learning experience in your institution.']:
        sentiment = pipe(data)[0]['label']
        if sentiment == "POS":
            pos_comments.append(data)
        if sentiment == "NEG":
            neg_comments.append(data)
        if sentiment == "NEU":
            neu_comments.append(data)
        else:
            neu_comments.append(data)

    st.subheader("Positive Comments")
    st.write(generate_summary(pos_comments))
    st.write("---")
    st.subheader("Negative Comments")
    st.write(generate_summary(neg_comments))

    st.sidebar.markdown("---")
    st.sidebar.write("#### Analysis Settings")
    selected_questions = [
        question for question in questions if st.sidebar.checkbox(question, value=True)]
    for question in selected_questions:
        st.write("")
        st.subheader(question)
        labels = col_labels(df, question)
        plots(labels)

    st.sidebar.markdown("---")
    st.sidebar.write("#### Download Report")
    if st.sidebar.button("Generate PDF Report"):
        pdf_file = generate_pdf_report(df, selected_questions)
        with open(pdf_file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            href = f"<a href='data:application/octet-stream;base64,{base64_pdf}' download='survey_report.pdf'><button>Download PDF Report</button></a>"
            st.sidebar.markdown(href, unsafe_allow_html=True)
