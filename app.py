import streamlit as st
import fitz  # PyMuPDF
import textstat
from transformers import pipeline

# Função para extrair texto do PDF
def extrair_texto(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    texto = ""
    for page in doc:
        texto += page.get_text()
    return texto

# Carregar pipeline de simplificação e resumo
@st.cache_resource
def carregar_modelos():
    resumo = pipeline("summarization", model="facebook/bart-large-cnn")
    return resumo

resumo_model = carregar_modelos()

# Interface do usuário
st.title("🧾 Analisador de Acessibilidade - Projeto de Lei")

uploaded_file = st.file_uploader("Envie o arquivo PDF do Projeto de Lei", type="pdf")

if uploaded_file:
    texto_original = extrair_texto(uploaded_file)

    st.subheader("📄 Texto Original")
    st.text_area("Texto extraído", texto_original[:3000], height=300)

    # Legibilidade antes
    score_antes = textstat.flesch_reading_ease(texto_original)
    escolaridade_antes = textstat.text_standard(texto_original, float_output=False)

    # Resumir e simplificar (resumo apenas por enquanto)
    with st.spinner("Gerando resumo simplificado..."):
        texto_simplificado = resumo_model(texto_original[:1024])[0]["summary_text"]

    st.subheader("📝 Texto Simplificado")
    st.text_area("Resumo simplificado", texto_simplificado, height=300)

    # Legibilidade depois
    score_depois = textstat.flesch_reading_ease(texto_simplificado)
    escolaridade_depois = textstat.text_standard(texto_simplificado, float_output=False)

    # Comparação
    st.subheader("📊 Comparação de Acessibilidade")
    st.write(f"**Legibilidade original (Flesch):** {score_antes:.2f}")
    st.write(f"**Escolaridade necessária (original):** {escolaridade_antes}")
    st.write(f"**Legibilidade simplificada:** {score_depois:.2f}")
    st.write(f"**Escolaridade necessária (simplificado):** {escolaridade_depois}")
