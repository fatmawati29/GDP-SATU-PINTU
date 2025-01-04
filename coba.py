from fpdf import FPDF
import io
import streamlit as st

# Membuat PDF
pdf = FPDF()
pdf.add_page()

# Menambahkan font DejaVu (Unicode font)
pdf.add_font("DejaVu", "", "path/to/DejaVuSans.ttf", uni=True)
pdf.set_font("DejaVu", size=12)

# Judul PDF
pdf.cell(200, 10, txt="Hasil Latihan Soal Pendapatan Nasional", ln=True, align='C')
pdf.ln(10)

# Soal dan jawaban (contoh)
questions = [
    {"question": "Apa itu Pendapatan Nasional?", "options": ["Pilihan 1", "Pilihan 2"], "answer": "Pilihan 1"},
    {"question": "Apa itu Produk Domestik Bruto?", "options": ["Pilihan A", "Pilihan B"], "answer": "Pilihan A"}
]

# Menambahkan soal dan jawaban
for i, q in enumerate(questions):
    pdf.multi_cell(0, 10, txt=f"Soal {i + 1}: {q['question']}")
    user_answer = "Pilihan 1"  # Jawaban pengguna, misalnya
    if user_answer == q["answer"]:
        pdf.multi_cell(0, 10, txt=f"Jawaban Anda: {user_answer} ✅")
    else:
        pdf.multi_cell(0, 10, txt=f"Jawaban Anda: {user_answer} ❌")

# Output PDF
pdf_output = io.BytesIO()
pdf.output(pdf_output)
pdf_output.seek(0)

# Mengunduh file PDF
st.download_button(
    label="Download PDF Hasil Latihan",
    data=pdf_output,
    file_name="hasil_latihan_soal.pdf",
    mime="application/pdf"
)
