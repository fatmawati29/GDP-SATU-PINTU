import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.image("income.png", use_container_width=True)
image_path = r'static\BACKGROUND InY (2).png'

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["Teori", "CalcY", "CalSimPro", "CalSimPan", "CalSimPel", "Pendapatan Perkapita", "Distribusi Pendapatan", "Latihan Soal", "Fakta-fakta", "Referensi Bacaan"])
def set_background(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_image});
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            min-height: 100vh;
        }}
        </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error setting background: {e}")

set_background(image_path)

st.markdown(
    """
    <style>
    body {
        color: #FFFFFF; /* Warna teks putih */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# teori
with tab1:
    st.header("Pengertian Pendapatan Nasional")
    st.markdown("""
    Pendapatan nasional merupakan total pendapatan yang diperoleh oleh seluruh pelaku ekonomi dalam suatu negara selama periode tertentu, biasanya satu tahun.
    Besarnya pendapatan nasional dipengaruhi oleh berbagai faktor, seperti ketersediaan faktor produksi, kualitas sumber daya manusia, teknologi,
    modal, stabilitas nasional, serta kebijakan yang diterapkan.
    Pendapatan nasional mencerminkan tingkat produksi barang dan jasa yang dihasilkan oleh suatu negara dan memiliki tiga pendekatan dalam perhitungannya, yaitu:
    - Pendekatan Produksi. Yaitu menghitung nilai semua barang dan jasa yang dihasilkan.""")
    st.info("Rumusnya: Y = P x Q")
    st.markdown("""
    - Pendekatan Pendapatan. Yaitu menjumlahkan semua pendapatan yang diperoleh pelaku ekonomi, baik individu maupun perusahaan.""")
    st.info("Rumusnya: Y = r + w + i + p")
    st.markdown("""
    - Pendekatan Pengeluaran. Yaitu menghitung total pengeluaran untuk barang dan jasa.
    """)
    st.info("Rumusnya:  Y = C + G + I + (X - M)")
    st.info("Kamu bisa mencoba mensimulasikan tiga pendekatan tadi di menu SimPro, SimPan, dan SimPel, ya!")
    
    st.markdown("""
    ### Manfaat Pendapatan Nasional
    - Memahami apakah struktur ekonomi suatu negara lebih dominan di sektor industri, agraris, atau jasa.
    - Membandingkan kemajuan ekonomi suatu negara dari waktu ke waktu serta antarnegara.
    - Memberikan pedoman kepada pemerintah untuk menyusun kebijakan pembangunan ekonomi nasional.
    - Menilai kinerja ekonomi suatu negara berdasarkan data pendapatan nasional.
    ### Konsep Pendapatan Nasional
    - **Gross Domestic Product (GDP).**
        GDP adalah nilai total barang dan jasa yang dihasilkan oleh unit produksi dalam wilayah suatu negara selama satu tahun, termasuk yang dihasilkan oleh perusahaan asing yang beroperasi di negara tersebut.
        GDP bersifat bruto (kotor), artinya belum dikurangi penyusutan atau depresiasi barang modal. Oleh karena itu, GDP sering disebut PDB atau Product Domestic Bruto.
    - **Gross National Product (GNP).**
        GNP adalah nilai total barang dan jasa yang dihasilkan oleh warga negara suatu negara, baik yang tinggal di dalam negeri maupun di luar negeri, tetapi tidak termasuk produksi perusahaan asing di dalam negeri.
        GNP menggambarkan kontribusi ekonomi warga negara dalam dan luar negeri..""")
    st.info("Rumusnya: GNP = GDP + Pendapatan WNI di Luar Negeri - Pendapatan WNA di Dalam Negeri")
    st.markdown("""
    - **Net National Product (NNP).**
        NNP adalah nilai barang dan jasa yang dihasilkan dalam satu tahun setelah dikurangi depresiasi (penyusutan barang modal).
        NNP digunakan untuk menunjukkan nilai bersih dari hasil produksi ekonomi.""")
    st.info("Rumusnya: NNP = GNP - Penyusutan")
    st.markdown("""
    - **Net National Income (NNI).**
        NNI adalah pendapatan bersih suatu negara yang dihitung dari NNP dikurangi pajak tidak langsung (seperti Pajak Pertambahan Nilai).
        NNI menunjukkan jumlah pendapatan yang diterima masyarakat sebagai balas jasa faktor produksi.""")
    st.info("Rumusnya: NNI = NNP - Pajak Tidak Langsung + Subsidi")
    st.markdown("""
    - **Personal Income (PI).**
        PI adalah seluruh pendapatan yang benar-benar diterima oleh individu dalam masyarakat, termasuk transfer payment (seperti bantuan sosial),
        tetapi tidak termasuk laba ditahan, pajak perusahaan, dan kontribusi asuransi sosial.""")
    st.info("Rumusnya: PI = (NNP + Transfer Payment) - (Laba ditahan + Iuran Asuransi + Pajak Perseroan + Jaminan Sosial)")
    st.markdown("""
    - **Disposable  Income (DI).**
        DI adalah pendapatan pribadi setelah dikurangi pajak langsung, yang dapat digunakan untuk konsumsi barang dan jasa atau ditabung.
        Pendapatan ini juga biasa disebut pendapatan yang siap dibelanjakan.
        """)
    st.info("Rumusnya: DI = PI - Pajak Langsung")
    
    st.info("Pelajari lebih detail rumus-rumus perhitungan setiap komponennya di menu CalcY, ya!")

# kalkulator komponen calcy
with tab2:
    st.header("Kalkulator Pendapatan (CalcY)")
    st.info("""
    Kamu bisa menggunakan CalcY untuk menghitung komponen-komponen Pendapatan Nasional, seperti GDP, GNP, NNP, NNI, PI, dan DI.
    """)
    st.markdown("""
    Sebuah negara memiliki data ekonomi sebagai berikut:
    - GDP: Rp150.000
    - Pendapatan Neto dari Luar Negeri: Rp50.000
    - Penyusutan: Rp100.000
    - Pajak Tidak Langsung: Rp70.000
    - Subsidi: Rp20.000
    - Pajak Penghasilan: Rp100.000
    - Iuran Jaminan Sosial: Rp50.000
    - Laba Ditahan: Rp100.000
    - Transfer Payment: Rp200.000

    Hitunglah GDP, GNP, NNP, NNI, PI, dan DI menggunakan CalcY!
    """)

    st.info("""
    Petunjuk penggunaan CalcY:
    - Masukkan angka tanpa tanda koma (,) atau titik (.). Contoh: memasukkan Rp150.000, menjadi 150000.
    - Perhatikan baik-baik data yang akan diinputkan.
    """)

    gdp2 = st.number_input("GDP", min_value=0, step=100)
    income_from_abroad = st.number_input("Pendapatan WNI di Luar Negeri", min_value=0, step=100)
    foreign_income = st.number_input("Pendapatan WNA di Dalam Negeri", min_value=0, step=100)
    depreciation = st.number_input("Depresiasi (Penyusutan)", min_value=0, step=100)
    subsidy = st.number_input("Subsidi", min_value=0, step=100)
    direct_tax = st.number_input("Pajak Langsung", min_value=0, step=100)
    indirect_tax = st.number_input("Pajak Tidak Langsung", min_value=0, step=100)
    trans_pay = st.number_input("Transfer Payment", min_value=0, step=100)
    laba_ditahan = st.number_input("Laba Ditahan", min_value=0, step=100)
    iuran_insurance = st.number_input("Iuran Asuransi", min_value=0, step=100)
    personal_tax = st.number_input("Pajak Perseroan", min_value=0, step=100)
    jamsos = st.number_input("Iuran Jaminan Sosial", min_value=0, step=100)

    gnp = gdp2 + income_from_abroad - foreign_income
    nnp = gnp - depreciation
    nni = nnp - indirect_tax + subsidy
    pi = (nni + trans_pay) - (laba_ditahan + iuran_insurance + personal_tax + jamsos)
    di = pi - direct_tax

    st.subheader("Hasil Kalkulasi Komponen Pendapatan Nasional")

    data = {
            "Komponen": ["GDP", "GNP", "NNP", "NNI", "PI", "DI"],
            "Nilai": [gdp2, gnp, nnp, nni, pi, di]
        }
    df = pd.DataFrame(data)
    df_reset = df.reset_index(drop=True)
    st.dataframe(df_reset, use_container_width=True)


# pendekatan produksi
with tab3:
    st.header("Kalkulator dan Simulasi Pendapatan Nasional - Pendekatan Produksi (CalSimPro)")
    st.info("""
    Kamu bisa menggunakan menu CalSimPro ini untuk mempelajari lebih dalam tentang perhitungan Pendapatan Nasional
    Pendekatan Produksi, ya. Selain itu, kamu bisa mencoba mensimulasikan komponen-komponen produksi
    dapat mempengaruhi pendapatan nasional suatu negara. Selamat mencoba, ya!
    """)

    st.info("""
    Petunjuk penggunaan CalSimPro:
    - Masukkan angka tanpa tanda koma (,) atau titik (.). Contoh: memasukkan Rp150.000, menjadi 150000.
    - Perhatikan baik-baik data yang akan diinputkan.
    """)
    
    agriculture_price = st.number_input("Harga Produk Sektor Pertanian (P)", min_value=0, step=100)
    agriculture_quantity = st.number_input("Jumlah Produk Sektor Pertanian (Q)", min_value=0, step=50)

    industry_price = st.number_input("Harga Produk Sektor Industri (P)", min_value=0, step=100)
    industry_quantity = st.number_input("Jumlah Produk Sektor Industri (Q)", min_value=0, step=50)

    services_price = st.number_input("Harga Produk Sektor Jasa (P)", min_value=0, step=100)
    services_quantity = st.number_input("Jumlah Produk Sektor Jasa (Q)", min_value=0, step=50)

    mining_price = st.number_input("Harga Produk Sektor Pertambangan (P)", min_value=0, step=100)
    mining_quantity = st.number_input("Jumlah Produk Sektor Pertambangan (Q)", min_value=0, step=50)

    others_price = st.number_input("Harga Produk Sektor Lainnya (P)", min_value=0, step=100)
    others_quantity = st.number_input("Jumlah Produk Sektor Lainnya (Q)", min_value=0, step=50)

    agriculture_gdp = agriculture_price * agriculture_quantity
    industry_gdp = industry_price * industry_quantity
    services_gdp = services_price * services_quantity
    mining_gdp = mining_price * mining_quantity
    others_gdp = others_price * others_quantity

    total_y2 = agriculture_gdp + industry_gdp + services_gdp + mining_gdp + others_gdp

    st.subheader("Hasil Simulasi")
    st.write(f"**GDP (Produk Domestik Bruto): {total_y2}**")

    data = {
        "Sektor": ["Pertanian", "Industri", "Jasa", "Pertambangan", "Lainnya", "Total Y"],
        "Nilai": [agriculture_gdp, industry_gdp, services_gdp, mining_gdp, others_gdp, total_y2]
    }
    df = pd.DataFrame(data)
    st.write(df)

    st.subheader("Penjelasan Perubahan")
    if agriculture_quantity > 500:
        st.markdown(f"🔵 **Produksi Pertanian meningkat**. Peningkatan produksi sektor pertanian ({agriculture_quantity}) akan meningkatkan kontribusi sektor ini terhadap GDP.")
    else:
        st.markdown(f"🔵 **Produksi Pertanian menurun**. Penurunan produksi sektor pertanian ({agriculture_quantity}) dapat menurunkan kontribusinya terhadap GDP.")

    if industry_quantity > 400:
        st.markdown(f"🟢 **Produksi Industri meningkat**. Peningkatan produksi sektor industri ({industry_quantity}) akan mendorong pertumbuhan ekonomi melalui tambahan output dan lapangan kerja.")
    else:
        st.markdown(f"🟢 **Produksi Industri menurun**. Penurunan produksi sektor industri ({industry_quantity}) dapat menghambat pertumbuhan ekonomi dan mengurangi kontribusi sektor industri terhadap GDP.")

    if services_quantity > 600:
        st.markdown(f"🟣 **Produksi Sektor Jasa meningkat**. Peningkatan sektor jasa ({services_quantity}) akan menambah kontribusi sektor ini terhadap GDP, terutama di ekonomi berbasis layanan.")
    else:
        st.markdown(f"🟣 **Produksi Sektor Jasa menurun**. Penurunan sektor jasa ({services_quantity}) dapat mengurangi kontribusi sektor ini dalam GDP.")

    if mining_quantity > 300:
        st.markdown(f"🟠 **Produksi Pertambangan meningkat**. Peningkatan produksi sektor pertambangan ({mining_quantity}) dapat menambah kontribusi sektor ini terhadap GDP, terutama di negara dengan sumber daya alam.")
    else:
        st.markdown(f"🟠 **Produksi Pertambangan menurun**. Penurunan produksi sektor pertambangan ({mining_quantity}) dapat mengurangi kontribusinya terhadap GDP.")

    st.subheader("Visualisasi Diagram")
    fig, ax = plt.subplots()
    ax.bar(data["Sektor"][:-1], data["Nilai"][:-1], color=["green", "blue", "orange", "purple", "red"])
    ax.set_title("Komponen Ekonomi dan Nilainya")
    ax.set_ylabel("Nilai")
    ax.set_xlabel("Sektor Ekonomi")
    st.pyplot(fig)

    st.markdown("""
    Cobalah mengubah nilai pada kotak input untuk melihat bagaimana masing-masing komponen memengaruhi **GDP**, serta baca penjelasan di atas untuk memahami dampaknya.
    """)

# pendekatan pendapatan
with tab4:
    st.header("Kalkulator dan Simulasi Pendapatan Nasional - Pendekatan Pendapatan (CalSimPan)")
    st.info("""
    Kamu bisa menggunakan menu CalSimPan ini untuk mempelajari lebih dalam tentang perhitungan Pendapatan Nasional
    Pendekatan Pendapatan, ya. Selain itu, kamu bisa mencoba mensimulasikan komponen-komponen pendapatan
    dapat mempengaruhi pendapatan nasional suatu negara. Selamat mencoba, ya!
    """)

    st.info("""
    Petunjuk penggunaan CalSimPan:
    - Masukkan angka tanpa tanda koma (,) atau titik (.). Contoh: memasukkan Rp150.000, menjadi 150000.
    - Perhatikan baik-baik data yang akan diinputkan.
    """)

    rent = st.number_input("Pendapatan dari Sewa (r)", min_value=0, max_value=100000, value=0, step=100)
    wages = st.number_input("Pendapatan dari Upah (w)", min_value=0, max_value=100000, value=0, step=100)
    interest = st.number_input("Pendapatan dari Bunga (i)", min_value=0, max_value=100000, value=0, step=100)
    profit = st.number_input("Pendapatan dari Keuntungan (p)", min_value=0, max_value=100000, value=0, step=100)

    total_y3 = rent + wages + interest + profit

    st.subheader("Hasil Simulasi")
    st.write(f"**GDP (Produk Domestik Bruto): {total_y3}**")

    data_income = {
        "Komponen": ["Sewa (r)", "Upah (w)", "Bunga (i)", "Keuntungan (p)", "Total Y"],
        "Nilai": [rent, wages, interest, profit, total_y3]
    }
    df_income = pd.DataFrame(data_income)
    st.write(df_income)

    st.subheader("Penjelasan Perubahan")
    if rent > 2000:
        st.markdown(f"🔵 **Pendapatan dari Sewa meningkat**. Peningkatan pendapatan dari sewa ({rent}) menunjukkan adanya kenaikan harga properti atau peningkatan permintaan sewa.")
    else:
        st.markdown(f"🔵 **Pendapatan dari Sewa menurun**. Penurunan pendapatan dari sewa ({rent}) dapat mengindikasikan melemahnya pasar properti atau rendahnya permintaan sewa.")

    if wages > 3000:
        st.markdown(f"🟢 **Pendapatan dari Upah meningkat**. Peningkatan upah ({wages}) menunjukkan adanya kenaikan gaji pekerja, yang berpotensi meningkatkan daya beli masyarakat.")
    else:
        st.markdown(f"🟢 **Pendapatan dari Upah menurun**. Penurunan upah ({wages}) dapat mengindikasikan penurunan daya beli masyarakat dan berpotensi mengurangi konsumsi.")
        
    if interest > 1500:
        st.markdown(f"🟣 **Pendapatan dari Bunga meningkat**. Peningkatan pendapatan dari bunga ({interest}) bisa disebabkan oleh suku bunga yang lebih tinggi, yang menguntungkan pemilik modal.")
    else:
        st.markdown(f"🟣 **Pendapatan dari Bunga menurun**. Penurunan pendapatan bunga ({interest}) dapat menunjukkan penurunan suku bunga atau berkurangnya investasi.")
        
    if profit > 2500:
        st.markdown(f"🟠 **Pendapatan dari Keuntungan meningkat**. Peningkatan keuntungan ({profit}) bisa disebabkan oleh peningkatan efisiensi atau permintaan yang lebih tinggi terhadap barang/jasa.")
    else:
        st.markdown(f"🟠 **Pendapatan dari Keuntungan menurun**. Penurunan keuntungan ({profit}) dapat mengindikasikan penurunan permintaan atau efisiensi dalam sektor bisnis.")

    st.subheader("Visualisasi Komponen Pendapatan Ekonomi")
    fig, ax = plt.subplots()
    ax.bar(data_income["Komponen"][:-1], data_income["Nilai"][:-1], color=["blue", "green", "purple", "red"])
    ax.set_title("Komponen Pendapatan Ekonomi dan Nilainya")
    ax.set_ylabel("Nilai")
    ax.set_xlabel("Komponen Ekonomi")
    st.pyplot(fig)

    st.markdown("""
    Cobalah mengubah nilai pada kotak input di atas untuk melihat bagaimana masing-masing komponen memengaruhi **GDP**, serta baca penjelasan di atas untuk memahami dampaknya.
    """)

# pendekatan pengeluaran
with tab5:
    st.header("Kalkulator dan Simulasi Pendapatan Nasional - Pendekatan Pengeluaran (CalSimPel)")
    st.info("""
    Kamu bisa menggunakan menu CalSimPel ini untuk mempelajari lebih dalam tentang perhitungan Pendapatan Nasional
    Pendekatan Pengeluaran, ya. Selain itu, kamu bisa mencoba mensimulasikan komponen-komponen pengeluaran
    dapat mempengaruhi pendapatan nasional suatu negara. Selamat mencoba, ya!
    """)

    st.info("""
    Petunjuk penggunaan CalSimPel:
    - Masukkan angka tanpa tanda koma (,) atau titik (.). Contoh: memasukkan Rp150.000, menjadi 150000.
    - Perhatikan baik-baik data yang akan diinputkan.
    """)

    consumption = st.number_input("Konsumsi (C)", min_value=0, max_value=10000, value=0, step=100)
    investment = st.number_input("Investasi (I)", min_value=0, max_value=10000, value=0, step=100)
    government_spending = st.number_input("Pengeluaran Pemerintah (G)", min_value=0, max_value=10000, value=0, step=100)
    exports = st.number_input("Ekspor (X)", min_value=0, max_value=10000, value=0, step=100)
    imports = st.number_input("Impor (M)", min_value=0, max_value=10000, value=0, step=100)

    total_y = consumption + investment + government_spending + (exports - imports)

    st.subheader("Hasil Simulasi")
    st.write(f"**GDP (Produk Domestik Bruto): {total_y}**")

    data = {
        "Komponen": ["(C)", "(I)", "(G)", "(X)", "(M)", "Total Y"],
        "Nilai": [consumption, investment, government_spending, exports, imports, total_y]
    }
    df = pd.DataFrame(data)
    st.write(df)

    st.subheader("Penjelasan Perubahan")
    if consumption > 5000:
        st.markdown(f"🔵 **Konsumsi meningkat**. Peningkatan konsumsi ({consumption}) menunjukkan daya beli masyarakat yang lebih tinggi, berkontribusi positif terhadap GDP.")
    elif consumption < 5000:
        st.markdown(f"🔵 **Konsumsi menurun**. Penurunan konsumsi ({consumption}) dapat mengindikasikan lemahnya daya beli masyarakat, sehingga berdampak negatif pada GDP.")

    if investment > 2000:
        st.markdown(f"🟢 **Investasi meningkat**. Dengan investasi sebesar {investment}, akan ada lebih banyak modal untuk produksi, yang berpotensi meningkatkan GDP.")
    elif investment < 2000:
        st.markdown(f"🟢 **Investasi menurun**. Investasi yang lebih rendah ({investment}) dapat memperlambat pertumbuhan ekonomi karena kurangnya dana untuk ekspansi bisnis.")

    if exports > imports:
        st.markdown(f"🟣 **Surplus perdagangan**. Ekspor ({exports}) lebih besar dari impor ({imports}), sehingga menghasilkan surplus perdagangan yang memperkuat GDP.")
    elif exports < imports:
        st.markdown(f"🟣 **Defisit perdagangan**. Ekspor ({exports}) lebih kecil dari impor ({imports}), yang dapat menurunkan kontribusi perdagangan terhadap GDP.")

    st.subheader("Visualisasi Diagram")
    fig, ax = plt.subplots()
    ax.bar(data["Komponen"][:-1], data["Nilai"][:-1], color=["blue", "green", "orange", "purple", "red"])
    ax.set_title("Komponen Ekonomi dan Nilainya")
    ax.set_ylabel("Nilai")
    ax.set_xlabel("Komponen Ekonomi")
    st.pyplot(fig)

    st.markdown("""
    Cobalah mengubah nilai pada kotak input di atas untuk melihat bagaimana masing-masing komponen memengaruhi **GDP**, serta baca penjelasan di atas untuk memahami dampaknya.
    """)
    
# pendapatan perkapita
with tab6:
    def pendapatan_perkapita():
        st.title("Pendapatan Perkapita")
    
    st.header("Pengertian Pendapatan Perkapita")
    st.markdown("""
    Pendapatan perkapita adalah ukuran rata-rata pendapatan yang diperoleh per orang dalam suatu negara dalam periode tertentu. 
    Pendapatan perkapita sering digunakan untuk mengukur tingkat kesejahteraan suatu negara.
    """)

    st.header("Rumus Pendapatan Perkapita")
    st.info("P = Y/N")
    st.markdown("""
    Di mana:
    - **P** = Pendapatan Perkapita  
    - **Y** = Pendapatan Nasional  
    - **N** = Jumlah Penduduk  
    """)

    st.header("Cara Menghitung")
    st.markdown("""
    Sebuah negara memiliki total pendapatan nasional sebesar Rp5.000.000.000.000 dalam satu tahun.
    Jumlah penduduk negara tersebut adalah 250 juta orang.
    Berapakah pendapatan perkapita negara tersebut?
                
    Penjelasan:

    P = Y/N

    P = 5.000.000.000.000/250.000.000

    P = 20.000.
    

    Pendapatan perkapita negara tersebut adalah 20.000.
    """)

    st.info("Ketika jumlah penduduk bertambah tetapi pendapatan nasional tetap, pendapatan per kapita menjadi lebih kecil. Ini menunjukkan bahwa pertumbuhan jumlah penduduk dapat mempengaruhi kesejahteraan rata-rata masyarakat jika tidak diimbangi dengan peningkatan pendapatan nasional.")

    st.header("Latihan Soal")
    st.markdown("""
    **Soal:**  
    Jika Pendapatan Nasional adalah **1.500.000** dan jumlah penduduk adalah **75**, hitunglah pendapatan perkapita.
    """)

    user_answer = st.number_input("Masukkan jawaban Anda:", min_value=0, step=1)
    if st.button("Kirim Jawaban"):
        correct_answer = 1500000 / 75
        if user_answer == correct_answer:
            st.success("Jawaban Anda benar! 🎉")
        else:
            st.error(f"Jawaban Anda salah. Jawaban yang benar adalah {correct_answer:.2f}.")

# distribusi pendapatan
with tab7:
    def distribusi_pendapatan():
     st.title("Distribusi Pendapatan")

    st.header("Pengertian Distribusi Pendapatan")
    st.markdown("""
    Distribusi pendapatan mengacu pada persebaran pendapatan di suatu wilayah.
    Ketimpangan dalam distribusi ini dapat menyebabkan kemakmuran hanya dirasakan oleh golongan tertentu.
    Untuk mengukur ketimpangan tersebut, salah satu indikator yang digunakan adalah Koefisien Gini, yang berkisar
    antara 0 (pemerataan sempurna) hingga 1 (ketimpangan sempurna). 
    """)

    st.header("Koefisien Gini")
    st.markdown("""
   Koefisien Gini memiliki kriteria ketimpangan sebagai berikut:
    - 0 = Distribusi merata sempurna
    - 0 - 0,4 = Ketimpangan rendah
    - 0,4 - 0,5 = Ketimpangan sedang
    - 0,5 - 1 = Ketimpangan tinggi
    - 1 = Distribusi tidak merata sempurna 
    """)

    st.header("Kriteria Bank Dunia")
    st.markdown("""
    - Suatu negara yang kelompok 40% penduduk termiskinnya memperoleh pendapatan lebih kecil dari 12%,
    maka negara tersebut berada di tingkat ketimpangan yang tinggi dalam distribusi pendapatan.
    - Suatu negara yang kelompok 40% penduduk termiskinnya pendapatannya diantara 12% - 17 %, 
    maka negara tersebut berada di tingkat ketimpangan sedang dalam distribusi pendapatan.
    - Suatu negara yang kelompok 40 penduduk termiskinnya pendapatannya lebih dari 17%, 
    maka negara tersebut berada di tingkat ketimpangan rendah dalam distribusi pendapatan.

    """)

    st.header("Kurva Lorenz")
    st.image('kurva_lorenz.png')
    st.markdown("""
    **Kurva Lorenz** adalah grafik yang menunjukkan distribusi pendapatan kumulatif dalam suatu populasi.  
    - Garis diagonal (45 derajat) menunjukkan distribusi pendapatan yang sepenuhnya merata.  
    - Kurva Lorenz menunjukkan ketimpangan ketika semakin jauh kurva dari garis diagonal, semakin besar ketimpangan.  
    """)

    def gini_coefficient(income):
        sorted_income = np.sort(income)
        n = len(income)
        cum_income = np.cumsum(sorted_income)
        total_income = cum_income[-1]
        
        gini = (2 / n) * np.sum((np.arange(1, n + 1) * sorted_income)) / total_income - (n + 1) / n
        return gini, sorted_income, cum_income

    def plot_lorenz_curve(sorted_income, cum_income):
        lorenz_x = np.arange(1, len(sorted_income) + 1) / len(sorted_income)
        lorenz_y = cum_income / cum_income[-1]
        
        plt.figure(figsize=(8, 6))
        plt.plot(lorenz_x, lorenz_y, label="Kurva Lorenz", color='blue')
        plt.plot([0, 1], [0, 1], linestyle='--', color='red', label="Garis Kesetaraan")
        plt.title("Kurva Lorenz")
        plt.xlabel("Persentase Penduduk (%)")
        plt.ylabel("Persentase Pendapatan (%)")
        plt.legend()
        plt.grid(True)
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        return img_buf

    st.title("Perhitungan Koefisien Gini dan Kurva Lorenz")

    st.write("Masukkan data pendapatan individu (pisahkan dengan koma, misalnya: 200, 500, 800, 1000)")
    income_input = st.text_input("Pendapatan (misalnya: 200, 500, 800, 1000)")

    if st.button("Kirim"):
        if income_input:
            try:
                income = [float(i) for i in income_input.split(",")]
                
                if len(income) < 2:
                    st.error("Masukkan lebih dari satu nilai pendapatan.")
                else:
                    gini, sorted_income, cum_income = gini_coefficient(income)
                    
                    st.success(f"Koefisien Gini: {gini:.4f}")
                    
                    img_buf = plot_lorenz_curve(sorted_income, cum_income)
                    st.image(img_buf, caption="Kurva Lorenz", use_container_width=True)
                    
                    st.write("""
                    **Kurva Lorenz** menggambarkan distribusi pendapatan di suatu negara atau wilayah. 
                    Pada grafik ini, sumbu X menunjukkan persentase kumulatif penduduk yang terurut berdasarkan pendapatan, 
                    sementara sumbu Y menunjukkan persentase kumulatif pendapatan yang diperoleh oleh persentase penduduk tersebut.

                    **Garis Kesetaraan** (garis merah putus-putus) menggambarkan distribusi pendapatan yang sempurna merata. 
                    Semakin jauh kurva Lorenz dari garis kesetaraan, semakin tinggi ketimpangan distribusi pendapatan.

                    Koefisien Gini, yang dihitung berdasarkan area antara kurva Lorenz dan garis kesetaraan, mengukur ketimpangan pendapatan:
                    - Nilai Gini = 0 menunjukkan kesetaraan sempurna (semua orang memiliki pendapatan yang sama).
                    - Nilai Gini = 1 menunjukkan ketimpangan maksimal (satu orang memiliki semua pendapatan).
                    """)

                    df = pd.DataFrame({
                        'Pendapatan': sorted_income,
                        'Pendapatan Kumulatif': cum_income
                    })

                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)

                    pdf.cell(200, 10, txt="Hasil Perhitungan Koefisien Gini dan Kurva Lorenz", ln=True, align='C')
                    pdf.ln(10)

                    pdf.multi_cell(0, 10, txt=f"Koefisien Gini: {gini:.4f}\n\n")

                    pdf.ln(5)
                    pdf.cell(200, 10, txt="Data Pendapatan dan Kumulatif Pendapatan:", ln=True)
                    pdf.ln(5)

                    for i in range(len(sorted_income)):
                        pdf.cell(95, 10, txt=f"Pendapatan: {sorted_income[i]}", border=1)
                        pdf.cell(95, 10, txt=f"Kumulatif Pendapatan: {cum_income[i]}", border=1)
                        pdf.ln()

                    img_path = "kurva_lorenz.png"
                    img_buf.seek(0)
                    with open(img_path, "wb") as img_file:
                        img_file.write(img_buf.read())
                    pdf.ln(5)
                    pdf.image(img_path, x=10, w=190)

                    pdf_output = io.BytesIO()
                    pdf.output(pdf_output)
                    pdf_output.seek(0)

                    st.download_button(
                        label="Unduh Hasil Simulasi",
                        data=pdf_output,
                        file_name="Hasil Simulasi Kurva Lorenz.pdf",
                        mime="application/pdf"
                    )

            except ValueError:
                st.error("Tolong masukkan angka yang valid.")
        else:
            st.error("Masukkan data pendapatan terlebih dahulu.")


# latihan soal
    questions = {
    "Mudah": [
        {
            "question": """Perhatikan beberapa pernyataan berikut ini!       
        1)    Pendapatan Nasional adalah jumlah seluruh pendapatan dan pengeluaran seluruh pelaku ekonomi suatu negara,
        2)    Konsumsi masyarakat tidak berpengaruh dalam jumlah Pendapatan Nasional suatu negara,
        3)    Pendapatan Nasional suatu negara tidak ditentukan oleh total produksi barang atau jasa, melainkan diukur dengan pendapatan dan pengeluaran pelaku ekonominya,
        4)    Pendapatan Nasional suatu negara tidak hanya ditentukan oleh total produksi barang atau jasa, melainkan juga diukur dengan pendapatan dan pengeluaran pelaku ekonominya,
        5)    Pendapatan Nasional dapat dihitung dengan menjumlahkan seluruh pendapatan (r, w, i, p) yang diperoleh oleh produsen.
        Dari pernyataan tersebut, pengertian Pendapatan Nasional yang BENAR ditunjukkan oleh nomor?""",
            "options": ["1, 2, dan 3", "1, 3, dan 5", "1, 4, dan 5", "2, 3, dan 5"],
            "answer": "1, 4, dan 5",
            "explanation": """Pengertian Pendaptan Nasional terdapat pada nomor 1) Pendapatan Nasional adalah jumlah seluruh pendapatan dan pengeluaran seluruh pelaku ekonomi suatu negara,
            4) Pendapatan Nasional suatu negara tidak hanya ditentukan oleh total produksi barang atau jasa, melainkan juga diukur dengan pendapatan dan pengeluaran pelaku ekonominya, dan,
            5) Pendapatan Nasional dapat dihitung dengan menjumlahkan seluruh pendapatan (r, w, i, p) yang diperoleh oleh produsen."""
        },
        {
            "question": "Apa yang dimaksud dengan Gross Domestic Product (GDP)?",
            "options": ["Total pendapatan warga negara suatu negara", "Total pendapatan yang diperoleh oleh pemerintah suatu negara", "Total pendapatan yang dihasilkan dalam batas wilayah suatu negara", "Total pendapatan yang berasal dari luar negeri yang dimiliki oleh warga negara suatu negara"],
            "answer": "Total pendapatan yang dihasilkan dalam batas wilayah suatu negara",
            "explanation": "Gross Domestic Bruto adalah Total pendapatan yang dihasilkan oleh Warga Negara dan Warga Negara Asing dalam batas wilayah suatu negara"
        },
        {
            "question": """Suatu negara memiliki pendapatan nasional sebesar Rp2.000 triliun pada tahun tertentu.
                        Jika total konsumsi rumah tangga adalah Rp1.200 triliun dan investasi sebesar Rp400 triliun,
                        berapa besar nilai ekspor netto negara tersebut?""",
            "options": ["Rp400 triliun", "Rp200 triliun", "Rp800 triliun", "Rp600 triliun"],
            "answer": "Rp400 triliun",
            "explanation": """Y = C + I + G + X
                2.000 = 1.200 + 400 + X
                2.000 - 1.200 - 400 = X
                400 = X
                Ekspor netto adalah selisih antar ekspor dan impor, karena data impor tidak ada dalam soal, maka tidak usah ditulis. Jadi ekspor netto sebesar Rp400 triliun"""
        },
        {
            "question": """Dalam suatu negara pada tahun 2021, diketahui GDP sebesar Rp10 triliun, GNP sebesar Rp11 triliun,
            dan depresiasi bersih Rp1 triliun. Hitunglah Net National Product (NNP) pada tahun tersebut!""",
            "options": ["Rp10 triliun", "Rp11 triliun", "Rp9 triliun", "Rp12 triliun"],
            "answer": "Rp10 triliun",
            "explanation": """NNP = GNP - Depresiasi
                NNP = 11 triliun - 1 triliun
                NNP = 10 triliun"""
        },
        {
            "question": """Berikut adalah beberapa komponen pendapatan nasional suatu negara.
            Manakah di antara berikut yang termasuk dalam komponen pendapatan primer?""",
            "options": ["Gaji dan Upah", "Bunga dan Pendapatan", "Pajak Penghasilan", "Subsidi Pemerintah"],
            "answer": "Gaji dan Upah",
            "explanation": "Komponen pendapatan primer melibatkan pendapatan yang diterima oleh individu secara langsung dari faktor produksi, seperti gaji dan upah dari pekerjaan."
        },
        {
            "question": """Tentukan Pendapatan Nasional suatu negara dengan menggunakan pendekatan Pengeluaran berdasarkan data berikut:
                        ·        Konsumsi Rumah Tangga: 3.000.000
                        ·        Investasi: 1.500.000
                        ·        Belanja Pemerintah: 2.000.000
                        ·        Ekspor: 1.200.000
                        ·        Impor: 800.000
                        Total pendapatan nasionalnya adalah?""",
            "options": ["Rp5.900.000", "Rp5.700.000", "Rp6.900.000", "Rp5.400.000"],
            "answer": "Rp6.900.000",
            "explanation": """Pendapatan Nasional berdasarkan pendekatan Pengeluaran dapat dihitung dengan rumus:
                    Y = C + I + G + (X - M)
                    Y = 3.000.000 + 1.500.000 + 2.000.000 + (1.200.000 – 800.000)
                    Y = 6.500.000 + (400.000)
                    Y = 6.900.000"""
        },
        {
            "question": """Sebuah pabrik memproduksi 1.000 unit sepeda motor senilai Rp 10.000.000 per unit. Pabrik tersebut juga memberikan upah kepada pekerja sebesar Rp 2.000.000.
            Berapa total pendapatan nasional pendekatan produksi?""",
            "options": ["Rp12.000.000", "Rp10.000.000", "Rp20.000.000", "Rp22.000.000"],
            "answer": "Rp10.000.000",
            "explanation": """Y = P x Q
                    Y = 10.000 X 1.000
                    Y = 10.000.000"""
        },
        {
            "question": """Penduduk suatu negara adalah 5.000.000 orang, dan total pendapatan nasionalnya adalah 25.000.000.
            Berapakah Pendapatan Perkapita negara tersebut?""",
            "options": ["5", "10", "15", "20"],
            "answer": "5",
            "explanation": """Pendapatan Per-Kapita = Pendapatan nasional : Jumlah penduduk
                    Y = 25.000.000 : 5.000.000
                    Y = 5"""
        },
        {
            "question": """Dalam suatu negara, Pendapatan Nasional adalah 40.000.000 dan Pendapatan Perkapita adalah 8.000.
            Berapakah jumlah penduduk negara tersebut?""",
            "options": ["2.500", "4.000", "5.000", "10.000"],
            "answer": "5.000",
            "explanation": """Y = Pendapatan nasional : Jumlah penduduk (JP)
                    8.000 = 40.000.000 : JP
                    JP = 40.000.000 : 8.000
                    JP = 5.000"""
        },
        {
            "question": """Jika tingkat inflasi suatu negara meningkat secara signifikan,
            bagaimana hal ini dapat memengaruhi pengukuran pendapatan nasional?""",
            "options": ["Meningkatkan nilai pendapatan nasional", "Meningkatkan daya beli masyarakat", "Mengurangi nilai pendapatan nasional", "Tidak memiliki pengaruh pada pengukuran pendapatan nasional"],
            "answer": "Mengurangi nilai pendapatan nasional",
            "explanation": "Inflasi yang tinggi dapat mengurangi daya beli uang dan, akibatnya, mengurangi nilai pendapatan nasional dalam istilah nyata."
        },
        {
            "question": """Apa yang dimaksud dengan konsep "pengeluaran agregat" dalam perhitungan pendapatan nasional?""",
            "options": ["Total pengeluaran pemerintah dalam suatu tahun", "Total pengeluaran masyarakat dalam suatu tahun", "Total pengeluaran perusahaan dalam suatu tahun", "Total pengeluaran seluruh sektor ekonomi dalam suatu tahun"],
            "answer": "Total pengeluaran seluruh sektor ekonomi dalam suatu tahun",
            "explanation": """Pengeluaran agregat adalah jumlah total pengeluaran dari seluruh sektor ekonomi dalam suatu negara dalam satu tahun, termasuk konsumsi, investasi, pengeluaran pemerintah, dan ekspor neto"""
        },
        {
            "question": """Pada suatu negara, distribusi pendapatan dapat diukur menggunakan Indeks Gini. Jika nilai Indeks Gini adalah 0,2,
            apa yang dapat disimpulkan tentang distribusi pendapatan di negara tersebut?""",
            "options": ["Distribusi pendapatan sangat merata", "Distribusi pendapatan cukup merata", "Distribusi pendapatan tidak merata", "Distribusi pendapatan sangat tidak merata"],
            "answer": "Distribusi pendapatan tidak merata",
            "explanation": """Nilai Indeks Gini berkisar antara 0 dan 1. Nilai 0 mengindikasikan distribusi yang sangat merata, sedangkan nilai 1 menunjukkan distribusi yang sangat tidak merata. Nilai 0,2 
            menunjukkan bahwa distribusi pendapatan di negara tersebut tidak merata, tetapi juga tidak terlalu ekstrem"""
        },
          {
            "question": """Pada sebuah negara, 20% penduduk memiliki 80% dari total pendapatan nasional.
            Apa yang dapat disimpulkan tentang distribusi pendapatan di negara ini?""",
            "options": ["Distribusi pendapatan sangat merata", "Distribusi pendapatan cukup merata", "Distribusi pendapatan tidak merata", "Distribusi pendapatan sangat tidak merata"],
            "answer": "Distribusi pendapatan sangat tidak merata",
            "explanation": "Pernyataan ini mengindikasikan adanya ketidakmerataan yang sangat besar dalam distribusi pendapatan, dengan sebagian kecil penduduk memiliki sebagian besar pendapatan nasional."
        },
          {
            "question": """Seorang petani memperoleh pendapatan sebesar Rp 50.000.000 dalam setahun. Dia menyimpan Rp 10.000.000 dan mengeluarkan sisanya untuk konsumsi.
            Berapa kontribusi petani ini terhadap pendapatan nasional pendekatan pendapatan?""",
            "options": ["Rp50.000.000", "Rp40.000.000", "Rp10.000.000", "Rp60.000.000"],
            "answer": "Rp50.000.000",
            "explanation": """Pendapatan nasional pendekatan pendapatan mengukur total pendapatan (r, w, I, p) yang diterima oleh semua faktor produksi dalam suatu perekonomian. Dalam hal ini, pendapatan petani sebesar 
            Rp 50.000.000 adalah bagian dari pendapatan nasional"""
        },
          {
            "question": """Apa yang dimaksud dengan pendapatan nasional bruto (Gross National Product/GNP?""",
            "options": ["Pendapatan total yang diterima oleh pemerintah dalam satu tahun", "Pendapatan total yang diterima oleh perusahaan dalam satu tahun", "Pendapatan total yang diterima oleh rakyat dalam satu tahun", "Pendapatan total yang diterima oleh penduduk suatu negara dalam dan luar negeri dalam satu tahun"],
            "answer": "Pendapatan total yang diterima oleh penduduk suatu negara dalam dan luar negeri dalam satu tahun",
            "explanation": """Pendapatan nasional bruto (Gross National Income/GNI) adalah jumlah total pendapatan yang diterima oleh penduduk suatu negara, baik dalam negeri maupun dari luar negeri, dalam satu tahun."""
        },
    ]
    
}

with tab8:
    st.header("Soal-soal Latihan Pendapatan Nasional")
    
    all_questions = questions["Mudah"]

    user_answers = {}
    with st.form("quiz_form"):
        name = st.text_input("Nama Lengkap")
        class_name = st.text_input("Kelas")
        for idx, q in enumerate(all_questions):
            st.subheader(f"Soal {idx + 1}")
            user_answers[idx] = st.radio(q["question"], q["options"], key=f"q{idx}")
        
        submitted = st.form_submit_button("Kirim Jawaban")
    
    if submitted:
        score = 0
        soal_salah = []
        for idx, q in enumerate(all_questions):
            if user_answers[idx] == q["answer"]:
                score += 1
            else:
                soal_salah.append(f"Soal {idx + 1}")

        st.subheader("Hasil Penilaian")
        st.info(f"Skor Anda: {score} dari {len(all_questions)}")
        
        st.subheader("Pembahasan Soal")
        for idx, q in enumerate(all_questions):
            st.write(f"**Soal {idx + 1}: {q['question']}**")
            if user_answers[idx] == q["answer"]:
                st.success(f"Jawaban Anda: {user_answers[idx]} ✅")
            else:
                st.error(f"Jawaban Anda: {user_answers[idx]} ❌")
                st.markdown(f"**Penjelasan**: {q['explanation']}")
        
        # Load dataset dan rekomendasi
        try:
            materi = pd.read_csv('Somat.csv', sep=';', on_bad_lines='skip')
            jawaban_siswa = pd.read_csv('Dataset.csv', sep=';', on_bad_lines='skip')
            print(materi)
            
            dataset_gabungan = pd.merge(materi, jawaban_siswa, on='Atribut', how='left')
            dataset_gabungan['Gabungan'] = dataset_gabungan['Pertanyaan'] + " " + dataset_gabungan['Atribut']
            
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(dataset_gabungan['Gabungan'])

            knn = NearestNeighbors(n_neighbors=3, metric='cosine')
            knn.fit(tfidf_matrix)

            required_columns = ['Atribut', 'Gabungan']
            missing_columns = [col for col in required_columns if col not in dataset_gabungan.columns]
            if missing_columns:
                st.error(f"Kolom berikut tidak ditemukan di dataset gabungan: {missing_columns}")


            def rekomendasi_materi(soal_salah):
                index_salah = dataset_gabungan[
                    dataset_gabungan['Soal'].isin(soal_salah)
                ].index
                if len(index_salah) == 0:
                    return []
                distances, indices = knn.kneighbors(tfidf_matrix[index_salah])
                rekomendasi = dataset_gabungan.iloc[indices.flatten()]['Atribut']
                return rekomendasi.unique()

            rekomendasi = rekomendasi_materi(soal_salah)
            index_salah = dataset_gabungan[
            dataset_gabungan['Soal'].isin(soal_salah)
            ].index

            if "Expected" in dataset_gabungan.columns:
                ground_truth = dataset_gabungan[
                    dataset_gabungan['Soal'].isin(soal_salah)
                ]['Expected']

                rekomendasi_benar = set(ground_truth) & set(rekomendasi)
                akurasi_model = len(rekomendasi_benar) / len(ground_truth) * 100 if len(ground_truth) > 0 else 0

                st.subheader("Akurasi Model Prediksi")
                st.info(f"Akurasi Model: {akurasi_model:.2f}%")
                if len(index_salah) > 0:
                    distances, indices = knn.kneighbors(tfidf_matrix[index_salah])
                else:
                    st.warning("Tidak ada data untuk diproses oleh KNN.")

            distances, indices = knn.kneighbors(tfidf_matrix[index_salah])

            st.info("Selamat ya, kamu sudah menyelesaikan latihan soal Pendapatan Nasional!")

            st.subheader("Rekomendasi materi yang harus kamu pelajari lagi, nih!")
            if len(rekomendasi) > 0:
                for materi in rekomendasi:
                    st.write(f"- {materi}")
            else:
                st.write("Tidak ada rekomendasi yang dapat diberikan.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memuat data: {e}")

        def generate_excel(name, class_name, user_answers, all_questions, score):
            data = {
                'Nama Lengkap': [],
                'Kelas': [],
                'Soal': [],
                'Jawaban Pengguna': [],
                'Jawaban Benar': [],
                'Penjelasan': [],
                'Status': [],
                'Rekomendasi Materi': []
            }
            
            for idx, q in enumerate(all_questions):
                data['Nama Lengkap'].append(name)
                data['Kelas'].append(class_name)
                data['Soal'].append(f"Soal {idx + 1}: {q['question']}")
                data['Jawaban Pengguna'].append(user_answers[idx])
                data['Jawaban Benar'].append(q["answer"])
                data['Penjelasan'].append(q["explanation"])
                status = "Benar" if user_answers[idx] == q["answer"] else "Salah"
                data['Status'].append(status)
                data['Rekomendasi Materi'].append(rekomendasi)

            data['Nama Lengkap'].append(name)
            data['Kelas'].append(class_name)
            data['Soal'].append("Skor Total")
            data['Jawaban Pengguna'].append(score)
            data['Jawaban Benar'].append(f"{score} dari {len(all_questions)}")
            data['Penjelasan'].append("")
            data['Status'].append("")
            data['Rekomendasi Materi'].append(rekomendasi)

            df = pd.DataFrame(data)

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name="Hasil Latihan")
            excel_buffer.seek(0)
            
            return excel_buffer

        if submitted:
            score = 0
            for idx, q in enumerate(all_questions):
                if user_answers[idx] == q["answer"]:
                    score += 1

            excel_output = generate_excel(name, class_name, user_answers, all_questions, score)

            st.download_button(
                label="Download Hasil Latihan Soal",
                data=excel_output,
                file_name= f"Hasil Latihan {name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# fakta-fakta
with tab9:
    st.header("Fakta-fakta menarik yang harus kamu ketahui tentang Pendapatan Nasional, nih!")
    st.subheader("Fakta 1")
    st.image('ue.jpg')
    st.info("""
    **Tahukah kamu?**
            

    Kegiatan ekonomi informal tidak selalu tercatat dalam GDP, lho.
    Di beberapa negara berkembang, kegiatan ekonomi informal ini bisa menyumbang lebih dari 40% dari ekonomi. Contohnya, pedagang kaki lima,
    buruh lepas, atau usaha rumahan yang tidak memiliki izin usaha. Usaha-usaha tersebut bisa menyediakan lapangan kerja dan membantu memenuhi kebutuhan sehari-hari banyak orang.
    Fenomena ini disebut "Underground Economy" atau "Ekonomi Bawah Tanah". 
        
    Kegiatan informal ini banyak ditemukan di beberapa negara berkembang, seperti India, Nigeria, Kenya, Filipina, dan negara kita tercinta, Indonesia.
  
            
    Teman-teman bayangkan saja. kalau ekonomi informal bisa sepenuhnya tercatat, dampaknya pasti luar biasa. Angka GDP bisa melonjak!
    Pemerintah juga dapat pendapatan pajak lebih banyak, yang artinya ada lebih banyak dana untuk membangun infrastruktur,
    memperbaiki layanan kesehatan, dan meningkatkan pendidikan. Tidak cuma itu, pekerja yang sebelumnya tidak terlindungi akan punya akses ke jaminan sosial, gaji yang layak, dan perlindungan hukum.
    Usaha kecil yang tadinya sulit berkembang juga bisa lebih mudah dapat pinjaman atau bantuan.
    Bayangkan saja kios kecil di pinggir jalan, yang selama ini hanya bertahan dari hari ke hari, bisa punya peluang untuk tumbuh jadi bisnis besar.       
    """)

    st.subheader("Fakta 2")
    st.image('ntl..png')
    st.info("""
    **Tahukah kamu?**
            

    Ada cara unik, lho, untuk mengukur pendapatan nasional di suatu negara, yaitu menggunakan pencahayaan malam hari yang ditangkap satelit!.
    Cara ini menggunakan sensor Nighttime Light (NTL) atau Pencahayaan Malam.
            
    
    Dilansir dari Mapid.co.id, salah satu indikator tingkat perekonomian yang baik dalam suatu wilayah dapat dikenali dari tingkat aktivitas komersial pada malam hari,
    yang tercermin dalam data NTL. Data ini dapat memberikan analisis komparasi wilayah yang memiliki tingkat aktivitas ekonomi yang tinggi dan rendah pada waktu tertentu. Wilayah dengan cahaya malam yang
    lebih terang seringkali menunjukkan pusat-pusat kegiatan ekonomi yang kuat,
    sementara wilayah dengan cahaya malam yang lebih redup cenderung memiliki aktivitas ekonomi yang rendah.
    Bagaimana, menarik banget bukan?
    """)

    st.subheader("Fakta 3")
    st.image('greengdp.jpg')
    st.info("""
    **Tahukah kamu?**
            

    Terdapat cara baru, lho, dalam menghitung GDP, yaitu menggunakan konsep "Green GDP" atau "GDP Hijau".
    
            
    Berdasarkan hasil webinar "The Quest for Green GDP" yang dilaksanakan pada 20 April 2023 oleh Divisi Statistik Perserikatan Bangsa-Bangsa dan Bank Dunia,
    GDP biasa hanya mengukur nilai moneter barang dan jasa yang diproduksi dalam setahun untuk mengukur pertumbuhan ekonomi, tetapi pertumbuhan ekonomi yang terjadi dengan mengorbankan alam justru bisa merugikan masa depan.
    Green GDP diperkenalkan pada akhir 1980-an untuk memperhitungkan dampak negatif ekonomi terhadap lingkungan.
    Intinya, konsep Green GDP yaitu cara mengukur GDP dengan menghitung dampak lingkungan dari aktivitas ekonomi. Jika suatu aktivitas membuat kerusakan lingkungan (seperti polusi), nilai GDPnya bisa dikurangi.
    """)

    st.subheader("Fakta 4")
    st.image('pp.jpg')
    st.info("""
    **Tahukah kamu?**

            
    Tahun 2016, terdapat dokumen rahasia yang bocor, bernama Panama Papers. Berdasarkan buku 'The Panama Papers: Breaking Story of How the Rich & Powerful Hide Their Money' yang ditulis oleh Bastian Obermayer dan Frederik Obermaier, bahwa Panama Papers ini
    pertama kali disampaikan kepada jurnalis oleh seorang whistleblower anonim yang menggunakan nama samaran John Doe. Melalui saluran komunikasi terenkripsi, John Dose
    menghubungi Bastian Obermayer (Penulis buku terseut yang adalah seorang jurnalis) dari Suddeutzche Zeitung, dan menawarkan akses ke sejumlah besar dokumen yang berasal dari firma
    hukum Mossac Fonseca. Dokumen-dokumen tersebut berisi rincian perusahaan offshore, pemiliknya, dan bagaimana kekayaan mereka disembunyikan dari otoritas pajak.
            
    Perusahaan offshore atau perusahaan cangkang adalah perusahaan yang didirikan di luar negara tempat pemiliknya tinggal atau beroperasi. Biasanya, perusahaan ini didirikan di negara
    yang memiliki regulasi perpajakan yang lebih menguntunhkan atau kebijakan bisnis yang lebih longgar. Mossack Fonseca adalah sebuah firma hukum yang berperan sebagai perantara utama dalam membantu individu atau perusahaan untuk mendirikan perusahaan offshore.
    Mereka menyediakan layanan untuk membuat dan mengelola perusahaan cangkang yang memungkinkan klien untuk menyembunyikan kekayaan dan menghindari pajak. Mozzack Fonseca juga mengurus berbagai dokumen dan izin yang diperlukan, sehingga klien mereka bisa beroperasi di luar jangkaun hukum
    negara mereka. Hal ini termasuk mengatur rekening bank di negara-negara yang memiliki kebijakan pajak rendah atau tidak ada pajak sama sekali. Jadi, mereka menjadi kunci dalam jaringan penghindaran pajak global yang terungkap dalam skandal Panama Papers.
            
    Awal mula pengungkapan Panama Papers ini melibatkan sekelompok jurnalis investigasi, terutama dari Suudeutzche Zeitung (salah satu surat kabar harian terbesar dan terkemuka di Jerman), yang dipimpin oleh Bastian Obermayer dan Frederik Obermaier. Investigasi terhadap dokumen ini
    melibatkan kolaborasi antara jurnalis dari berbagai negara yang dipimpin oleh International Consartium of Investigative Jounalists (ICIJ).
    Berikut adalah isi dari Panama Papers yang terungkap:
    - Total dokumen yang terekspos sebanyak 11,5 juta dokumen.
    - Nama-nama perusahaan, alamat, dan detail pendaftaran 214.000 perusahaan offshore yang didirikan oleh Mossac Fonseca.
    - Nama-nama individu atau entitas yang memiliki atau terlibat dengan perusahaan-perusahaan tersebut.
    - Rekening bank, laporan keuangan, dan transaksi keuangan.
    - Data dari ICIJ, ada 2.961 nama warga negara Indonesia yang terdaftar pada dokumen tersebut.

            
    Panama Papers ini sangat erat kaitannya dengan konsep Tax Heaven atau Surga Pajak. Tax Heaven adalah negara atau wilayah yang memiliki kebijakan pajak yang sangat rendah atau bahkan tidak ada pajak yang ditarik. Padahal, untuk negara yang sangat mengandalkan pajak sebagai pendapatan negara pasti sangat dirugikan.
    Indonesia misalnya, berdasarkan data dari Badan Pusat Statistika (BPS), pendapatan pajak Indonesia menyentuh 82,4% dari pendapatan nasional secara keseluruhan. Gede banget, kan, angkanya!.
    Oleh karena itu, Tax Heaven ini sangat merugikan negara, karena memunginkan individu atau perusahaan kaya untuk menghindari kewajiban membayar pajak mereka di negara asal.
            
    Orang kaya dapat menyimpan uang mereka di Tax Heaven dan menghindari pajak, sementara masyarakat biasa yang tidak memiliki akses ke cara-cara tersebut tetap membayar pajak mereka dengan susah payah. 'Nggak adil banget, kan?
    """)

# referensi bacaan
with tab10:
    st.header("Referensi bacaan materi Pendapatan Nasional")
    st.markdown("""
    Kamu bisa memperdalam pengetahuan kamu tentang Pendapatan Nasional dengan membaca referensi berikut ini:
    
    - E-Modul Ekonomi XI, Kemendikbud 2019, Penyusun: Wahyu Rini Mulyasari, S.Pd.
    - Modul Pembelajaran SMA Ekonomi XI, Kemendikbud 2020, Penyusun: Anna Monalita de Fretes, S.Pd., M.Pd.
    """)

