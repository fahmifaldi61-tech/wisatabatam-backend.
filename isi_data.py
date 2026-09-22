import os
import shutil
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

db_path = "./chroma_db_local"

# 1. Bersihkan database lama agar tidak ada duplikasi
if os.path.exists(db_path):
    shutil.rmtree(db_path)
    print("Database lama dibersihkan.")

# 2. Muat mesin embedding
print("Memuat mesin embedding...")
embeddings = FastEmbedEmbeddings()
vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

# 3. Data Dummy (ID & EN)
dummy_data = [
    Document(page_content="Pantai Nongsa adalah destinasi wisata bahari di Batam yang menghadap langsung ke Singapura. Cocok untuk liburan keluarga dengan fasilitas resor mewah. Harga tiket masuk area pantai publik sekitar Rp 10.000."),
    Document(page_content="Nongsa Beach is a marine tourism destination in Batam facing Singapore. It is suitable for family holidays with luxury resort facilities. Public beach entrance fee is around Rp 10,000."),
    Document(page_content="Jembatan Barelang adalah ikon Kota Batam yang menghubungkan Pulau Batam, Rempang, dan Galang. Wisatawan sering datang ke sini untuk berfoto dan menikmati kuliner seafood (kelong) di sekitarnya."),
    Document(page_content="Barelang Bridge is the icon of Batam City connecting Batam, Rempang, and Galang islands. Tourists often visit here to take pictures and enjoy local seafood (kelong) nearby.")
]

# 4. Suntikkan ke ChromaDB
print("Menyuntikkan data ke ChromaDB...")
vectorstore.add_documents(dummy_data)
print("Sukses! Data pariwisata Batam sudah siap digunakan.")