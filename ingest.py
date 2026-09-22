import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def setup_all_in_one():
    print("\n=== Memulai Setup RAG Pariwisata Batam (Versi HUGGINGFACE) ===")
    
    # 1. Membuat folder dan file data otomatis
    folder_name = "data_wisata"
    file_path = f"{folder_name}/info_wisata.txt"
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"[+] Folder '{folder_name}' berhasil dibuat.")
        
    if not os.path.exists(file_path):
        # Menyuntikkan data pariwisata Batam
        teks_wisata = """Batam adalah salah satu kota terbesar di Kepulauan Riau yang menjadi pintu gerbang pariwisata Indonesia ketiga setelah Bali dan Jakarta. 
Destinasi wisata populer di Batam antara lain Jembatan Barelang yang ikonik, wisata pantai di Nongsa, serta wisata belanja di Nagoya. 
Selain itu, Batam Tourism Polytechnic (BTP) dan Kelompok Sadar Wisata (Pokdarwis) Kota Batam aktif mengembangkan program Tourism 5.0. Program ini bertujuan untuk meningkatkan kualitas layanan pariwisata berbasis digital dan AI.
Wisata kuliner di Batam sangat beragam, didominasi oleh hidangan laut (seafood) segar yang bisa dinikmati di berbagai restoran pinggir laut (kelong)."""
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(teks_wisata)
        print(f"[+] File '{file_path}' berhasil dibuat dan diisi data wisata.")

    # 2. Proses Ingestion (RAG) ke ChromaDB
    print("\n[Proses] 1. Membaca dokumen wisata Batam...")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    print("[Proses] 2. Memecah teks (chunking)...")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print("[Proses] 3. Mengunduh model AI (satu kali saja) dan menyimpan ke ChromaDB...")
    
    # MENGGUNAKAN HUGGINGFACE (Paling Stabil di Windows)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory="./chroma_db_final"
    )
    print("\n=== SUKSES BESAR! Database vektor pariwisata Batam (ChromaDB) berhasil dibuat! ===")

if __name__ == "__main__":
    setup_all_in_one()