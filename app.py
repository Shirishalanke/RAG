from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb




pdf_file = "Sample.pdf"

print("Reading PDF...")

reader = PdfReader(pdf_file)

text = ""

for page_number, page in enumerate(reader.pages):

    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"


print("Number of pages:", len(reader.pages))
print("Total characters extracted:", len(text))




if not text.strip():

    print("\nERROR: No text could be extracted from the PDF.")

    print("The PDF may be scanned/image-based.")

    exit()




def create_chunks(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

    return chunks


chunks = create_chunks(text)


print("Number of chunks:", len(chunks))




print("\nFirst chunk:")
print("--------------------")
print(chunks[0])
print("--------------------")




print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")



print("\nGenerating embeddings...")

embeddings = model.encode(chunks)

print("Embeddings generated.")
print("Number of embeddings:", len(embeddings))



print("\nCreating ChromaDB...")

client = chromadb.PersistentClient(
    path="./chroma_db" #where the database will be stored
)



collection = client.get_or_create_collection(
    name="documents"
)



ids = []

for i in range(len(chunks)):

    ids.append(f"chunk_{i}")



collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings.tolist()
)


print("\n====================================")
print("PDF stored successfully in ChromaDB!")
print("====================================")



while True:

    question = input("\nAsk a question (type 'exit' to stop): ")

    if question.lower() == "exit":
        print("Program stopped.")
        break


    
    question_embedding = model.encode(
        [question]
    )


    
    results = collection.query(
        query_embeddings=question_embedding.tolist(), #storing in to the embeddings list
        n_results=3
    )


    
    print("\nRelevant information:")
    print("========================")


    documents = results["documents"][0]

    for i, document in enumerate(documents):

        print(f"\nChunk {i + 1}")
        print("------------------------")
        print(document)


print("\nThank you!")