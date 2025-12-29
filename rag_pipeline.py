from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from chromadb.config import Settings  # optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv


# -------------------------------------------------------------------
# 1) PDF loading & splitting helpers (for index-building scripts)
# -------------------------------------------------------------------

def process_all_pdfs(pdf_directory: str) -> List[Any]:
    """
    Load all PDFs from pdf_directory and attach basic metadata.

    NOTE: Use this from a build script / notebook. Do NOT call at import time.
    """
    all_documents: List[Any] = []
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")

    act_name_map = {
        "pharmacy.pdf": "Nepal Pharmacy Council Act, 2057",
        "immunization.pdf": "Immunization Act, 2072",
        "single_women.pdf": "Single Women Act",
        "constitution.pdf": "Constitution of Nepal",
        "sports.pdf": "Sports Act",
    }

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            act_name = act_name_map.get(pdf_file.name, pdf_file.stem)
            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"
                doc.metadata["act_name"] = act_name
            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages")
        except Exception as e:
            print(f"Error: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents


def load_all_acts() -> List[Any]:
    """
    Load all specific law PDFs and attach act_name and source_file metadata.

    NOTE: Use this in an indexing/build script, not at import time.
    """
    docs: List[Any] = []

    pdf_specs = [
        {"filename": "pharmacy.pdf",      "act_name": "Nepal Pharmacy Council Act, 2057"},
        {"filename": "immunization.pdf",  "act_name": "Immunization Act, 2072"},
        {"filename": "single_women.pdf",  "act_name": "Single Women Act"},
        {"filename": "constitution.pdf",  "act_name": "Constitution of Nepal"},
        {"filename": "sports.pdf",        "act_name": "Sports Act"},
    ]

    for spec in pdf_specs:
        # IMPORTANT: project‑relative path (assuming you run from project root)
        pdf_path = f"data/pdf/{spec['filename']}"
        print(f"Loading {spec['filename']} ...")
        try:
            loader = PyPDFLoader(pdf_path)
            pdf_docs = loader.load()
            for d in pdf_docs:
                d.metadata["act_name"] = spec["act_name"]
                d.metadata["source_file"] = spec["filename"]
                d.metadata["file_type"] = "pdf"
            docs.extend(pdf_docs)
            print(f"  Loaded {len(pdf_docs)} pages")
        except Exception as e:
            print(f"  Error loading {spec['filename']}: {e}")

    print(f"Loaded {len(docs)} documents (pages) from PDFs")
    return docs


def split_documents(
    documents: List[Any],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Any]:
    """
    Split page-level documents into smaller chunks and optionally
    extract 'section_number' from text (धारा ...).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\nधारा",
            "\n\n",
            "\n",
            "। ",
            " ",
        ],
    )

    split_docs = text_splitter.split_documents(documents)

    for doc in split_docs:
        text = doc.page_content
        match = re.search(r"धारा\s*([०१२३४५६७८९0-9]+)", text)
        if match:
            doc.metadata["section_number"] = match.group(1)

    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    if split_docs:
        print("\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")

    return split_docs


# -------------------------------------------------------------------
# 2) Embedding manager
# -------------------------------------------------------------------

class EmbeddingManager:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ):
        """
        Embedding manager for multilingual (including Nepali) legal text.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.model: SentenceTransformer | None = None
        self.load_model()
    
    def load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name} on {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded successfully. Embedding dimension: {dim}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded")
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings


# -------------------------------------------------------------------
# 3) Vector store
# -------------------------------------------------------------------




class VectorStore:
    def __init__(
        self,
        collection_name: str = "pdf_documents_v2",  # new name to avoid mixing old embeddings
        persist_directory: str = "/Users/rijuphaiju/Documents/ytrag/data/vector_store",
        reset: bool = False,  # if True, delete existing collection on init
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.reset = reset
        self._initialize_store()
    
    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Optionally drop old collection (if you are rebuilding from scratch)
            if self.reset:
                try:
                    self.client.delete_collection(self.collection_name)
                    print(f"Deleted existing collection: {self.collection_name}")
                except Exception:
                    # If it doesn't exist yet, ignore
                    pass

            # Use cosine distance since we normalized embeddings
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF document embeddings for RAG (Nepali law)",
                    "hnsw:space": "cosine",  # important if you want cosine similarity
                },
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        print(f"Adding {len(documents)} documents to vector store...")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(getattr(doc, "metadata", {}))
            metadata["doc_index"] = i
            metadata["content_length"] = len(getattr(doc, "page_content", ""))
            metadatas.append(metadata)

            documents_text.append(getattr(doc, "page_content", ""))
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text,
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise



# -------------------------------------------------------------------
# 4) RAG Retriever
# -------------------------------------------------------------------

from typing import Any, Dict, List, Optional

class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[dict] = None,  # <-- NEW PARAM
    ) -> List[Dict[str, Any]]:
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top k: {top_k}, where: {where}")

        # 1) Generate embedding for query
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        retrieved_docs: List[Dict[str, Any]] = []

        try:
            # 2) Query the collection (pass where filter)
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where,  # <-- USE where HERE
            )
            # print("Raw results from vector store:", results)  # optional debug

            # 3) Process results if there are any
            if results and results.get("documents") and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    retrieved_docs.append({
                        "id": doc_id,
                        "content": document,
                        "metadata": metadata,
                        "distance": distance,  # smaller = more similar
                        "rank": i + 1,
                    })

                print(f"Retrieved {len(retrieved_docs)} documents")
            else:
                print("No documents found")

        except Exception as e:
            print(f"Error during retrieval: {e}")

        # 4) Always return a list
        return retrieved_docs

# -------------------------------------------------------------------
# 5) LLM and helpers
# -------------------------------------------------------------------

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    raise ValueError("GROQ_API_KEY not set in environment or .env file")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=1024,
)





def correct_nepali_spelling(text: str, llm) -> str:
    prompt = f"""तलको पाठ नेपाली भाषामा लेखिएको छ तर वर्तनी (spelling) 
र व्याकरणमा केही त्रुटि हुन सक्छ।

तपाईंको काम:
- अर्थ (meaning) नबदलिकन, केवल वर्तनी र व्याकरण सुधार्नुहोस्।
- केवल देवनागरी लिपि (क, ख, ग ...) प्रयोग गर्नुहोस् (कुनै रोमन/Latin अक्षर नलेख्नुहोस्)।
- वाक्यसंख्या वा जानकारी नथप्नुहोस् वा नघटाउनुहोस्, केवल सुधार गर्नुहोस्।
- केवल सुधार गरिएको पाठ मात्र आउटपुट गर्नुहोस्, अरू कुनै व्याख्या नगर्नुहोस्।

पाठ:
{text}

सुधार गरिएको पाठ:
"""
    resp = llm.invoke(prompt)
    return resp.content.strip()


def expand_query_with_llm(query: str, llm) -> str:
    prompt = f"""तपाईं प्रश्न पुनर्लेखन (query expansion) गर्ने सहयोगी हुनुहुन्छ।

कडा नियम:
- प्रयोगकर्ताको मूल प्रश्नको विषय, कानुन, र मुख्य शब्द (entities) नबदल्नुहोस्।
- नयाँ कानुनी विषय (जस्तै नागरिकता, नयाँ ऐन, अन्य विषय) नजोड्नुहोस्।
- प्रश्न यदि संविधानबारे छ भने, त्यसलाई नै संविधानकै सीमामा पुनर्लेखन गर्नुहोस्।
- प्रश्न यदि फार्मेसी बारे छ भने, त्यसलाई अन्य कानुन (जस्तै नागरिकता) मा नलगजानुहोस्।
- कुनै पनि हालतमा प्रश्नको विषय परिवर्तन गर्नु हुदैन (topic drift निषेध)।

तपाईंको काम:
- सोधिएको कानुनी प्रश्नलाई अलि स्पष्ट, प्रष्ट नेपालीमा पुनर्लेखन (परिवर्धन) गर्नुहोस्।
- केवल थोरै विस्तृत र स्पष्ट रूप दिनुहोस् (यसलाई नाटकीय रूपमा लामो वा फरक नबनाउनुहोस्)।
- प्रयोगकर्ताले प्रयोग गरेका मुख्य शब्दहरू (जस्तै "संविधान", "नागरिकता", "फार्मेसी", "खेलकुद") जस्ताको तस्तै राख्नुहोस्।
- यदि प्रयोगकर्ताले रोमन नेपालीमा सोधेको छ भने, त्यसलाई सही देवनागरी नेपालीमा रूपान्तरण गरेर प्रष्ट प्रश्नको रूपमा लेख्नुहोस्।
- केवल पुनर्लेखन गरिएको प्रश्न मात्र आउटपुट गर्नुहोस्, कुनै व्याख्या, टिप्पणी, वा कोष्ठकमा लेखिएको वाक्य नथप्नुहोस्।

उदाहरण (राम्रा व्यवहार):
- इनपुट: "kati ota sarkar hunxa constitution anusar?"
  आउटपुट: "नेपालको संविधान अनुसार नेपालमा कति तहका सरकार (संघीय, प्रदेश, स्थानीय) छन् ?"

- इनपुट: "pharmacy kholna k kei process xa?"
  आउटपुट: "नेपालमा फार्मेसी खोल्ने प्रक्रिया के–के छन् ?"

- इनपुट: "sports bikash kina chahinxa?"
  आउटपुट: "नेपालमा खेलकुद विकास किन आवश्यक छ ?"

इनपुट प्रश्न:
{query}

पुनर्लेखन/परिवर्धित प्रश्न (नेपालीमा, केवल वाक्य मात्र):
"""
    resp = llm.invoke(prompt)
    return resp.content.strip()


def contains_devanagari(text: str) -> bool:
    return bool(re.search(r'[\u0900-\u097F]', text))


def normalize_to_nepali(query: str, llm) -> str:
    if contains_devanagari(query):
        return query

    prompt = f"""You are a transliteration engine, not a chatbot.

TASK:
- Convert Romanized Nepali written in Latin script into correct Nepali in Devanagari.
- Do NOT translate, rephrase, or change the meaning.
- Do NOT guess a different question.
- Keep all words; if you don't know how to transliterate a word, copy it as-is.
- Preserve question structure (question marks etc.).
- Output ONLY the converted sentence, no explanation.

GOOD examples (do this):
- "pharmacy ain le ke vanxa?" -> "फार्मेसी ऐनले के भन्छ ?"
- "pharmacy council le yo ainma ke vanxa?" -> "फार्मेसी काउन्सिलले यो ऐनमा के भन्छ ?"
- "immunization act kaile aayeko ?" -> "इम्युनाइजेशन ऐन कहिले आएको ?"
- "yo ain namane kehi karbahi hunxa?" -> "यो ऐन नमाने केहि कारबाही हुन्छ ?"
- "nepalko sambidhan ke ho?" -> "नेपालको संविधान के हो ?"
- "ekal mahila ko ke ke adhikar chan?" -> "एकल महिलाको के के अधिकार छन् ?"
- "khel ko bikas ko lagi ke byabastha cha?" -> "खेलको विकासका लागि के व्यवस्था छ ?"

BAD examples (never do this):
- Changing "pharmacy ain le ke vanxa?" into 
  "फार्मेसी व्यवसाय सञ्चालन गर्न के–के शर्त चाहिन्छ?"  ✗
- Changing topic or inventing extra information.

User input:
{query}

Output (only the transliterated Nepali sentence):
"""
    resp = llm.invoke(prompt)
    return resp.content.strip()


# -------------------------------------------------------------------
# 6) Where filters
# -------------------------------------------------------------------

CATEGORY_TO_SOURCES = {
    "All (auto)": None,
    "Pharmacy Act": ["pharmacy.pdf"],
    "Immunization Act": ["immunization.pdf"],
    "Constitution of Nepal": ["constitution.pdf"],
    "Single Women Act": ["single_women.pdf"],
    "Sports Act": ["sports.pdf"],
}


def build_where_for_category(category: str) -> dict | None:
    sources = CATEGORY_TO_SOURCES.get(category)
    if not sources:
        return None
    if len(sources) == 1:
        return {"source_file": sources[0]}
    return {"$or": [{"source_file": s} for s in sources]}


def choose_where(query: str, norm_query: str) -> dict | None:
    """
    Decide which PDF to search based on keywords in the original and normalized query.
    """
    text = (query + " " + norm_query).lower()

    if any(word in text for word in ["pharmacy", "pharmasi", "फार्मेसी"]):
        return {"source_file": "pharmacy.pdf"}

    if any(word in text for word in ["immunization", "khop", "खोप", "इम्युनाइजेशन"]):
        return {"source_file": "immunization.pdf"}

    if any(word in text for word in ["constitution", "संविधान", "citizenship", "नागरिकता", "nagrita"]):
        return {"source_file": "constitution.pdf"}

    if any(word in text for word in ["single women", "single woman", "एकल महिला", "विधवा"]):
        return {"source_file": "single_women.pdf"}

    if any(word in text for word in ["sports", "खेलकुद", "खेल"]):
        return {"source_file": "sports.pdf"}

    return None
def rag_with_context(
    query: str,
    retriever,
    llm,
    top_k: int = 6,
    arena: str = "All (auto)",
):
    """
    Same as rag_simple, but returns (final_answer, context_text)
    so we can evaluate faithfulness/correctness against the legal text.
    """
    # Expand and normalize
    expanded_query = expand_query_with_llm(query, llm)
    norm_query = normalize_to_nepali(expanded_query, llm)

    # Decide which PDFs to search
    category_where = build_where_for_category(arena)
    auto_where = choose_where(query, norm_query)
    where = category_where if category_where is not None else auto_where

    if arena != "All (auto)":
        effective_top_k = max(top_k, 10)
    else:
        effective_top_k = top_k

    # Retrieve
    results = retriever.retrieve(norm_query, top_k=effective_top_k, where=where)
    if not results:
        results = retriever.retrieve(query, top_k=effective_top_k, where=where)

    if not results:
        return "सहित सन्दर्भ (context) फेला परेन, त्यसैले म जवाफ दिन सक्दिन।", ""

    # Build context
    max_chars = 2000
    context_parts = []
    current_len = 0
    for doc in results:
        text = doc["content"]
        if current_len + len(text) > max_chars:
            break
        context_parts.append(text)
        current_len += len(text)
    context = "\n\n".join(context_parts)

    # Same answering prompt as rag_simple
    prompt = f"""तपाईं नेपाली कानुन बुझ्ने कानुनी सहायक हुनुहुन्छ। तल दिइएको सन्दर्भ 
फार्मेसी, खोप, संविधान, एकल महिला, खेलकुद लगायतका नेपाली कानून तथा नीतिहरूबाट 
लिइएको हो। सन्दर्भको मूल पाठमा टाइप/OCR सम्बन्धी त्रुटि हुन सक्छ।

कडा नियम:
- सोधिएको प्रश्नको उत्तर केवल सन्दर्भमा भएको कानुनी व्यवस्थामा आधारित भएर मात्र दिनुहोस्।
- कानुनको भाषा (धारा/उपधारा) जति सकिन्छ त्यति नजिकबाट प्रस्तुत गर्नुहोस्; यदि आवश्यक परे मात्र
  छोटो व्याख्या/स्पष्टीकरण थप्नुहोस्।
- एउटै कुरा अनावश्यक रूपमा धेरैचोटि दोहोर्याउनु हुँदैन।
- यदि सन्दर्भमा स्पष्ट जवाफ छैन भने, प्रष्ट रूपमा लेख्नुस्:
  "मलाई थाहा छैन। यो जानकारी दिइएको सन्दर्भमा छैन।"
- कुनै पनि हालतमा सन्दर्भमा नदेखिएको नयाँ कानुनी दाबी वा धारा/व्यवस्था नबनाउनुहोस्। अनुमान नगरौँ।

सन्दर्भ (कानुनी पाठ):
{context}

प्रश्न (प्रयोगकर्ताको मूल इनपुट):
{query}

अन्तर्रूप (normalize) गरिएको प्रश्न:
{norm_query}

कृपया पहिलो भागमा कानुनको प्रासंगिक अंश (धारा/उपधारा) उद्धृत/सारांशित गर्नुहोस्,
र दोस्रो भागमा छोटो बुँदागत व्याख्या दिनुहोस्। कानूनबाहिरको अनुमान नगर्नुहोस्।

जवाफ नेपाली भाषामा:
"""
    resp = llm.invoke(prompt)
    raw_answer = resp.content.strip()

    # De-duplicate lines
    lines = [l.strip() for l in raw_answer.splitlines() if l.strip()]
    seen = set()
    dedup_lines = []
    for line in lines:
        norm_line = re.sub(r"\s+", " ", line)
        if norm_line not in seen:
            seen.add(norm_line)
            dedup_lines.append(line)
    dedup_answer = "\n".join(dedup_lines)

    final_answer = correct_nepali_spelling(dedup_answer, llm)

    return final_answer, context




# -------------------------------------------------------------------
# 7) Main RAG function
# -------------------------------------------------------------------


def rag_simple(
    query: str,
    retriever,
    llm,
    top_k: int = 6,
    arena: str = "All (auto)",  
) -> str:
    
    expanded_query = expand_query_with_llm(query, llm)
    print("Original query:", query)
    print("Expanded query:", expanded_query)

    
    norm_query = normalize_to_nepali(expanded_query, llm)
    print("Normalized query:", norm_query)

    
    category_where = build_where_for_category(arena)

  
    auto_where = choose_where(query, norm_query)

   
    if category_where is not None:
        where = category_where
    else:
        where = auto_where

    print("Using where filter:", where)

    
    if arena != "All (auto)":
        effective_top_k = max(top_k, 10)  # e.g., look at 10 chunks from that PDF
    else:
        effective_top_k = top_k

   
    results = retriever.retrieve(norm_query, top_k=effective_top_k, where=where)

    # 4b) Fallback: if no results, try original query
    if not results:
        print("No results with normalized query, trying original query...")
        results = retriever.retrieve(query, top_k=effective_top_k, where=where)

    if not results:
        return "सहित सन्दर्भ (context) फेला परेन, त्यसैले म जवाफ दिन सक्दिन।"



    # 4) Build context (truncate if too long)
    max_chars = 2000  # slightly smaller to reduce repetition
    context_parts = []
    current_len = 0
    for doc in results:
        text = doc["content"]
        if current_len + len(text) > max_chars:
            break
        context_parts.append(text)
        current_len += len(text)
    context = "\n\n".join(context_parts)

        # 5) Prompt for answer (simpler, direct answer)
    prompt = f"""तपाईं नेपाली कानुन बुझ्ने कानुनी सहायक हुनुहुन्छ। तल दिइएको सन्दर्भ 
फार्मेसी, खोप, संविधान, एकल महिला, खेलकुद लगायतका नेपाली कानून तथा नीतिहरूबाट 
लिइएको हो। सन्दर्भको मूल पाठमा टाइप/OCR सम्बन्धी त्रुटि हुन सक्छ।

नियम:
- सोधिएको प्रश्नको जवाफ आफूले बुझेको अर्थका आधारमा आफ्नै शब्दमा दिनुहोस्।
- सन्दर्भको ठ्याक्कै वाक्य वा अनुच्छेद जस्ताको तस्तै नक्कल गर्नु भन्दा,
  अर्थ/जानकारी समेटेर पुनर्लेखन (paraphrase) गर्नुहोस्।
- अधिकतम ५ बुँदामा मुख्य कुरा राख्नुहोस्, अनावश्यक रूपमा एउटै बुँदा दोहोर्याउनु हुँदैन।
- मानक, शुद्ध वर्तनी भएको देवनागरी नेपाली प्रयोग गर्नुहोस्।
- सकेसम्म धाराको/परिच्छेदको नम्बर वा शीर्षक (metadata मा section_number भएमा) उल्लेख गर्नुहोस्।
- यदि सन्दर्भमा सोधिएको विषयसँग स्पष्ट रूपमा सम्बन्धित कुनै जानकारी नै छैन भने मात्र
  यो वाक्य लेख्नुहोस्:
  "मलाई थाहा छैन। यो जानकारी दिइएको सन्दर्भमा छैन।"
- सामान्य ज्ञान मात्रबाट नयाँ कानुनी दाबी नबनाउनुहोस्।

सन्दर्भ:
{context}

प्रश्न (प्रयोगकर्ताको मूल इनपुट):
{query}

अन्तर्रूप (normalize) गरिएको प्रश्न:
{norm_query}

जवाफ नेपाली भाषामा:
"""
    response = llm.invoke(prompt)
    raw_answer = response.content.strip()

    # 6) Optionally de-duplicate nearly identical lines (avoid repetition)
    lines = [l.strip() for l in raw_answer.splitlines() if l.strip()]
    seen = set()
    dedup_lines = []
    for line in lines:
        norm_line = re.sub(r"\s+", " ", line)
        if norm_line not in seen:
            seen.add(norm_line)
            dedup_lines.append(line)
    dedup_answer = "\n".join(dedup_lines)

    # 7) Fix spelling/grammar in the answer
    final_answer = correct_nepali_spelling(dedup_answer, llm)

    return final_answer







# -------------------------------------------------------------------
# 8) Global instances to import elsewhere
# -------------------------------------------------------------------

embedding_manager = EmbeddingManager()
vectorstore = VectorStore(reset=False)  # assume index already built
rag_retriever = RAGRetriever(vectorstore, embedding_manager)