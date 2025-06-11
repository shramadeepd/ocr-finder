import os
import io
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import psycopg2
from psycopg2 import Error
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# --- Configuration ---
# You can set these via environment variables or modify directly
# Example DB_URL: "postgresql://user:password@host:port/database"
DB_URL = os.getenv("DB_URL", "postgresql://default:otvb4S5GlBgq@ep-dry-haze-a49225ub-pooler.us-east-1.aws.neon.tech/verceldb?sslmode=require")

# Directory where your pool of images are stored for indexing
IMAGE_POOL_DIR = os.getenv("IMAGE_POOL_DIR", "updates")

# Model configuration
MODEL_IMG_SIZE = (224, 224)  # ResNet50 input size
BATCH_SIZE = 32              # Number of images to process at once during indexing

# Similarity threshold: How "close" embeddings need to be to be considered similar.
# Adjust this value based on your definition of "similar"
SIMILARITY_THRESHOLD = 0.75

# --- Global Model Instance (Loaded once at startup) ---
_model = None

# --- Pydantic Models for API Responses ---
class SearchResult(BaseModel):
    image_name: str
    similarity_score: float

class SearchResponse(BaseModel):
    query_image_name: str
    most_similar_image: Optional[SearchResult] = None
    top_matches: List[SearchResult] = [] # List of top N results (even if below threshold)
    message: str

class IndexResponse(BaseModel):
    indexed_count: int
    message: str

# --- Database Operations ---
def _get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    conn = None
    try:
        conn = psycopg2.connect(DB_URL)
        return conn
    except Error as e:
        print(f"Database connection error: {e}")
        return None

def _create_embeddings_table(conn):
    """Creates the image_embeddings table if it doesn't exist."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_embeddings (
                id SERIAL PRIMARY KEY,
                image_path VARCHAR(500) UNIQUE NOT NULL,
                embedding FLOAT[] NOT NULL
            );
        """)
        conn.commit()
        print("Table 'image_embeddings' ensured to exist.")
    except Error as e:
        print(f"Error creating table: {e}")
        conn.rollback()
        raise # Re-raise to ensure startup fails if table cannot be created

def _insert_embedding(conn, image_path: str, embedding: np.ndarray):
    """Inserts or updates an image embedding in the database."""
    try:
        cursor = conn.cursor()
        embedding_list = embedding.tolist() # Convert NumPy array to Python list
        cursor.execute(
            """INSERT INTO image_embeddings (image_path, embedding) 
               VALUES (%s, %s) 
               ON CONFLICT (image_path) DO UPDATE SET embedding = EXCLUDED.embedding;""",
            (image_path, embedding_list)
        )
        conn.commit()
        return True
    except Error as e:
        print(f"Error inserting/updating embedding for {image_path}: {e}")
        conn.rollback()
        return False

def _get_all_embeddings_from_db(conn):
    """Fetches all image paths and embeddings from the database."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT image_path, embedding FROM image_embeddings;")
        db_results = cursor.fetchall()
        
        indexed_image_paths = [row[0] for row in db_results]
        indexed_features = np.array([np.array(row[1]) for row in db_results])
        
        return indexed_image_paths, indexed_features
    except Error as e:
        print(f"Error fetching all embeddings: {e}")
        return [], np.array([])

# --- Embedding Model and Feature Extraction ---
def _load_model():
    """Loads the pre-trained ResNet50 model once."""
    global _model
    if _model is None:
        print("Loading pre-trained ResNet50 model (this may take a moment)...")
        _model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        print("ResNet50 model loaded.")
    return _model

def _preprocess_image_bytes(image_bytes: bytes):
    """
    Loads an image from bytes, resizes it, and preprocesses it for the model.
    Ensures the image is converted to RGB (3 channels).
    Returns a NumPy array ready for model prediction.
    """
    img_stream = io.BytesIO(image_bytes)
    img = Image.open(img_stream).resize(MODEL_IMG_SIZE)
    
    # --- ADD THIS LINE TO ENSURE 3 CHANNELS (RGB) ---
    if img.mode != 'RGB':
        img = img.convert('RGB') 
    # --------------------------------------------------

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = preprocess_input(img_array) # Preprocess for ResNet50
    return img_array


def _extract_features_from_preprocessed(img_data_preprocessed: np.ndarray):
    """
    Extracts features from an already preprocessed image array.
    Assumes img_data_preprocessed is already batched (e.g., (1, 224, 224, 3)).
    """
    model = _load_model() # Ensure model is loaded
    features = model.predict(img_data_preprocessed, verbose=0)
    return features.flatten() # Flatten to 1D array

# --- FastAPI Application ---
app = FastAPI(
    title="Single-File Reverse Image Search API",
    description="API for finding similar images using deep learning embeddings and PostgreSQL in one file.",
    version="1.0.0"
)

# --- CORS Configuration (Adjust for production) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # WARNING: Change this to specific domains in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FastAPI Lifespan Events ---
@app.on_event("startup")
async def startup_event():
    """Actions to perform when the FastAPI application starts up."""
    print("Application startup initiated...")
    
    # 1. Load the deep learning model (happens once)
    _load_model()
    
    # 2. Connect to the database and ensure table exists
    conn = _get_db_connection()
    if conn:
        _create_embeddings_table(conn)
        conn.close()
    else:
        # If DB connection fails at startup, likely a configuration issue.
        # Raising an exception here will prevent the app from starting.
        raise RuntimeError("CRITICAL: Database connection failed on startup. Check DB_URL.")
    
    print("Application startup complete.")

# --- API Endpoints ---

@app.post("/index_images", response_model=IndexResponse, summary="Index images from a folder")
async def index_images_endpoint():
    """
    Indexes all images found in the configured `IMAGE_POOL_DIR`.
    Calculates embeddings for each image and stores them in the PostgreSQL database.
    This operation can take a while depending on the number of images and hardware.
    """
    conn = _get_db_connection()
    if not conn:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection failed during indexing."
        )

    all_img_files = [
        os.path.join(IMAGE_POOL_DIR, img_name)
        for img_name in os.listdir(IMAGE_POOL_DIR)
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))
    ]

    if not all_img_files:
        conn.close()
        return IndexResponse(indexed_count=0, message=f"No images found in '{IMAGE_POOL_DIR}'.")

    print(f"\n--- Starting Indexing of {len(all_img_files)} images from '{IMAGE_POOL_DIR}' ---")
    
    processed_count = 0
    total_files = len(all_img_files)

    for i in range(0, total_files, BATCH_SIZE):
        batch_img_paths = all_img_files[i:i + BATCH_SIZE]
        batch_preprocessed_tensors = []
        batch_original_paths = []

        # Load and preprocess images for the current batch
        for img_path in batch_img_paths:
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                preprocessed_tensor = _preprocess_image_bytes(img_bytes)
                batch_preprocessed_tensors.append(preprocessed_tensor)
                batch_original_paths.append(img_path)
            except (UnidentifiedImageError, Exception) as e:
                print(f"Skipping {os.path.basename(img_path)} due to error during preprocessing: {e}")

        if not batch_preprocessed_tensors:
            continue # Skip to next batch if all images in this batch failed

        # Stack the preprocessed tensors into a single NumPy array for batch prediction
        batched_input = np.vstack(batch_preprocessed_tensors)

        # Get features for the entire batch
        features_batch = _load_model().predict(batched_input, verbose=0)

        # Insert each image's features into the database
        for j, features in enumerate(features_batch):
            img_path = batch_original_paths[j]
            if _insert_embedding(conn, img_path, features.flatten()):
                processed_count += 1
                # print(f"  Indexed {os.path.basename(img_path)} ({processed_count}/{total_files})")
            # else: print(f"  Skipped (already indexed): {os.path.basename(img_path)}") # Can be noisy

    conn.close()
    return IndexResponse(
        indexed_count=processed_count,
        message=f"Indexing complete. {processed_count} images indexed/updated from '{IMAGE_POOL_DIR}'."
    )


@app.post("/search_image", response_model=SearchResponse, summary="Search for similar images")
async def search_image_endpoint(
    file: Optional[UploadFile] = File(None),
    img_base64: Optional[str] = None,  # For JSON payload containing base64 image
    top_n: int = 5
):
    """
    Takes an input image either as an UploadFile (multipart/form-data) or as a base64 string in JSON,
    extracts its embedding, and finds the most similar images in the database based on a similarity threshold.
    """
    conn = _get_db_connection()
    if not conn:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection failed during search."
        )

    try:
        # Determine input method: file upload or base64 JSON payload
        if file is not None:
            query_image_bytes = await file.read()
            query_image_name = file.filename
        elif img_base64 is not None:
            try:
                query_image_bytes = base64.b64decode(img_base64)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid base64 image encoding."
                )
            query_image_name = "base64_image.png"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No image provided. Please upload a file or provide a base64-encoded image."
            )
        
        # Preprocess query image
        query_img_preprocessed = _preprocess_image_bytes(query_image_bytes)
        
        # Extract features for query image
        query_features = _extract_features_from_preprocessed(query_img_preprocessed)
        query_features = query_features.reshape(1, -1)  # Reshape for cosine_similarity

        # Fetch all indexed embeddings from the database
        indexed_image_paths, indexed_features = _get_all_embeddings_from_db(conn)
        
        if not indexed_features.size:
            conn.close()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No image embeddings found in the database. Please index images first."
            )

        # Calculate cosine similarity
        similarities = cosine_similarity(query_features, indexed_features)[0]

        # Sort and get top matches
        sorted_indices = np.argsort(similarities)[::-1]

        top_matches_list = []
        most_similar_result: Optional[SearchResult] = None

        for i, idx in enumerate(sorted_indices[:top_n]):
            score = float(similarities[idx])
            img_path = indexed_image_paths[idx]
            img_name = os.path.basename(img_path)
            
            current_match = SearchResult(image_name=img_name, similarity_score=score)
            top_matches_list.append(current_match)

            if i == 0 and score >= SIMILARITY_THRESHOLD:
                most_similar_result = current_match
        
        if most_similar_result:
            message = f"Found a similar image above threshold {SIMILARITY_THRESHOLD:.2f}."
        else:
            message = f"No image found above similarity threshold {SIMILARITY_THRESHOLD:.2f}. Closest match had score {top_matches_list[0].similarity_score:.4f}."

        return SearchResponse(
            query_image_name=query_image_name,
            most_similar_image=most_similar_result,
            top_matches=top_matches_list,
            message=message
        )

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file format. Please upload a valid image (e.g., JPG, PNG)."
        )
    except Exception as e:
        print(f"Error during search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during search: {e}"
        )
    finally:
        if conn:
            conn.close()

# --- Main Execution Block ---
if __name__ == "__main__":
    import uvicorn
    import base64

    # --- Dummy Data Generation (for testing if you don't have images ready) ---
    # This block will run IF you execute this file directly (python main.py)
    # It will create 'my_image_pool' with 10 dummy images and a 'test_query_image.jpg'
    if not os.path.exists(IMAGE_POOL_DIR) or not os.listdir(IMAGE_POOL_DIR):
        print(f"'{IMAGE_POOL_DIR}' is empty or does not exist. Generating dummy images for testing...")
        os.makedirs(IMAGE_POOL_DIR, exist_ok=True)
        
        for i in range(10): # Create 10 dummy images
            dummy_img = Image.new('RGB', (224, 224), color=(i*20, i*10, i*5))
            dummy_img.save(os.path.join(IMAGE_POOL_DIR, f'pool_image_{i}.jpg'))
        
        # Create a test query image (very similar to pool_image_0.jpg)
        Image.new('RGB', (224, 224), color=(5, 5, 5)).save('test_query_image.jpg')
        
        print("Dummy images and test query image created.")
        print(f"You can now run 'uvicorn main:app --reload' and navigate to http://127.0.0.1:8000/docs")
        print(f"  1. Call /index_images/ to populate the DB.")
        print(f"  2. Call /search_image/ and upload 'test_query_image.jpg'.")
        print("\nNote: Dummy data is for testing. For real use, place your images in '{IMAGE_POOL_DIR}'.")
    
    # Run the FastAPI app using uvicorn
    # --reload is good for development as it restarts the server on code changes
    uvicorn.run(app, host="0.0.0.0", port=8001)