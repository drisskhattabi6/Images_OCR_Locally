import io, os, shutil
import logging
import subprocess
import streamlit as st
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from dotenv import load_dotenv
from PIL import Image
from ocr import openrouter_perform_ocr, ollama_perform_ocr

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set up logging
logger = logging.getLogger(__name__)
logger.info(f"Streamlit app is running")

def get_available_models():
    """Fetches the installed Ollama models, excluding 'NAME' and models containing 'embed'."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        models = [
            line.split(" ")[0] for line in result.stdout.strip().split("\n")
            if line and "NAME" not in line and "embed" not in line.lower()
        ]
        return models
    except subprocess.CalledProcessError as e:
        print(f"Error fetching models: {e}")
        return []

# Fetch available models
available_models = get_available_models()
if not available_models:
    st.error("No installed Ollama models found. Please install one using `ollama pull <model_name>`.")

st.set_page_config(page_title="Local OCR", page_icon="🤖")
st.markdown("#### 🗨️ Local OCR")

with st.sidebar:
    st.title("Settings :")

    # Upload the image
    image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if image is not None:
        temp_dir = "tmp"
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        # Clear the temp directory before saving the new image
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # remove file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # remove folder if somehow exists
            except Exception as e:
                st.error(f"Failed to delete {file_path}. Reason: {e}")

        # Save the uploaded image to the temp directory
        image_path = os.path.join(temp_dir, image.name)
        with open(image_path, "wb") as f:
            f.write(image.getbuffer())

        st.success(f"Image saved to {image_path}")

    # User selects the model Provider
    llm_provider = st.selectbox("Select LLM Provider:", ['Openrouter', 'Ollama'], index=0)

    if llm_provider == 'Ollama' :
        # User selects the model to use
        selected_model = st.selectbox("Select an Ollama model:", available_models, index=0)

    else : 
        selected_model = st.text_input("Enter LLM Name:", value='qwen/qwen2.5-vl-32b-instruct:free')
        # openrouter_api_key = st.text_input("Enter Openrouter API Key", type="password", value=os.getenv("OPENROUTER_API_KEY"))

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown("----")

if image_path:
    st.session_state.messages.append({"role": "user", "content": image_path})
    with st.chat_message("user"):
        st.markdown(image_path)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                llm_response = ""
                if llm_provider == 'Ollama' :
                    llm_response = ollama_perform_ocr(image_path=image_path)
                else : 
                    llm_response = openrouter_perform_ocr(image_path=image_path, llm_name=selected_model)
                    
                response = f"""
                {llm_response}
                """
                st.markdown(response)

                # Store assistant response
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
            st.rerun()
