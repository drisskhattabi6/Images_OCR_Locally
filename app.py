import os, shutil
import logging
import mimetypes
import base64
import subprocess
import streamlit as st
from dotenv import load_dotenv
from ocr import perform_ocr, ollama_perform_ocr

load_dotenv()

API_KEY = os.getenv('API_KEY')

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set up logging
logger = logging.getLogger(__name__)
logger.info(f"Streamlit app is running")

def display_message(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)

    if isinstance(file_path, str) and file_path.startswith("tmp"): 
        mime_type, _ = mimetypes.guess_type(file_path)

        if mime_type and mime_type.startswith("image/"):
            st.image(file_path, width=250)

        elif mime_type == "application/pdf":
            with open(file_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="250" height="400" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

        else:
            st.markdown(f"Unsupported file type: {mime_type}")

    else:
        st.markdown(file_path)

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

def clear_temp_dir():
    """Clears the temporary directory."""
    temp_dir = "tmp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

file_path = None

with st.sidebar:
    st.title("🗨️ Local OCR")
    st.markdown("Settings :")

    # Upload the file
    file = st.file_uploader("Upload File (image/pdf)", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

    if file is not None:
        # Ensure the tmp directory exists
        os.makedirs("tmp", exist_ok=True)

        # Save the uploaded file
        file_path = os.path.join("tmp", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        process = st.button("Process File")

    # User selects the model Provider
    llm_provider = st.selectbox("Select LLM Provider:", ['Sambanova', 'Ollama'], index=0)

    if llm_provider == 'Ollama' :
        # User selects the model to use
        selected_model = st.selectbox("Select an Ollama model:", available_models, index=0)

    else : 
        selected_model = st.text_input("Enter LLM Name:", value='Llama-4-Maverick-17B-128E-Instruct')

    if st.button("Clear Chat"):
        st.session_state.messages = []
        clear_temp_dir()
        st.session_state.file = None
        file_path = None
        st.rerun()

    st.markdown("----")
    st.info(""" Some Notes:
    - Ollama models are installed locally. Sambanova models are hosted on the cloud.
    - **Note:** For Sambanova, you need to set up your API key in the `.env` file.
    - **Note:** For Ollama, you need to install a vision model like llama3.2-vision (`ollama pull llama3.2-vision`) command.""")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        display_message(message["content"])

if file_path and process:
    st.session_state.messages.append({"role": "user", "content": file_path})
    with st.chat_message("user"):
        display_message(file_path)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = ""
                # Determine file type
                mime_type, _ = mimetypes.guess_type(file_path)

                if llm_provider == 'Ollama' :
                    response = ollama_perform_ocr(image_path=file_path)
                else : 
                    response = perform_ocr(file_path=file_path, llm_name=selected_model)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
            st.rerun()
