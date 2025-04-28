import os, shutil
import logging
import mimetypes
import base64
import subprocess
import streamlit as st
from dotenv import load_dotenv
from ocr import openrouter_perform_ocr, ollama_perform_ocr

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set up logging
logger = logging.getLogger(__name__)
logger.info(f"Streamlit app is running")

def display_message(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)

    if isinstance(content, str) and content.startswith("tmp"): 
        mime_type, _ = mimetypes.guess_type(content)

        if mime_type and mime_type.startswith("image/"):
            st.image(content, width=250)

        elif mime_type == "application/pdf":
            with open(content, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="250" height="400" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

        else:
            st.markdown(f"Unsupported file type: {mime_type}")

    else:
        st.markdown(content)

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

def clear_temp_dir():
    """Clears the temporary directory."""
    temp_dir = "tmp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

file_path = None

with st.sidebar:
    st.title("Settings :")

    # Upload the file
    file = st.file_uploader("Upload File (image/pdf)", type=["jpg", "jpeg", "png", "pdf"], label_visibility="collapsed")

    if file is not None:
        # Save the uploaded file
        file_path = os.path.join("tmp", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        process = st.button("Process File")

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
        clear_temp_dir()
        st.session_state.file = None
        file_path = None
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]

        display_message(content)

st.markdown("----")

if file_path and process:
    st.session_state.messages.append({"role": "user", "content": file_path})
    with st.chat_message("user"):
        display_message(file_path)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = ""
                if llm_provider == 'Ollama' :
                    response = ollama_perform_ocr(image_path=file_path)
                else : 
                    response = openrouter_perform_ocr(file_path=file_path, llm_name=selected_model)

                st.markdown(response)

                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
            st.rerun()
