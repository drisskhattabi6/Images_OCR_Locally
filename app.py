import io, os
import logging
import subprocess
import streamlit as st
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set up logging
logger = logging.getLogger(__name__)
logger.info(f"Streamlit app is running")

# Function to generate PDF
def generate_pdf():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 40  # Start position for text
    max_width = width - 80  # Margin for text wrapping

    c.setFont("Helvetica-Bold", 16)
    header_text = "Conversation History"
    text_width = c.stringWidth(header_text, "Helvetica-Bold", 16)
    c.drawString((width - text_width) / 2, height - 40, header_text)

    y -= 30  # Adjust position after header
    c.setFont("Helvetica", 12)

    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "LLM"
        text = f"{role}: {msg['content']}"

        # Separate user questions with a line
        if msg["role"] == "user":
            y -= 10
            c.setStrokeColorRGB(0, 0, 0)
            c.line(40, y, width - 40, y)
            y -= 20  

        # Wrap text within max_width
        wrapped_lines = simpleSplit(text, c._fontname, c._fontsize, max_width)

        for line in wrapped_lines:
            c.drawString(40, y, line)
            y -= 20
            
            # Handle page breaks
            if y < 40:
                c.showPage()
                c.setFont("Helvetica", 12)
                y = height - 40  # Reset position after new page

    c.save()
    buffer.seek(0)
    return buffer

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

    images = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if images is not None:
        # Save the uploaded image to a temporary file
        image_path = os.path.join("temp", images.name)

    # User selects the model Provider
    llm_provider = st.selectbox("Select LLM Provider:", ['Openrouter', 'Ollama'], index=0)

    if llm_provider == 'Ollama' :
        # User selects the model to use
        selected_model = st.selectbox("Select an Ollama model:", available_models, index=0)

    else : 
        selected_model = st.text_input("Enter LLM Name:", value='qwen/qwq-32b:free')
        # openrouter_api_key = st.text_input("Enter Openrouter API Key", type="password", value=os.getenv("OPENROUTER_API_KEY"))

    # Button to download PDF
    if st.button("Download Chat as PDF"):
        if not st.session_state.messages:
            st.warning("No messages to download.")
        else :
            pdf_buffer = generate_pdf()
            st.download_button(
                label="Download", data=pdf_buffer,
                file_name="chat_history.pdf",
                mime="application/pdf"
            )

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

if query := st.chat_input("Ask Me..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                llm_response = ""
                if llm_provider == 'Ollama' :

                    llm_response = ollama_chat()

                    # st.markdown(f"""
                    #     \n---- 
                    #     Response Time: {response_time}, LLM Name: {selected_model}, Number of Retrieved Documents: {n_retrieved_docs}, Query Total Tokens: {query_token_count}, Prompt Token Count: {prompt_token_count}, Output Token Count: {output_token_count} | Document Type: {documnents_type}
                    #     """)

                    response = f"""
                        {remove_tags(llm_response)}

                        \n----
                        Response Time: {response_time}, LLM Name: {selected_model}, Number of Retrieved Documents: {n_retrieved_docs}, Query Token Count: {query_token_count}, Prompt Token Count: {prompt_token_count}, Output Token Count: {output_token_count} | Document Type: {documnents_type}
                        """

                else : 
                    llm_response, total_tokens = openrouter_chat(query, selected_model)
                    
                    response = f"""
                        {remove_tags(llm_response)}

                        \n----
                        LLM Name: {selected_model} | Total Tokens : {total_tokens} | Number of Retrieved Documents: {n_results} | Document Type: {documnents_type}
                        """
                    st.markdown(response)

                # Store assistant response
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
            st.rerun()
