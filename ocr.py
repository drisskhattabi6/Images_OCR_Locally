import base64, os
import ollama
import requests
import json
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

SYSTEM_PROMPT = """Act as an OCR assistant. Analyze the provided image and:
1. Recognize all visible text in the image as accurately as possible.
2. Maintain the original structure and formatting of the text.
3. If any words or phrases are unclear, indicate this with [unclear] in your transcription.
Provide only the transcription without any additional comments."""

def encode_to_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def ollama_perform_ocr(image_path):
    """Perform OCR on the given image using Llama 3.2-Vision."""
    base64_image = encode_to_base64(image_path)

    response = ollama.chat(
        model= "llama3.2-vision",
        messages= [{
            "role": "user",
            "content": SYSTEM_PROMPT,
            "images": [base64_image],
        }],
    )

    return response['message']['content'].strip()


# def generate_response2(query, scraped_data, llm_name='qwen/qwq-32b:free', retry=False) :
def openrouter_perform_ocr(file_path, llm_name='qwen/qwen2.5-vl-32b-instruct:free') :
    print("--> Generate Response from OpenRouter LLM : ", llm_name)
    base64_file = encode_to_base64(file_path)

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        data_url = f"data:application/pdf;base64,{base64_file}"
        file_type = {
            "type": "image_url",
            "image_url": {
                "url": data_url
            }
        }
    else:
        data_url = f"data:image/jpeg;base64,{base64_file}"
        file_type = {
            "type": "file",
            "file": {
                "filename": file_path,
                "file_data": data_url
            }
        }

    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": SYSTEM_PROMPT
        }, file_type]
    }]

    # print("messages : ", messages)

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data = json.dumps({
            "model": llm_name,
            "messages": messages
        })
    )
    response_data = json.loads(response.text)

    # print(response_data)
    if "error" in response_data:
        error_message = response_data["error"].get("message", "Unknown error")
        raise Exception(f"API Error: {error_message}")
    
    print( 'response_data : ', response_data["choices"][0]["message"]["content"])
    return response_data["choices"][0]["message"]["content"]
    
if __name__ == "__main__":
    image_path = "imgs/text_exp1.png"
    result = openrouter_perform_ocr(image_path)
    if result:
        print("OCR Recognition Result:")
        print(result)
