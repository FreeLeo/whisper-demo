"""
At the command line, only need to run once to install the package via pip:

$ pip install google-generativeai

https://makersuite.google.com/app/apikey
"""

from pathlib import Path
import google.generativeai as genai

genai.configure(
    api_key="********************", transport="rest")

# Set up the model
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

model = genai.GenerativeModel(model_name="gemini-pro-vision",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# Validate that an image is present
if not (img := Path("file/dogandcat.webp")).exists():
    raise FileNotFoundError(f"Could not find image: {img}")

image_parts = [
    {
        "mime_type": "image/webp",
        "data": Path("file/dogandcat.webp").read_bytes()
    },
]

prompt_parts = [
    image_parts[0],
    "\n图片上有什么？",
]

response = model.generate_content(prompt_parts)
print(response.text)
