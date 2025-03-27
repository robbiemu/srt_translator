import gradio as gr
from pathlib import Path
from srt_translator import SRTTranslator
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama  
import os
import tempfile
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


def create_translator(
    target_language: str = "es", 
    style_guideline: str = "Natural speech patterns", 
    technical_guideline: str = "Use standard terminology"
):    
    MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")
    MODEL_NAME = os.getenv("MODEL_NAME")
    MODEL_PARAM_TEMPERATURE = os.getenv("OPENAI_TEMPERATURE", "0.3")
    MODEL_PROVIDER_URL = os.getenv("MODEL_PROVIDER_URL", "http://localhost:11434")
    
    if MODEL_PROVIDER == "openai":
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        llm = OpenAI(
            model=os.getenv("OPENAI_MODEL", MODEL_NAME), # "gpt-4"
            temperature=float(MODEL_PARAM_TEMPERATURE),
            api_key=OPENAI_API_KEY
        )
    elif MODEL_PROVIDER == "ollama":
        OLLAMA_PARAM_TOP_P = os.getenv("OLLAMA_PARAM_TOP_P", "0.9")
        OLLAMA_PARAM_NUM_CTX = os.getenv("OLLAMA_PARAM_NUM_CTX", "4096")
        OLLAMA_PARAM_NUM_PREDICT = os.getenv("OLLAMA_PARAM_NUM_PREDICT", "1024")

        llm = Ollama( 
            model=os.getenv("OLLAMA_MODEL", MODEL_NAME),
            base_url=os.getenv("OLLAMA_BASE_URL", MODEL_PROVIDER_URL), 
            temperature=float(MODEL_PARAM_TEMPERATURE),
            top_p=float(OLLAMA_PARAM_TOP_P),
            num_ctx=int(OLLAMA_PARAM_NUM_CTX),
            num_predict=int(OLLAMA_PARAM_NUM_PREDICT)
        )
    else:
        raise ValueError("Invalid model provider")
    
    return SRTTranslator(
        llm=llm,
        target_language=target_language,
        tone_guidelines={
            'style': style_guideline,
            'technical': technical_guideline
        }
    )


def translate_srt_file(srt_file, target_language, style_guideline, technical_guideline):
    try:
        # Save uploaded file to temp location
        temp_dir = tempfile.mkdtemp()
        input_path = Path(temp_dir) / "input.srt"
        output_path = Path(temp_dir) / "output.srt"
        
        with open(input_path, "wb") as f:
            f.write(srt_file)
        
        # Create translator with dynamic settings
        translator = create_translator(
            target_language=target_language, 
            style_guideline=style_guideline, 
            technical_guideline=technical_guideline
        )
        
        # Perform translation
        translator.translate_srt(input_path, output_path)
        
        # Return translated file
        if output_path.exists():
            return str(output_path)  # Convert Path to string before returning
        return None
    except Exception as e:
        raise gr.Error(f"Translation failed: {str(e)}")
    

# Gradio interface
with gr.Blocks(title="SRT File Translator") as demo:
    gr.Markdown("""
    # SRT File Translator
    Upload an SRT subtitle file and customize your translation.
    """)
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload SRT File", type="binary")
            
            # Language Dropdown
            language_dropdown = gr.Dropdown(
                label="Target Language",
                choices=["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],
                value="es"
            )
            
            # Style Guidelines
            style_guideline = gr.Textbox(
                label="Style Guideline", 
                value="Natural speech patterns",
                placeholder="E.g., Conversational, Formal, Casual"
            )
            
            # Technical Guidelines
            technical_guideline = gr.Textbox(
                label="Technical Guideline", 
                value="Use standard terminology",
                placeholder="E.g., Use industry-specific terms, Localize technical language"
            )
            
            submit_btn = gr.Button("Translate", variant="primary")
        
        with gr.Column():
            file_output = gr.File(label="Translated SRT File")
    
    submit_btn.click(
        fn=translate_srt_file,
        inputs=[file_input, language_dropdown, style_guideline, technical_guideline],
        outputs=file_output
    )
    
    gr.Examples(
        examples=[
            ["example.srt", "es", "Natural speech patterns", "Use standard terminology"],
            ["example.srt", "fr", "Formal and elegant", "Precise technical translations"]
        ],
        inputs=[file_input, language_dropdown, style_guideline, technical_guideline],
        outputs=file_output,
        fn=translate_srt_file,
        cache_examples=True
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
