import gradio as gr
from pathlib import Path
from srt_translator import SRTTranslator
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama  
import os
import tempfile
from dotenv import load_dotenv
import logging


load_dotenv()

# Set up logging to capture warnings
logging.basicConfig()
logger = logging.getLogger('srt_translator')


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
            model=os.getenv("OPENAI_MODEL", MODEL_NAME),
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


def format_warning(warning):
    """Formats a single warning/error with proper structure"""
    if warning['type'] == 'error':
        return (
            f"\n❌ ENTRY {warning['entry']} ERROR: {warning['message']}\n"
            f"ORIGINAL TEXT:\n{warning['original']}\n"
            f"TRANSLATION: [FAILED]\n"
            "────────────────────"
        )
    else:
        return (
            f"\n⚠️ ENTRY {warning['entry']} WARNING: {warning['message']}\n"
            f"ORIGINAL TEXT:\n{warning['original']}\n"
            f"TRANSLATED TEXT:\n{warning.get('translation', '')}\n"
            "────────────────────"
        )


def translate_srt_file(srt_file, target_language, style_guideline, technical_guideline):
    try:
        # Save uploaded file
        temp_dir = tempfile.mkdtemp()
        input_path = Path(temp_dir) / "input.srt"
        output_path = Path(temp_dir) / "output.srt"
        
        with open(input_path, "wb") as f:
            f.write(srt_file)

        # Setup translator and parse
        translator = create_translator(
            target_language=target_language,
            style_guideline=style_guideline,
            technical_guideline=technical_guideline
        )
        entries = translator.parse_srt(input_path)
        
        # Initialize streaming output
        warning_message = "Starting translation...\n"
        yield None, warning_message, 0  # Initialize progress to 0
        
        for i, entry in enumerate(entries):
            # Calculate progress
            progress = (i + 1) / len(entries)
            
            try:
                # Get context (simplified for example)
                context = (2, [
                    entries[max(0,i-1)]['text'],
                    entries[min(len(entries)-1,i+1)]['text']
                ])
                
                # Translate
                entry['translation'] = translator.translation_tool(
                    entry['text'],
                    context=context
                )
                
                # Validate
                is_valid, feedback = translator.validation_tool(
                    entry['text'],
                    entry['translation'],
                    context=context
                )
                
                if not is_valid:
                    warning = {
                        'entry': entry['number'],
                        'type': 'validation',
                        'message': feedback,
                        'original': entry['text'],
                        'translation': entry['translation']
                    }
                    formatted = format_warning(warning)
                    warning_message += formatted
                    yield None, warning_message, progress
                else:
                    # Only yield progress if no warning
                    yield None, warning_message, progress
                    
            except Exception as e:
                error = {
                    'entry': entry['number'],
                    'type': 'error',
                    'message': str(e),
                    'original': entry['text'],
                    'translation': None
                }
                formatted = format_warning(error)
                warning_message += formatted
                yield None, warning_message, progress  # Yield progress even on error
                continue

        # Final save
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(f"{entry['number']}\n")
                f.write(f"{entry['timecode']}\n")
                f.write(f"{entry.get('translation', entry['text'])}\n\n")
        
        # Final output
        warning_message += "\n\n✅ Translation complete!"
        yield str(output_path), warning_message, 1  # Final progress: 100%
        
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
            language_dropdown = gr.Dropdown(
                label="Target Language",
                choices=["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],
                value="es"
            )
            style_guideline = gr.Textbox(
                label="Style Guideline", 
                value="Natural speech patterns",
                placeholder="E.g., Conversational, Formal, Casual"
            )
            technical_guideline = gr.Textbox(
                label="Technical Guideline", 
                value="Use standard terminology",
                placeholder="E.g., Use industry-specific terms, Localize technical language"
            )
            submit_btn = gr.Button("Translate", variant="primary")
        
        with gr.Column():
            file_output = gr.File(label="Translated SRT File")
            warnings_output = gr.Textbox(label="Translation Warnings", interactive=False)
            progress_bar = gr.Slider(label="Translation Progress", minimum=0, maximum=1, step=0.01, interactive=False)  # Add progress bar

    
    submit_btn.click(
        fn=translate_srt_file,
        inputs=[file_input, language_dropdown, style_guideline, technical_guideline],
        outputs=[file_output, warnings_output, progress_bar]  # Add progress bar to outputs
    )
    
    gr.Examples(
        examples=[
            ["example.srt", "es", "Natural speech patterns", "Use standard terminology"],
            ["example.srt", "fr", "Formal and elegant", "Precise technical translations"]
        ],
        inputs=[file_input, language_dropdown, style_guideline, technical_guideline],
        outputs=[file_output, warnings_output, progress_bar],
        fn=translate_srt_file,
        cache_examples=True
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)