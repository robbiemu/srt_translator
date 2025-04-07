import gradio as gr
from pathlib import Path
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
import os
import time
from dotenv import load_dotenv
import logging

from srt_translator import SRTTranslator 


load_dotenv()

# Set up logging
logging.basicConfig()
logger = logging.getLogger("srt_translator")


def create_translator(
    target_language: str = "es",
    style_guideline: str = "Natural speech patterns",
    technical_guideline: str = "Use standard terminology",
):
    # --- LLM Setup ---
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
            api_key=OPENAI_API_KEY,
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
            num_predict=int(OLLAMA_PARAM_NUM_PREDICT),
        )
    else:
        raise ValueError("Invalid model provider")

    # --- SRTTranslator Initialization ---
    return SRTTranslator(
        llm=llm,
        target_language=target_language,
        tone_guidelines={
            "style": style_guideline,
            "technical": technical_guideline,
        },
    )


def format_warning(warning):
    """Formats a single warning/error with proper structure"""
    if warning["type"] == "error":
        return (
            f"\n❌ ENTRY {warning['entry']} ERROR: {warning['message']}\n"
            f"ORIGINAL TEXT:\n{warning['original']}\n"
            f"TRANSLATION: [FAILED]\n"
            "────────────────────"
        )
    else: # Assuming 'validation' type
        return (
            f"\n⚠️ ENTRY {warning['entry']} WARNING: {warning['message']}\n"
            f"ORIGINAL TEXT:\n{warning['original']}\n"
            f"TRANSLATED TEXT:\n{warning.get('translation', '')}\n"
            "────────────────────"
        )


def get_persistent_temp_dir():
    """Returns a persistent temp directory path that won't be automatically cleaned up"""
    temp_base = os.getenv("GRADIO_TEMP_DIR", "/tmp/srt_translator")
    os.makedirs(temp_base, exist_ok=True)
    return temp_base


def translate_srt_file(srt_file, target_language, style_guideline, technical_guideline, progress=gr.Progress()):
    temp_dir = None
    try:
        temp_dir = os.path.join(get_persistent_temp_dir(), f"tmp{os.urandom(4).hex()}")
        os.makedirs(temp_dir, exist_ok=True)

        input_path = Path(temp_dir) / "input.srt"
        output_path = Path(temp_dir) / "output.srt"

        with open(input_path, "wb") as f:
            f.write(srt_file)

        translator = create_translator(
            target_language=target_language,
            style_guideline=style_guideline,
            technical_guideline=technical_guideline
        )
        entries = translator.parse_srt(input_path)
        total_entries = len(entries)

        warning_message = "Starting translation...\n"
        progress_desc = "Initializing..."
        yield None, warning_message

        for i, entry in enumerate(entries, 1):
            progress_desc = f"Translating: {i}/{total_entries} entries ({int((i/total_entries)*100)}%)"
            progress(i / total_entries)
            current_warning_update = None

            try:
                context = (
                    2,  # Context window size
                    [
                        entries[max(0, i-1)]['text'],
                        entries[min(total_entries-1, i+1)]['text']
                    ]
                )

                entry['translation'] = translator.translation_tool(entry['text'], context=context)
                is_valid, feedback = translator.validation_tool(entry['text'], entry['translation'], context=context)

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
                    current_warning_update = warning_message

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
                entry['translation'] = f"[ERROR] {entry['text']}"
                current_warning_update = warning_message

            yield None, current_warning_update if current_warning_update else gr.update()

        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(f"{entry['number']}\n")
                f.write(f"{entry['timecode']}\n")
                f.write(f"{entry.get('translation', entry['text'])}\n\n")

        # Final output with caching consideration
        warning_message += "\n\n✅ Translation complete!"
        progress_desc = "Translation complete!"
        
        yield str(output_path), warning_message
        if os.getenv("GRADIO_CACHING_EXAMPLES"): # For caching, keep files around and let Gradio handle them
            time.sleep(2)  # Give Gradio time to cache

    except Exception as e:
        raise gr.Error(f"Translation failed: {str(e)}")
    finally:
        if not os.getenv("GRADIO_CACHING_EXAMPLES") and temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup {temp_dir}: {cleanup_error}")


# Gradio interface
with gr.Blocks(title="SRT File Translator") as demo:
    gr.Markdown("""
    # SRT File Translator
    Upload an SRT subtitle file and customize your translation.
    """)

    with gr.Row():
        # --- Input Column ---
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

        # --- Output Column ---
        with gr.Column():
            file_output = gr.File(label="Translated SRT File")
            warnings_output = gr.Textbox(label="Translation Warnings", interactive=False)

    # *** Click handler outputs ***
    submit_btn.click(
        fn=translate_srt_file,
        inputs=[file_input, language_dropdown, style_guideline, technical_guideline],
        outputs=[file_output, warnings_output],  
        show_progress_on=[file_output]
    )

    # *** Examples outputs ***
    gr.Examples(
        examples=[
            ["example.srt", "es", "Natural speech patterns", "Use standard terminology"],
            ["example.srt", "fr", "Formal and elegant", "Precise technical translations"],
        ],
        inputs=[file_input, language_dropdown, style_guideline, technical_guideline],
        outputs=[file_output, warnings_output],
        fn=translate_srt_file,
        cache_examples=True,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)