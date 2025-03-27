import logging
from typing import List, Dict, Tuple, Optional, Generator
from pathlib import Path
from datetime import datetime
import re
import os
from dotenv import load_dotenv


load_dotenv()

LOGGING = "true" == os.getenv("LOGGING", "false")
if LOGGING:
    handlers=[
        logging.StreamHandler()
    ]

    LOG_AGENT_FILE = os.getenv("LOG_AGENT_FILE", False)
    if LOG_AGENT_FILE:
        handlers.insert(0, logging.FileHandler(LOG_AGENT_FILE))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    logger = logging.getLogger(__name__)
else:
    class DummyLogger:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    logger = DummyLogger()


class SRTTranslator:
    MAX_CONTEXT_WINDOW = int(os.getenv("MAX_CONTEXT_WINDOW", "5")) # Maximum number of surrounding lines for context
    DEFAULT_CONTEXT_WINDOW = int(os.getenv("DEFAULT_CONTEXT_WINDOW", "2")) # Default context window size
    MAX_SUBTITLE_LENGTH = int(os.getenv("MAX_SUBTITLE_LENGTH", "80")) # Recommended max characters per subtitle line
    MAX_LINES_PER_SUBTITLE = int(os.getenv("MAX_LINES_PER_SUBTITLE", "2")) # Recommended max lines per subtitle


    def __init__(self, llm, target_language: str, tone_guidelines: Optional[Dict] = None):
        """
        Enhanced SRT file translator with dynamic context handling and validation.
        
        Args:
            llm: Language model instance
            target_language: Translation target language (ISO 639-1 code)
            tone_guidelines: Optional style guidelines for translation
            
        Raises:
            ValueError: If target_language is not a 2-letter ISO code
        """
        if len(target_language) != 2 or not target_language.isalpha():
            raise ValueError("target_language must be a 2-letter ISO 639-1 code")
            
        self.llm = llm
        self.target_language = target_language.lower()
        self.tone_guidelines = self._validate_tone_guidelines(tone_guidelines)
        
        logger.info(f"Initialized SRTTranslator for {target_language} with guidelines: {tone_guidelines}")
        
        # Create direct translation and validation tools
        self.translation_tool = self._create_translation_tool()
        self.validation_tool = self._create_validation_tool()

    def _validate_tone_guidelines(self, guidelines: Optional[Dict]) -> Dict:
        """Validate and set default tone guidelines."""
        default_guidelines = {
            'general': 'Natural speech patterns, concise for reading',
            'formal': 'Use polite forms when appropriate',
            'casual': 'Use colloquial language naturally',
            'technical': 'Use standard terminology',
            'cultural': 'Adapt references for target audience'
        }
        
        if not guidelines:
            return default_guidelines
            
        return {**default_guidelines, **guidelines}

    def _create_translation_tool(self):
        """Create a direct translation tool method."""
        def translate(
            content: str,
            context: Optional[Tuple[int, List[str]]] = None
        ) -> str:
            """
            Translate subtitle text with optional context.
            
            Args:
                content: Text to translate
                context: Tuple of (window_size, context_lines)
                
            Returns:
                Translated text with preserved formatting
            """
            try:
                window_size, context_lines = context or (0, [])
                
                # Validate input before processing
                if not content.strip():
                    logger.warning("Empty content passed to translation tool")
                    return ""
                
                # Preserve any formatting markers (like italics)
                formatting_tags = self._extract_formatting_tags(content)
                
                prompt = self._build_translation_prompt(content, window_size, context_lines)
                response = self.llm.complete(prompt).text.strip()
                
                # Restore formatting tags if they were removed during translation
                if formatting_tags:
                    response = self._restore_formatting_tags(response, formatting_tags)
                
                # Validate basic subtitle constraints
                self._validate_subtitle_length(response)
                
                return response
            except Exception as e:
                logger.error(f"Translation failed for content: {content[:50]}... Error: {str(e)}")
                raise

        return translate

    def _create_validation_tool(self):
        """Create a direct validation tool method."""
        def validate(
            original: str,
            translation: str,
            context: Optional[Tuple[int, List[str]]] = None
        ) -> Tuple[bool, str]:
            """
            Validate translation quality and subtitle constraints.
            
            Returns:
                Tuple of (is_valid, feedback_message)
            """
            try:
                window_size, context_lines = context or (0, [])
                
                # Basic validation checks
                if not translation.strip():
                    return False, "Empty translation"
                    
                if len(translation) > len(original) * 1.5:
                    return False, "Translation too long compared to original"
                
                # LLM-based validation
                validation_result = self._llm_validation(original, translation, window_size, context_lines)
                
                # Additional technical checks
                line_breaks = translation.count('\n')
                if line_breaks > self.MAX_LINES_PER_SUBTITLE:
                    validation_result = (
                        False,
                        f"{validation_result[1]} | Too many lines ({line_breaks})"
                    )
                
                return validation_result
            except Exception as e:
                logger.error(f"Validation failed for translation: {translation[:50]}... Error: {str(e)}")
                return False, f"Validation error: {str(e)}"

        return validate

    def _build_translation_prompt(self, content: str, window_size: int, context_lines: List[str]) -> str:
        """Construct the translation prompt with context and guidelines."""
        return (
            f"Translate this subtitle to {self.target_language} while following these rules:\n"
            f"1. Preserve all formatting (italics, etc.)\n"
            f"2. Keep timing constraints in mind (be concise)\n"
            f"3. Maintain consistency with context\n\n"
            f"Tone Guidelines:\n{self._format_guidelines()}\n\n"
            f"Current Text:\n{content}\n\n"
            f"Context Window: {window_size} lines\n"
            f"Context Content:\n{'- '.join(context_lines) if context_lines else 'None'}\n\n"
            f"Provide only the translation without explanations."
        )

    def _format_guidelines(self) -> str:
        """Format tone guidelines for prompt inclusion."""
        return '\n'.join(f"- {k}: {v}" for k, v in self.tone_guidelines.items())

    def _llm_validation(self, original: str, translation: str, window_size: int, context_lines: List[str]) -> Tuple[bool, str]:
        """Perform LLM-based quality validation of translation."""
        prompt = (
            f"Validate this {self.target_language} subtitle translation:\n\n"
            f"Original: {original}\n\n"
            f"Translation: {translation}\n\n"
            f"Context Window: {window_size} lines\n"
            f"Context Used:\n{'- '.join(context_lines) if context_lines else 'None'}\n\n"
            f"Check for:\n"
            f"1. Accuracy of translation\n"
            f"2. Natural flow in target language\n"
            f"3. Consistency with context\n"
            f"4. Adherence to tone guidelines\n\n"
            f"Respond with:\n"
            f"VALID: <feedback> - if translation meets all criteria\n"
            f"ISSUES: <feedback> - if any issues found\n"
            f"Provide specific, actionable feedback."
        )
        
        response = self.llm.complete(prompt).text.strip()
        
        if response.startswith('VALID:'):
            return True, response[6:].strip()
        return False, response[8:].strip() if response.startswith('ISSUES:') else response

    def parse_srt(self, file_path: Path) -> List[Dict]:
        """Parse SRT file into structured entries with validation."""
        entries = []
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:  # Handle BOM if present
                content = f.read().strip()
                
            for entry in content.split('\n\n'):
                if not entry.strip():
                    continue
                
                parts = [p.strip() for p in entry.split('\n') if p.strip()]
                if len(parts) < 3:
                    logger.warning(f"Skipping malformed entry: {parts[0] if parts else 'Unknown'}")
                    continue
                
                try:
                    entry_dict = {
                        'number': int(parts[0]),
                        'timecode': self._validate_timecode(parts[1]),
                        'text': '\n'.join(parts[2:]),
                        'translation': None,
                        'context_used': None,
                        'validated': None,
                        'formatting': self._extract_formatting_tags('\n'.join(parts[2:]))
                    }
                    entries.append(entry_dict)
                except (ValueError, IndexError) as e:
                    logger.error(f"Error parsing entry {parts[0] if parts else 'Unknown'}: {str(e)}")
                    continue
                    
            logger.info(f"Successfully parsed {len(entries)} entries from {file_path}")
            return entries
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {str(e)}")
            raise SRTParseError(f"SRT file parsing failed: {str(e)}") from e

    def _validate_timecode(self, timecode: str) -> str:
        """Validate SRT timecode format."""
        pattern = r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$'
        if not re.match(pattern, timecode):
            logger.warning(f"Invalid timecode format: {timecode}")
            raise ValueError(f"Invalid timecode format: {timecode}")
        return timecode

    def _extract_formatting_tags(self, text: str) -> Dict[str, List[str]]:
        """Extract and preserve formatting tags from text."""
        tags = {
            'italics': re.findall(r'<i>(.*?)</i>', text),
            'bold': re.findall(r'<b>(.*?)</b>', text),
            'underline': re.findall(r'<u>(.*?)</u>', text)
        }
        return {k: v for k, v in tags.items() if v}

    def _restore_formatting_tags(self, text: str, tags: Dict[str, List[str]]) -> str:
        """Restore formatting tags to translated text when possible."""
        # Simple implementation - could be enhanced with position tracking
        for tag_type, tag_contents in tags.items():
            for content in tag_contents:
                if content in text:
                    text = text.replace(content, f'<{tag_type[0]}>{content}</{tag_type[0]}>')
        return text

    def _validate_subtitle_length(self, text: str) -> None:
        """Check subtitle length against recommended constraints."""
        lines = text.split('\n')
        for line in lines:
            if len(line) > self.MAX_SUBTITLE_LENGTH:
                logger.warning(f"Subtitle line exceeds recommended length: {len(line)} characters")
        if len(lines) > self.MAX_LINES_PER_SUBTITLE:
            logger.warning(f"Subtitle exceeds recommended line count: {len(lines)} lines")

    def _get_context(self, entries: List[Dict], idx: int, window: int) -> List[str]:
        """Get context lines around given index with boundary checks."""
        if window <= 0:
            return []
            
        start = max(0, idx - window)
        end = min(len(entries), idx + window + 1)
        return [
            entries[i]['text'] 
            for i in range(start, end) 
            if i != idx and entries[i]['text'].strip()
        ]

    def _determine_context(self, current_text: str, prev_text: str, next_text: str) -> int:
        """Determine optimal context window size based on content analysis."""
        try:
            # Simple heuristic before LLM call
            if not any((prev_text, next_text)):
                return 0
                
            if '...' in current_text or any(w in current_text.lower() for w in ['this', 'that', 'those', 'he', 'she', 'they']):
                return self.DEFAULT_CONTEXT_WINDOW
                
            prompt = (
                f"Analyze this subtitle context needs (respond with number 0-{self.MAX_CONTEXT_WINDOW}):\n"
                f"Current: {current_text}\n"
                f"Previous: {prev_text or '[START]'}\n"
                f"Next: {next_text or '[END]'}\n"
                f"Consider:\n"
                f"1. Pronoun resolution needs\n"
                f"2. Continuation markers (...)\n"
                f"3. Incomplete sentences\n"
                f"4. Clear standalone meaning\n"
                f"Return only the number."
            )
            response = self.llm.complete(prompt).text.strip()
            return min(self.MAX_CONTEXT_WINDOW, max(0, int(response)))
        except Exception as e:
            logger.warning(f"Context determination failed, using default: {str(e)}")
            return self.DEFAULT_CONTEXT_WINDOW

    def translate_srt(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        save_every: int = 20,
        max_retries: int = 3
    ) -> str:
        """
        Translate SRT file with dynamic context handling and validation.
        
        Args:
            input_path: Path to input SRT file
            output_path: Optional output path
            save_every: Save progress every N entries
            max_retries: Maximum retries for failed translations
            
        Returns:
            Final translated SRT content
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            SRTParseError: If file parsing fails
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        logger.info(f"Starting translation of {input_path}")
        start_time = datetime.now()
        
        entries = self.parse_srt(input_path)
        if not entries:
            logger.warning("No valid entries found in SRT file")
            return ""
            
        for i, entry in enumerate(entries):
            try:
                for attempt in range(max_retries):
                    try:
                        # Get surrounding context
                        prev_text = entries[i-1]['text'] if i > 0 else None
                        next_text = entries[i+1]['text'] if i < len(entries)-1 else None
                        window = self._determine_context(entry['text'], prev_text, next_text)
                        context = (window, self._get_context(entries, i, window))
                        
                        # Use directly created translation and validation tools
                        entry['translation'] = self.translation_tool(
                            entry['text'],
                            context=context
                        )
                        
                        # Validate with same context
                        is_valid, feedback = self.validation_tool(
                            entry['text'],
                            entry['translation'],
                            context=context
                        )
                        
                        entry.update({
                            'context_used': window,
                            'validated': {
                                'is_valid': is_valid,
                                'feedback': feedback,
                                'attempts': attempt + 1
                            }
                        })
                        
                        if is_valid:
                            break
                            
                        logger.warning(f"Validation issues in entry {entry['number']}: {feedback}")
                        
                    except Exception as e:
                        logger.error(f"Attempt {attempt + 1} failed for entry {entry['number']}: {str(e)}")
                        if attempt == max_retries - 1:
                            raise
                        continue
                
                # Periodic saving and progress logging
                if i and i % save_every == 0:
                    self._save_partial(entries[:i+1], output_path)
                    logger.info(f"Progress: {i+1}/{len(entries)} entries processed")
                    
            except Exception as e:
                logger.error(f"Fatal error processing entry {entry['number']}: {str(e)}")
                raise SRTTranslationError(f"Translation failed at entry {entry['number']}") from e
        
        # Final save and logging
        result = self._save_results(entries, output_path)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Completed translation of {len(entries)} entries in {duration:.2f} seconds")
        
        return result

    def _save_partial(self, entries: List[Dict], base_path: Path) -> None:
        """Save partial results with timestamp."""
        if not base_path:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = base_path.with_stem(f"{base_path.stem}_partial_{timestamp}")
        try:
            path.write_text(self._format_srt(entries), encoding='utf-8')
            logger.debug(f"Saved partial results to {path}")
        except Exception as e:
            logger.error(f"Failed to save partial results: {str(e)}")

    def _save_results(self, entries: List[Dict], path: Optional[Path]) -> str:
        """Save final results and return SRT content."""
        srt_content = self._format_srt(entries)
        if path:
            try:
                path.write_text(srt_content, encoding='utf-8')
                logger.info(f"Saved final results to {path}")
            except Exception as e:
                logger.error(f"Failed to save final results: {str(e)}")
                raise
        return srt_content

    def _format_srt(self, entries: List[Dict]) -> str:
        """Format entries into valid SRT file content."""
        srt_entries = []
        for e in entries:
            try:
                srt_entry = [
                    str(e['number']),
                    e['timecode'],
                    e.get('translation', '')
                ]
                srt_entries.append('\n'.join(srt_entry))
            except KeyError as ke:
                logger.error(f"Missing key {str(ke)} in entry {e.get('number', 'unknown')}")
                continue
                
        return '\n\n'.join(srt_entries)

    def stream_translate(self, input_path: Path) -> Generator[Dict, None, None]:
        """
        Stream translations one by one for large files.
        
        Yields:
            Dictionary with original and translated entry
        """
        try:
            entries = self.parse_srt(input_path)
            for entry in entries:
                prev_text = None
                next_text = None
                if entries.index(entry) > 0:
                    prev_text = entries[entries.index(entry)-1]['text']
                if entries.index(entry) < len(entries)-1:
                    next_text = entries[entries.index(entry)+1]['text']
                
                window = self._determine_context(entry['text'], prev_text, next_text)
                context = (window, self._get_context(entries, entries.index(entry), window))
                
                entry['translation'] = self.translation_tool(
                    entry['text'],
                    context=context
                )
                
                yield {
                    'original': entry['text'],
                    'translation': entry['translation'],
                    'context_window': window,
                    'position': entry['number']
                }
        except Exception as e:
            logger.error(f"Stream translation failed: {str(e)}")
            raise


class SRTParseError(Exception):
    """Custom exception for SRT parsing errors"""
    pass


class SRTTranslationError(Exception):
    """Custom exception for translation errors"""
    pass


# Usage Example
if __name__ == "__main__":
    from llama_index.llms.openai import OpenAI
    
    try:
        translator = SRTTranslator(
            llm=OpenAI(model="gpt-4", temperature=0.3),
            target_language="es",  # Using ISO 639-1 code
            tone_guidelines={
                'style': 'Natural Latin American Spanish',
                'technical': 'Use common localized terms',
                'humor': 'Adapt jokes for cultural relevance'
            }
        )
        
        result = translator.translate_srt(
            Path("input.srt"),
            Path("output.srt"),
            save_every=30
        )
        print(f"Translation completed successfully. Output length: {len(result)} characters")
        
    except Exception as e:
        print(f"Translation failed: {str(e)}")
        raise


class SRTParseError(Exception):
    """Custom exception for SRT parsing errors"""
    pass


class SRTTranslationError(Exception):
    """Custom exception for translation errors"""
    pass


# Usage Example
if __name__ == "__main__":
    from llama_index.llms.openai import OpenAI
    
    try:
        translator = SRTTranslator(
            llm=OpenAI(model="gpt-4", temperature=0.3),
            target_language="es",  # Using ISO 639-1 code
            tone_guidelines={
                'style': 'Natural Latin American Spanish',
                'technical': 'Use common localized terms',
                'humor': 'Adapt jokes for cultural relevance'
            }
        )
        
        result = translator.translate_srt(
            Path("input.srt"),
            Path("output.srt"),
            save_every=30
        )
        print(f"Translation completed successfully. Output length: {len(result)} characters")
        
    except Exception as e:
        print(f"Translation failed: {str(e)}")
        raise
