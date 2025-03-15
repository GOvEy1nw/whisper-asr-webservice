import os
import subprocess
import tempfile
import time
from io import StringIO
from threading import Thread
from typing import Union

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG


class FasterWhisperXXLASR(ASRModel):
    def load_model(self):
        # This model uses an executable, so there's no need to load a model
        # Just verify that the executable exists
        if not os.path.exists(CONFIG.FASTER_WHISPER_XXL_PATH):
            raise FileNotFoundError(f"Faster-Whisper-XXL executable not found at {CONFIG.FASTER_WHISPER_XXL_PATH}")

        # Start monitoring thread
        Thread(target=self.monitor_idleness, daemon=True).start()

    def transcribe(
        self,
        audio_file,
        task: Union[str, None],
        language: Union[str, None],
        initial_prompt: Union[str, None],
        vad_filter: Union[bool, None],
        word_timestamps: Union[bool, None],
        options: Union[dict, None],
        output,
    ):
        """
        Transcribe audio using the Faster-Whisper-XXL executable.

        Parameters:
        - audio_file: path to the audio file
        - task: transcribe or translate
        - language: language code
        - initial_prompt: text for initial prompt
        - vad_filter: enable voice activity detection
        - word_timestamps: enable word-level timestamps
        - options: additional options
        - output: output format (txt, vtt, srt, tsv, json)
        """
        self.last_activity_time = time.time()

        # Create a temporary directory for the output
        with tempfile.TemporaryDirectory() as temp_output_dir:
            # Construct the command to run Faster-Whisper-XXL
            cmd = [CONFIG.FASTER_WHISPER_XXL_PATH, audio_file]

            # Add task parameter (transcribe or translate)
            if task:
                cmd.extend(["--task", task])

            # Add language parameter if specified
            if language:
                cmd.extend(["-l", language])

            # Add initial prompt if specified
            if initial_prompt:
                cmd.extend(["--initial_prompt", initial_prompt])

            # Add VAD filter if enabled
            if vad_filter:
                cmd.extend(["--vad_filter", "True"])

            # Add word timestamps if enabled
            if word_timestamps:
                cmd.extend(["--word_timestamps", "True"])

            # Add standard formatting for subtitles and max gap setting
            if CONFIG.FASTER_WHISPER_XXL_STANDARD:
                cmd.append("--standard")

            # Set maximum gap between subtitle segments
            cmd.extend(["--max_gap", str(CONFIG.FASTER_WHISPER_XXL_MAX_GAP)])

            # Set output format and directory
            cmd.extend(["-f", output if output in ["srt", "vtt", "txt", "json", "tsv"] else "txt"])
            cmd.extend(["-o", temp_output_dir])

            # Run the command
            try:
                subprocess.run(cmd, check=True, capture_output=True)

                # Find the output file
                output_files = [f for f in os.listdir(temp_output_dir) if f.endswith(f".{output}")]
                if not output_files:
                    raise FileNotFoundError(f"No output file with extension .{output} found in {temp_output_dir}")

                output_file_path = os.path.join(temp_output_dir, output_files[0])

                # Read the output file
                with open(output_file_path, "r", encoding="utf-8") as f:
                    result_content = f.read()

                # Create a StringIO object for the result
                result_file = StringIO(result_content)
                result_file.seek(0)

                return result_file

            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to run Faster-Whisper-XXL: {e.stderr.decode() if e.stderr else str(e)}")

    def language_detection(self, audio_file):
        """
        Detect language using the Faster-Whisper-XXL executable.

        Parameters:
        - audio_file: path to the audio file

        Returns:
        - language_code: detected language code
        - confidence: confidence score for the detection
        """
        self.last_activity_time = time.time()

        # Run Faster-Whisper-XXL with language detection segments option
        cmd = [CONFIG.FASTER_WHISPER_XXL_PATH, audio_file, "--language_detection_segments", "1"]

        try:
            # Run the command
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Extract language from the output
            output_lines = result.stdout.splitlines()
            language_info = None
            confidence = 0.0

            # Parse the output to find the detected language
            for line in output_lines:
                if "Detected language:" in line:
                    language_info = line.split("Detected language:")[1].strip()
                    language_code = language_info.split()[0].strip()
                    confidence_str = line.split("probability:")[1].strip() if "probability:" in line else "1.0"
                    confidence = float(confidence_str)
                    return language_code, confidence

            # If no language detected, return a default
            return "en", 0.0

        except subprocess.CalledProcessError as e:
            # Return a default on error
            print(f"Error detecting language: {e.stderr.decode() if e.stderr else str(e)}")
            return "en", 0.0
