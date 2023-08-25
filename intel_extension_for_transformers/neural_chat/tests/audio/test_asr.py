from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.asr import AudioSpeechRecognition
import unittest
import shutil
import torch

class TestASR(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.asr = AudioSpeechRecognition("openai/whisper-small", device=device)
        if not torch.cuda.is_available():
            self.asr_bf16 = AudioSpeechRecognition("openai/whisper-small", bf16=True)

    def test_audio2text(self):
        audio_path = "../../assets/audio/pat.wav"
        text = self.asr.audio2text(audio_path)
        self.assertEqual(text.lower(), "Welcome to Neural Chat".lower())

    def test_audio2text_bf16(self):
        if torch.cuda.is_available():
            return
        audio_path = "../../assets/audio/pat.wav"
        text = self.asr_bf16.audio2text(audio_path)
        self.assertEqual(text.lower(), "Welcome to Neural Chat".lower())

if __name__ == "__main__":
    unittest.main()
