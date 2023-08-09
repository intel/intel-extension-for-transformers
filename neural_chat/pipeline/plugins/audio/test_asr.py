from asr import AudioSpeechRecognition
import unittest
import shutil

class TestASR(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.asr = AudioSpeechRecognition("openai/whisper-tiny")
        self.asr_bf16 = AudioSpeechRecognition("openai/whisper-tiny", bf16=True)

    def test_audio2text(self):
        audio_path = "pat.wav"
        text = self.asr.audio2text(audio_path)
        self.assertEqual(text.lower(), "Welcome to Neural Chat".lower())

    def test_audio2text(self):
        audio_path = "pat.wav"
        text = self.asr_bf16.audio2text(audio_path)
        self.assertEqual(text.lower(), "Welcome to Neural Chat".lower())

if __name__ == "__main__":
    unittest.main()
