import torch
import intel_extension_for_pytorch as ipex
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset, Audio, Dataset
import time

class AudioSpeechRecognition:
    """Convert audio to text."""
    def __init__(self):
        self.device = "cpu"
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(self.device)
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model.eval()
        #self.model = ipex.optimize(self.model, dtype=torch.bfloat16)

    def audio2text(self, audio_path):
        """Convert audio to text

        audio_path: the path to the input audio, e.g. ~/xxx.mp3
        """
        start = time.time()
        audio_dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column("audio", Audio(sampling_rate=16000))
        waveform = audio_dataset[0]["audio"]['array']
        inputs = self.processor.feature_extractor(waveform, return_tensors="pt", sampling_rate=16_000).input_features.to(self.device)
        #with torch.cpu.amp.autocast():
        predicted_ids = self.model.generate(inputs)
        result = self.processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True, normalize=True)[0]
        print(f"generated text in {time.time() - start} seconds, and the result is: {result}")
        return result
