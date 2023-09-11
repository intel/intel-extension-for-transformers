from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.asr import AudioSpeechRecognition
import argparse
parser = argparse.ArgumentParser(
                    prog='asr',
                    description='Audio Speech Recognition')
parser.add_argument('-i', '--input_audio')
parser.add_argument('-m', '--model_name_or_path', default="openai/whisper-tiny")
parser.add_argument('-d', '--device', default="cuda")
args = parser.parse_args()
asr = AudioSpeechRecognition(model_name_or_path=args.model_name_or_path, device=args.device)
text = asr.audio2text(args.input_audio)
print(text)