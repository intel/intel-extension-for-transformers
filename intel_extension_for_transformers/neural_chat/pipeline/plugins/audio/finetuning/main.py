
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset, Audio, Dataset, Features, ClassLabel
import os
import torch
from speechbrain.pretrained import EncoderClassifier
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import SpeechT5HifiGan
import soundfile as sf

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

device = "cuda" if torch.cuda.is_available() else "cpu"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name)
)

def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

def prepare_dataset(example):
    # load the audio data; if necessary, this resamples the audio to 16kHz
    audio = example["audio"]

    # feature extraction and tokenization
    example = processor(
        text=example["normalized_text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )

    # strip off the batch dimension
    example["labels"] = example["labels"][0]

    # use SpeechBrain to obtain x-vector
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

    return example

####### Prepare Pat's dataset ########
from datasets import Dataset

# raw_txt = ['I am so happy to speak to our FPGA community of customers and partners. Each of you innovates in so many areas, from data center and cloud to networking and countless embedded applications, whether it be transportation, manufacturing, defense, video, medical and the list goes on and on. FPGA demand has been on fire these past several years, resulting in challenges on the supply front to keep up. I appreciate your patience and support as we have worked to increase capacity and build resiliency into our FPGA and intel supply chain.',
#   'While we\'ve considerably increased supply and reduced our lead times, we continue to invest in this space to ensure intel is the industry leader, reliably supplying critical semiconductors to customers and mitigating future supply disruptions. That\'s why in PSG we\'re developing a full suite of custom logic solutions from FPGAs to standard cell and structured Asics to take advantage of the huge investments intel is making in wafer packaging and test capacity. Of course, supply is important, but we also need to deliver the most advanced products in the world. Our leadership, performance, per watt and compelling features of the Agilex Seven and Nine families are now starting to roll out and ramp, including our recently announced Agilex Seven FPGAs with Ftile demonstrating our bandwidth, leadership and next gen innovation. And I got one right here.',
#   'We will also start sampling our new midrange Agilex Five products while making advances to our quartess development environment to improve ease of use and performance. We know how important it is to keep pace with your innovations. We look forward to rolling out new families of products so that we can collaborate on designs that deliver the performance and features that you need.']
raw_txt = ['I am so happy to speak to our FPGA community of customers and partners.',
           'Each of you innovates in so many areas, from data center and cloud to networking and countless embedded applications',
           'whether it be transportation, manufacturing, defense, video, medical, and the list goes on and on.']
# raw_txt = [
#     'absolutely so as you mentioned i lead our worldwide team of engineers who are responsible for delivering end to end ai software solutions',
#         'and this means that they are optimized to achieve peak performance across a variety of hardware architecture',
#         'we work with popular open source product ecosystem like tensorflow we are working with google or pytorch facebook and mxnet amazon so we work with a popular open source product ecosystem product',
#         'we want to make sure that we meet all our customer needs meaning that our ai software is easy to use we are focusing on simplicity',
#         'easy to use we are focusing on simplicity a productivity gains the software is performant it is unified',
#         'so if you want to deploy ai software today it is not very straightforward you have to find the right solution the right software you have to find the right model right data set',
#         'and all these things require skill set training people who understand ai and our goal is to make sure that that it is easy to use for different types of persona different types of customer',
#         'so we want to make sure the software solutions have a low code or no code meaning very easy to use it is like a turnkey',
#         'it is easy to use containers or we create vertical toolkits reference tools again the focus being that it is performant and productive and it is out of box that is the goal that we want to give our customers a very good out of box experience']

normalized_txt = [i.lower().replace(",","").replace(".", "") + "." for i in raw_txt]
print(normalized_txt)
print()
# S = 6
pat_dataset = Dataset.from_dict({"audio_id": ["id1", "id2", "id3"], "language": [0, 0, 0], "audio": ["vgjwo-5bunm-1.mp3", "vgjwo-5bunm-2.mp3", "vgjwo-5bunm-3.mp3"], 'raw_text': raw_txt, 'normalized_text': normalized_txt,
                                 "gender": ['male', 'male', 'male'],
                                 'speaker_id': ['10001', '10001', '10001'], "is_gold_transcript": [True, True, True], "accent": ["None", "None", "None"]}).cast_column("audio", Audio(sampling_rate=16000)).cast_column("language", ClassLabel(names=['en', 'de', 'fr', 'es', 'pl', 'it', 'ro', 'hu', 'cs', 'nl', 'fi', 'hr', 'sk', 'sl', 'et', 'lt', 'en_accented'], id=None))
# L = len(normalized_txt)
# print(normalized_txt)
# pat_dataset = Dataset.from_dict({"audio_id": [f"id{i+1}" for i in range(S,L)], "language": [0 for i in range(S,L)], "audio": [f"huma_all_{i+1}.mp3" for i in range(S,L)], 'raw_text': raw_txt[S:L], 'normalized_text': normalized_txt[S:L],
#                                  "gender": ['female' for i in range(S,L)],
#                                  'speaker_id': ['10001' for i in range(S,L)], "is_gold_transcript": [True for i in range(S,L)], "accent": ["None" for i in range(S,L)]}).cast_column("audio", Audio(sampling_rate=16000)).cast_column("language", ClassLabel(names=['en', 'de', 'fr', 'es', 'pl', 'it', 'ro', 'hu', 'cs', 'nl', 'fi', 'hr', 'sk', 'sl', 'et', 'lt', 'en_accented'], id=None))

print(pat_dataset.features)
for i in pat_dataset.features:
  print(pat_dataset[0][i])

pat_dataset = pat_dataset.map(
    prepare_dataset, remove_columns=pat_dataset.column_names,
)

# Some of the examples in the dataset are apparently longer than the maximum input length the model can handle (600 tokens), 
# so we should remove those from the dataset. In fact, to allow for larger batch sizes we'll remove anything over 200 tokens.
def is_not_too_long(input_ids):
    input_length = len(input_ids)
    return input_length < 200

pat_dataset = pat_dataset.filter(is_not_too_long, input_columns=["input_ids"])


pat_dataset = pat_dataset.train_test_split(test_size=0.1)
print(pat_dataset)


# Collator to make batches

# We need to define a custom collator to combine multiple examples into a batch. This will pad shorter sequences with padding tokens. For the spectrogram labels, the padded portions are replaced with the special value -100. This special value tells the model to ignore that part of the spectrogram when calculating the spectrogram loss.


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(
            input_ids=input_ids,
            labels=label_features,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor([
                len(feature["input_values"]) for feature in label_features
            ])
            target_lengths = target_lengths.new([
                length - length % model.config.reduction_factor for length in target_lengths
            ])
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch

data_collator = TTSDataCollatorWithPadding(processor=processor)


######## Training ###########
# model.config.use_cache = False
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./output",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    # warmup_steps=500,
    # max_steps=4000,
    warmup_steps=125, # !!!!!!!!!! TODO change it back
    max_steps=1000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    # report_to=["tensorboard"],
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    # push_to_hub=True,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=pat_dataset["train"],
    eval_dataset=pat_dataset["test"],
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
)

trainer.train()

# get new fine-tuned model, use cpu to infer
# Do not use cpu!
# model = trainer.model.to('cpu')
# model = model.to('cpu')

###### Inference #####
# texts = ["this means that they are optimized to achieve peak performance across a variety of hardware architecture"]
texts = ["We will enable more technologies on Artificial Intelligence.", "Artificial Intelligence is an incredibly exciting field with endless possibilities. As a young Pat Gelsinger, I would tell myself to embrace AI's potential to revolutionize industries and improve lives. It's a fascinating journey that I'm eager to take part in.",
         "The China market is a dynamic and rapidly growing market with tremendous opportunities. It has a diverse consumer base and a thriving tech industry. Investing in the China market can lead to significant growth and expansion for businesses. However, it's important to understand the unique cultural, regulatory, and competitive landscape to navigate successfully.",
         "To the developers in Intel China, I want to say that your work is invaluable and plays a crucial role in shaping the future of technology. Your dedication and expertise are greatly appreciated. Keep pushing boundaries, innovating, and creating amazing things. You're making a difference, and I'm excited to see what you'll accomplish next!"]
# text = "We will enable more technologies on Artificial Intelligence."
# text = "Artificial Intelligence is an incredibly exciting field with endless possibilities. As a young Pat Gelsinger, I would tell myself to embrace AI's potential to revolutionize industries and improve lives. It's a fascinating journey that I'm eager to take part in."
# text = "The China market is a dynamic and rapidly growing market with tremendous opportunities. It has a diverse consumer base and a thriving tech industry. Investing in the China market can lead to significant growth and expansion for businesses. However, it's important to understand the unique cultural, regulatory, and competitive landscape to navigate successfully."
# text = "To the developers in Intel China, I want to say that your work is invaluable and plays a crucial role in shaping the future of technology. Your dedication and expertise are greatly appreciated. Keep pushing boundaries, innovating, and creating amazing things. You're making a difference, and I'm excited to see what you'll accomplish next!"

# use hf utility to convert audio to waveform
audio_dataset = Dataset.from_dict({"audio": ["vgjwo-5bunm.mp3"]}).cast_column("audio", Audio(sampling_rate=16000))
# audio_dataset = Dataset.from_dict({"audio": ["huma_all_8.mp3"]}).cast_column("audio", Audio(sampling_rate=16000))
sembeddings = create_speaker_embedding(audio_dataset[0]["audio"]['array'])
speaker_embeddings = torch.tensor(sembeddings).unsqueeze(0)

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").cuda()


for i, text in enumerate(texts):
    inputs = processor(text=text, return_tensors="pt")
    spectrogram = model.generate_speech(inputs["input_ids"].cuda(), speaker_embeddings.cuda())  # here should move input to cuda

    with torch.no_grad():
        speech = vocoder(spectrogram)

    sf.write(f"output_{i}.wav", speech.cpu().numpy(), samplerate=16000)

torch.save(model,"pat.pt")
print("done")