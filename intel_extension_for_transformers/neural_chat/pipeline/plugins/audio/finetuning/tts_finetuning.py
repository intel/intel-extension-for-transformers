from intel_extension_for_transformers.neural_chat.config import TTSFinetuningConfig, TrainingArguments, TTSDatasetArguments, TTSModelArguments

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Audio, Dataset, Features, ClassLabel
import os
import torch
from speechbrain.pretrained import EncoderClassifier
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import SpeechT5HifiGan
import soundfile as sf
from datasets import Dataset


@dataclass
class TTSDataCollatorWithPadding:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = self.processor.pad(
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
        if self.model.config.reduction_factor > 1:
            target_lengths = torch.tensor([
                len(feature["input_values"]) for feature in label_features
            ])
            target_lengths = target_lengths.new([
                length - length % self.model.config.reduction_factor for length in target_lengths
            ])
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch
 
class TTSFinetuning:
    def __init__(self, finetuning_config: TTSFinetuningConfig):
        self.dataset_args, self.model_args = (
            finetuning_config.dataset_args,
            finetuning_config.model_args
        )
        self.audio_folder_path = self.dataset_args.audio_folder_path
        self.text_folder_path = self.dataset_args.text_folder_path
        self.gender = self.dataset_args.gender
        self.language = self.dataset_args.language
        self.step = self.model_args.step
        self.warmup_step = self.model_args.warmup_step
        self.learning_rate = self.model_args.learning_rate

        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.language_lst = ['en', 'de', 'fr', 'es', 'pl', 'it', 'ro', 
                            'hu', 'cs', 'nl', 'fi', 'hr', 'sk', 'sl', 'et', 'lt', 'en_accented']
    def _construct_text_list(self):
        try:
            text_paths = sorted(os.listdir(self.text_folder_path),
                    key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        except ValueError as e:
            raise(f"Please make sure that your texts under {self.text_folder_path} are named like 1.txt, 2.txt...")
        texts = []
        for p in text_paths:
            with open(os.path.join(self.text_folder_path, p)) as f:
                texts.append(f.read())
        normalized_texts = [i.lower().replace(",","").replace(".", "") + "." for i in texts]
        return texts, normalized_texts

    def _construct_audio_list(self):
        try:
            audio_paths = sorted(os.listdir(self.audio_folder_path),
                                 key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        except ValueError as e:
            raise(f"Please make sure that your audios under {self.audio_folder_path} are named like 1.wav, 2.wav...")
        audio_paths = [os.path.join(self.audio_folder_path, i) for i in audio_paths]
        return audio_paths
    
    def _construct_finetuning_dataset(self):
        raw_texts, normalized_texts = self._construct_text_list()
        audio_paths = self._construct_audio_list()
        if len(raw_texts) != len(audio_paths):
            raise Exception("The length of files under the audio folder and the text folder should be the same!")
        L = len(audio_paths)
        dataset = Dataset.from_dict({
            "audio_id": [f"id{i+1}" for i in range(L)],
            "language": [self.language_lst.index(self.language) for i in range(L)],
            "audio": audio_paths,
            'raw_text': raw_texts,
            'normalized_text': normalized_texts,
            'gender': [self.gender for i in range(L)],
            'speaker_id': ['10001' for i in range(L)],
            "is_gold_transcript": [True for i in range(L)],
            "accent": ["None" for i in range(L)]}).cast_column(
                "audio", Audio(sampling_rate=16000)).cast_column(
                    "language", ClassLabel(names=['en', 'de', 'fr', 'es', 'pl', 'it', 'ro', 
                                                    'hu', 'cs', 'nl', 'fi', 'hr', 'sk', 'sl', 'et', 'lt', 
                                                    'en_accented'], id=None))
        return dataset

    def _construct_training_arguments(self):
        training_args = Seq2SeqTrainingArguments(
            output_dir="./output",
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_step,
            max_steps=self.step,
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="steps",
            per_device_eval_batch_size=8,
            save_steps=self.step,
            eval_steps=self.step,
            logging_steps=25,
            load_best_model_at_end=True,
            greater_is_better=False,
            label_names=["labels"],
        )
        return training_args

    def _create_speaker_embedding(self, waveform):
        spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
        device = self.device
        speaker_model = EncoderClassifier.from_hparams(
            source=spk_model_name,
            run_opts={"device": device},
            savedir=os.path.join("/tmp", spk_model_name)
        )
        with torch.no_grad():
            speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
            speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
            speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
        return speaker_embeddings

    def _prepare_dataset(self, example):
        # load the audio data; if necessary, this resamples the audio to 16kHz
        audio = example["audio"]

        # feature extraction and tokenization
        example = self.processor(
            text=example["normalized_text"],
            audio_target=audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=False,
        )

        # strip off the batch dimension
        example["labels"] = example["labels"][0]

        # use SpeechBrain to obtain x-vector
        example["speaker_embeddings"] = self._create_speaker_embedding(audio["array"])

        return example

    def _is_not_too_long(self, input_ids):
        input_length = len(input_ids)
        return input_length < 200

    def finetune(self):
        dataset = self._construct_finetuning_dataset()
        dataset = dataset.map(
            self._prepare_dataset, remove_columns=dataset.column_names,
        ).filter(
            self._is_not_too_long, input_columns=["input_ids"]
        )
        dataset = dataset.train_test_split(test_size=0.1)
        data_collator = TTSDataCollatorWithPadding(model=self.model, processor=self.processor)

        training_args = self._construct_training_arguments()
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            tokenizer=self.processor.tokenizer,
        )
        trainer.train()

        return self.model