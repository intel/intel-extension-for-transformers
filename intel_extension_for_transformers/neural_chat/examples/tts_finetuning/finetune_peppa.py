from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.finetuning.tts_finetuning import TTSFinetuning
from intel_extension_for_transformers.neural_chat.config import TTSFinetuningConfig, TTSDatasetArguments, TTSModelArguments
import torch
import os

workdir = os.getcwd()
data_args = TTSDatasetArguments(audio_folder_path=os.path.join(workdir, "audios"),
                                text_folder_path=os.path.join(workdir, "texts"),
                                gender="male")
model_args = TTSModelArguments(step=2000, warmup_step=125, learning_rate=1e-5)
finetuning_config = TTSFinetuningConfig(data_args, model_args)

tts_fintuner = TTSFinetuning(finetuning_config=finetuning_config)
finetuned_model = tts_fintuner.finetune()

torch.save(finetuned_model, "peppa.pt")


