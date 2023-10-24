import os
import shutil
import unittest

from datasets import load_dataset
from intel_extension_for_transformers.utils.data_augmentation import DataAugmentation


def build_fake_dataset(save_path):
    from intel_extension_for_transformers.utils.utils import EOS

    split = 'validation'
    count = 10
    dataset = load_dataset('glue', 'sst2', split='validation')
    origin_data = os.path.join(save_path, split + '.csv')
    print("original data:")
    with open(origin_data, 'w') as fw:
        fw.write('label' + '\t' + 'sentence' + '\n')
        for d in dataset:
            fw.write(str(d['label']) + '\t' + d['sentence'] + EOS + '\n')
            print(str(d['label']) + '\t' + d['sentence'] + EOS + '\n')
            count -= 1
            if count == 0:
                break
    return origin_data


class TestDataAugmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs("test_data", exist_ok=True)
        cls.result_path = "test_data"
        cls.origin_data = build_fake_dataset(cls.result_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.result_path, ignore_errors=True)

    def test_augmentation(self, augmenter_type, augmenter_arguments=None):
        aug = DataAugmentation(augmenter_type=augmenter_type)
        aug.input_dataset = self.origin_data
        aug.column_names = "sentence"
        aug.output_path = os.path.join(self.result_path, f"{augmenter_type}.csv")
        if augmenter_arguments:
            aug.augmenter_arguments = augmenter_arguments
        aug.data_augment()
        raw_datasets = load_dataset("csv", data_files=aug.output_path, delimiter="\t", split="train")
        return len(raw_datasets)

    def test_text_generation_augmentation(self):
        count = self.test_augmentation("TextGenerationAug", {'model_name_or_path': 'hf-internal-testing/tiny-random-gpt2'})
        print("Augmented data:")
        with open(os.path.join(self.result_path, "TextGenerationAug.csv"), encoding='utf8') as f:
            for line in f:
                print(line)
        print("count:", count)
        self.assertTrue(count == 11)

    def test_keyboard_augmentation(self):
        count = self.test_augmentation("KeyboardAug")
        self.assertTrue(count == 10)

    def test_ocr_augmentation(self):
        count = self.test_augmentation("OcrAug")
        self.assertTrue(count == 10)

    def test_spelling_augmentation(self):
        count = self.test_augmentation("SpellingAug")
        self.assertTrue(count == 10)

    def test_contextualwordembsforsentence_augmentation(self):
        count = self.test_augmentation("ContextualWordEmbsForSentenceAug", {"model_path": "hf-internal-testing/tiny-random-xlnet"})
        self.assertTrue(count == 10)


if __name__ == "__main__":
    unittest.main()
