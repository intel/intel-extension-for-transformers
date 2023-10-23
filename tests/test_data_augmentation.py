import os
import shutil
import unittest
from datasets import load_dataset
from intel_extension_for_transformers.utils.data_augmentation import DataAugmentation
from intel_extension_for_transformers.utils.utils import EOS

def build_fake_dataset(save_path, split, count):
    dataset = load_dataset('glue', 'sst2', split=split)
    origin_data = os.path.join(save_path, f'{split}.csv')

    with open(origin_data, 'w') as fw:
        fw.write('label' + '\t' + 'sentence' + '\n')
        for d in dataset:
            fw.write(f"{d['label']}\t{d['sentence']}{EOS}\n")
            count -= 1
            if count == 0:
                break
    return origin_data

def test_augmentation(augmenter_type, model_name_or_path, output_path, count):
    aug = DataAugmentation(augmenter_type=augmenter_type)
    aug.input_dataset = build_fake_dataset("test_data", "validation", 10)
    aug.output_path = os.path.join("test_data", output_path)

    if model_name_or_path:
        aug.augmenter_arguments = {'model_name_or_path': model_name_or_path}
    aug.data_augment()

    print(f"Augmented data ({augmenter_type}):")
    count = 0
    with open(aug.output_path, encoding='utf8') as f:
        for line in f:
            count += 1
            print(line)
    print("count:", count)
    return count

class TestDataAugmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs("test_data", exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("test_data", ignore_errors=True)

    def test_text_generation_augmentation(self):
        count = test_augmentation("TextGenerationAug", 'hf-internal-testing/tiny-random-gpt2', "test1.csv", 10)
        self.assertTrue(count == 11)

    def test_keyboard_augmentation(self):
        count = test_augmentation("KeyboardAug", None, "test2.csv", 10)
        self.assertTrue(count == 10)

    def test_ocr_augmentation(self):
        count = test_augmentation("OcrAug", None, "test3.csv", 10)
        self.assertTrue(count == 10)

    def test_spelling_augmentation(self):
        count = test_augmentation("SpellingAug", None, "test4.csv", 10)
        self.assertTrue(count == 10)

    def test_contextualwordembsforsentence_augmentation(self):
        count = test_augmentation("ContextualWordEmbsForSentenceAug", 'hf-internal-testing/tiny-random-xlnet', "test5.csv", 10)
        self.assertTrue(count == 10)

if __name__ == "__main":
    unittest.main()
