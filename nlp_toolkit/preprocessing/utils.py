import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
from enum import Enum


EOS = '</s>'


class AugmenterType(Enum):
    TEXTGENERATIONAUG = "textgenerationaug"
    KEYBOARDAUG = "KeyboardAug"
    OCRAUG = "OcrAug"
    SPELLINGAUG = "SpellingAug"
    CONTEXTUALWORDEMBSFORSENTENCEAUG = "ContextualWordEmbsForSentenceAug"


AUGMENTER_MAPPING = {
    AugmenterType.KEYBOARDAUG.value: nac,
    AugmenterType.OCRAUG.value: nac,
    AugmenterType.SPELLINGAUG.value: naw,
    AugmenterType.CONTEXTUALWORDEMBSFORSENTENCEAUG.value: nas,

}


def get_augmenter_from_type(aug_type: str):
    assert aug_type in AUGMENTER_MAPPING, "Unspported the augmenter type:{}".format(aug_type)
    return AUGMENTER_MAPPING[aug_type]
