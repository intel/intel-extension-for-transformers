import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
from enum import Enum


EOS = '</s>'


class AugmenterType(Enum):
    TEXTGENERATIONAUG = "textgenerationaug"
    KEYBOARDAUG = "KeyboardAug"


AUGMENTER_MAPPING = {
    AugmenterType.KEYBOARDAUG.value: nac,
}
def get_augmenter_from_type(aug_type: str):
    assert aug_type in AUGMENTER_MAPPING, "Unspported the augmenter type:{}".format(aug_type)
    return AUGMENTER_MAPPING[aug_type]
