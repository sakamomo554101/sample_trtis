from enum import Enum


class ModelName(Enum):
    SAMPLE_INSTANCE = "sample_instance"
    SAMPLE_SEQUENCE = "sample_sequence"
    MECAB_MODEL = "mecab_model"
    FACE_MODEL = "face_recognition_model"

    @classmethod
    def get_all_names(cls):
        names = []
        for n in cls:
            names.append(n.value)
        return names
