class HuggingFaceDataNotFound(Exception):
    """Hugging Face Dataset not found"""
    pass

class ImgProcessorNotSupported(Exception):
    """Hugging Face Image Processor not among supported choices"""
    pass

class ImgProcessorNotFound(Exception):
    """Hugging Face Image Processor not found"""
    pass

class TxtTokenizerNotFound(Exception):
    """Hugging Face Text Tokenizer not found"""
    pass

class ImgModelNotFound(Exception):
    """Image Model is not found in Hugging Face directory"""
    pass

class ImgModelNotSupported(Exception):
    """Image Model is not supported in this module"""
    pass


class LanguageModelNotFound(Exception):
    """Language Model not found in Hugging Face directory"""