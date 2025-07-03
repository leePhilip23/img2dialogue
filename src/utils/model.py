import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils import log
from transformers import AutoTokenizer, Blip2VisionModel, CLIPVisionModel, AutoModelForCausalLM
from .exceptions import ImgModelNotSupported, ImgModelNotFound, LanguageModelNotFound, TxtTokenizerNotFound


class SpecialTokens(enum):
    """All special tokens used for model"""
    BOS: str = "<BOS>"
    EOS: str = "<EOS>"
    IGNORE_TOKEN: int = -100


class Model(nn.Module):
    def __init__(
        self, 
        img_model_name: str = "Salesforce/blip-image-captioning-base", 
        slm_name: str = "Qwen/Qwen3-0.6B"
    ):
        self.img_model = self._load_vision_model(img_model_name)
        self.tokenizer = self._load_txt_token(slm_name)
        self.slm = self._load_slm(slm_name)
        self.word_embed = self.slm.model.get_input_embeddings()

        #TODO: Make config for this
        self.slm_projector = nn.Linear(768, 1024)


    def _load_txt_token(self, token_name: str) -> AutoTokenizer:
        """
        Loads the text tokenizer and returns it

        Args:
            url (str): Directory of Hugging Face Tokenizer
        
        Returns: 
            AutoTokenizer: Tokenizer downloaded from Hugging Face

        Raises:
            TxtTokenizerNotFound
        """
        try:
            return AutoTokenizer.from_pretrained(
                token_name, 
                use_fast=True,
                padding_side='left',
                bos_token=SpecialTokens.BOS, 
                eos_token=SpecialTokens.EOS
            )
        except TxtTokenizerNotFound:
            log.error(
                f"Tokenizer not found for: {token_name} in Hugging Face Directory"
            )
            raise

    
    def _load_vision_model(self, img_model_name: str) -> Blip2VisionModel | CLIPVisionModel:
        """Loads the vision model and returns it"""
        try:
            if "blip" in img_model_name.lower():
                return Blip2VisionModel.from_pretrained(img_model_name)
            elif "clip" in img_model_name.lower():
                return CLIPVisionModel.from_pretrained(img_model_name)
            else:
                log.error(f"Image Model: {img_model_name} is not a valid choice")
                raise ImgModelNotSupported("Model type is not among the supported choices")
        except ImgModelNotFound:
            log.error(f"Image Model: {img_model_name} is not found in Hugging Face")
            raise

        
    def _load_slm(self, slm_name: str) -> AutoModelForCausalLM:
        """Loads the Small Language Model and returns it"""
        try:
            return AutoModelForCausalLM.from_pretrained(slm_name)
        except LanguageModelNotFound:
            log.error(f"Language Model: {slm_name} not found in Hugging Face directory")
            raise
            

    def forward(
        self, 
        img: Tensor, 
        input: Tensor,
        mask: Tensor,
        labels: Tensor
    ) -> Tensor:
        """Model does forward prediction"""
        pooled_tensor = self.img_model(img)
        img_feature = self.slm_projector(pooled_tensor)

        bos_input = self.tokenizer(SpecialTokens.BOS).input_ids
        tok_inputs = self.tokenizer(input)
        batch_outputs = self.tokenizer(labels).input_ids

        batch_inputs = []
        batch_outputs = []
        batch_masks = []

        # Append concatenated embeddings for each batch
        for row in range(len(tok_inputs)):
            bos_token = self.word_embed(bos_input)
            input_tokens = self.word_embed(torch.as_tensor(tok_inputs.input_ids[row]))
            output_tokens = self.word_embed(torch.as_tensor(batch_outputs.input_ids[row]))
            comb_features = torch.cat((bos_token, img_feature, input_tokens), dim=0)
            batch_inputs.append(comb_features)
            batch_outputs.append(output_tokens)
            batch_masks.append(tok_inputs.attention_masks[row])


        # Pad all combined embeddings
        input_embeds = nn.utils.rnn.pad_sequence(
            batch_inputs, 
            batch_first=True, 
            padding_side='left'
        )
        attention_mask = nn.utils.rnn.pad_sequence(
            batch_masks, 
            batch_first=True, 
            padding_side='left', 
            padding_value=SpecialTokens.IGNORE_TOKEN
        )
        labels = nn.utils.rnn.pad_sequence(
            batch_outputs, 
            batch_first=True, 
            padding_side='left'
        )


        return self.slm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
