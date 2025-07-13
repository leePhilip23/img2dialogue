from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils import log
from transformers import AutoTokenizer, Blip2VisionModel, CLIPVisionModel, AutoModelForCausalLM
from .exceptions import ImgModelNotSupported, ImgModelNotFound, LanguageModelNotFound, TxtTokenizerNotFound


class SpecialTokens(Enum):
    """All special tokens used for model"""
    BOS: str = "<BOS>"
    EOS: str = "<EOS>"
    IGNORE_TOKEN: int = -100


class Model(nn.Module):
    def __init__(
        self, 
        img_model: str = "Salesforce/blip-image-captioning-base", 
        small_lm: str = "Qwen/Qwen3-0.6B"
    ):
        super(Model, self).__init__()
        self.img_model = self._load_vision_model(img_model)
        self.tokenizer = self._load_txt_token(small_lm)
        self.slm = self._load_slm(small_lm)
        self.word_embed = self.slm.model.get_input_embeddings()

        #TODO: Make config for this
        self.slm_projector = nn.Linear(768, 1024)


    def _load_txt_token(self, token_name: str) -> AutoTokenizer:
        """Loads the text tokenizer and returns it"""
        try:
            return AutoTokenizer.from_pretrained(
                token_name, 
                use_fast=True,
                padding_side='left',
                bos_token=SpecialTokens.BOS.value, 
                eos_token=SpecialTokens.EOS.value
            )
        except TxtTokenizerNotFound:
            log.error(f"Tokenizer not found for: {token_name} in Hugging Face Directory")
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
            return AutoModelForCausalLM.from_pretrained(
                slm_name,
                low_cpu_mem_usage=True
            )
        except LanguageModelNotFound:
            log.error(f"Language Model: {slm_name} not found in Hugging Face directory")
            raise


    def _single_features(
        self, 
        bos: Tensor, 
        img_feature: Tensor,
        tokens: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Calculates features for BOS token, input text tokens and combines
        them with the image features to provide model input.
        """
        bos_token = self.word_embed(bos)
        input_tokens = self.word_embed(tokens)
        comb_features = torch.cat((bos_token, img_feature, input_tokens), dim=0)
        return comb_features, tokens.attention_masks

    
    def _batch_features(
        self, 
        bos_input: Tensor,
        img_feature: Tensor, 
        tok_inputs: Tensor, 
        outputs: Tensor
    ) -> tuple[
            list[Tensor], 
            list[Tensor], 
            list[Tensor]
        ]:
        """
        Creates data batches for batch inference. This requires concatenation of image
        and text features as well as padding for each row of data.

        Args:
        bos_input (Tensor): BOS tokens for each item
        tok_inputs (Tensor): Tokenized input sequences
        img_feature (Tensor): Image features to concatenate
        outputs (Tensor): Target output sequences

        Returns:
            tuple[list[Tensor], list[Tensor], list[Tensor]]:
                - batch_inputs: Concatenated BOS, input tokens, and image features
                - batch_masks: Attention masks
                - batch_outputs: Target output sequences
        """
        batch_inputs = []
        batch_outputs = []
        batch_masks = []

        # Append concatenated embeddings for each batch
        for row in range(len(tok_inputs)):
            bos_token = self.word_embed(bos_input)
            input_ids = torch.as_tensor(tok_inputs[row].unsqueeze(0))
            output_ids = torch.as_tensor(outputs[row].unsqueeze(0))

            comb_features, mask = self._single_features(bos_token, img_feature, input_ids)
            batch_inputs.append(comb_features)
            batch_outputs.append(output_ids)
            batch_masks.append(mask[row])

        return batch_inputs, batch_masks, batch_outputs
            

    def forward(
        self, 
        img: Tensor, 
        input: Tensor,
        labels: Tensor,
        batch_predict: bool = True
    ) -> Tensor:
        """Model does forward prediction"""

        pooled_tensor = self.img_model(img)
        img_feature = self.slm_projector(pooled_tensor)

        bos = self.tokenizer(SpecialTokens.BOS).input_ids
        tokens = self.tokenizer(input).input_ids
        outputs = self.tokenizer(labels).input_ids

        if batch_predict:
            inputs, masks, outputs = self._batch_features(bos, tokens, img_feature, outputs)
        else:
            inputs, masks = self._single_features(bos, tokens, img_feature, outputs)

        # Pad all combined embeddings
        input_embeds = nn.utils.rnn.pad_sequence(
            inputs, 
            batch_first=True, 
            padding_side='left'
        )
        attention_mask = nn.utils.rnn.pad_sequence(
            masks, 
            batch_first=True, 
            padding_side='left', 
            padding_value=SpecialTokens.IGNORE_TOKEN
        )
        labels = nn.utils.rnn.pad_sequence(
            outputs, 
            batch_first=True, 
            padding_side='left'
        )

        return self.slm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )