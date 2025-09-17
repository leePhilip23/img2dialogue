import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import Optional
from torch import Tensor
from transformers import (
    AutoTokenizer, 
    Blip2VisionModel, 
    CLIPVisionModel, 
    AutoModelForCausalLM
)
from .logger import log
from .exceptions import (
    ImgModelNotSupported, 
    ImgModelNotFound, 
    LanguageModelNotFound, 
    TxtTokenizerNotFound
)


class Tokens(Enum):
    """All special tokens used for model"""
    BOS: str = "<BOS>"
    EOS: str = "<EOS>"
    BEGIN_Q: str = "[Q]"
    END_Q: str = "[/Q]"
    BEGIN_A: str = "[A]"
    END_A: str = "[/A]"
    MAX_LEN: int = 512
    IGNORE: int = -100
    PAD: int = 0


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
        self.img_projector = nn.Linear(768, 1024)


    def _load_txt_token(self, token_name: str) -> AutoTokenizer:
        """Loads the text tokenizer and returns it"""
        try:
            return AutoTokenizer.from_pretrained(
                token_name, 
                use_fast=True,
                padding_side='left',
                bos_token=Tokens.BOS.value, 
                eos_token=Tokens.EOS.value
            )
        except TxtTokenizerNotFound as e:
            log.error(f"Tokenizer not found for: {token_name} in Hugging Face Directory")
            raise TxtTokenizerNotFound(e)


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


    def _single_feature(
        self,
        bos_feature : Tensor,
        img_feature: Tensor,
        att_masks: list[int],
        tokens: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Calculates features for input text tokens and combines
        them with the image features to provide model input.
        """
        input_masks = torch.as_tensor([1, 1] + att_masks)
        token_feats = self.word_embed(tokens)
        cat_inputs = torch.cat((bos_feature, img_feature, token_feats), dim=0)
        return input_masks, cat_inputs

    
    def _batch_features(
        self, 
        bos_feature: Tensor,
        img_feature: Tensor, 
        tok_inputs: dict[str, list[int]]
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
        batch_masks = []

        # Append concatenated embeddings for each batch
        for row in range(len(tok_inputs.input_ids)):
            mask, input_feats = self._single_feature(
                bos_feature.squeeze(0),
                img_feature[row].unsqueeze(0),
                tok_inputs.attention_mask[row],
                torch.as_tensor(tok_inputs.input_ids[row])
            )

            batch_masks.append(mask)
            batch_inputs.append(input_feats)

        return batch_masks, batch_inputs
        
    
    def _padding(
        self, 
        masks: list[Tensor], 
        inputs: list[Tensor], 
        outputs: Optional[list]):
        """
        Pads the input and output tensors to the maximum length defined by Tokens.MAX_LEN.
        This is necessary for batch processing in transformers.

        Args:
            masks (list[Tensor]): List of attention masks for each input sequence
            inputs (list[Tensor]): List of input feature tensors
            outputs (list[Tensor]): List of output feature tensors

        Returns:
            tuple[Tensor, Tensor, Tensor]: Padded attention masks, input features, and output features
        """

        padded_masks = nn.utils.rnn.pad_sequence(
            masks, 
            batch_first=True, 
            padding_side='left', 
            padding_value=Tokens.PAD.value
        )
        padded_inputs = nn.utils.rnn.pad_sequence(
            inputs, 
            batch_first=True, 
            padding_side='left', 
            padding_value=Tokens.PAD.value
        )

        if outputs:
            padded = []
            for t in outputs:
                pad_len = padded_inputs.size(1) - len(t)
                padded_tensor = F.pad(
                    torch.as_tensor(t), 
                    (pad_len, 0),
                    value=Tokens.IGNORE.value
                )
                padded.append(padded_tensor)
            padded_outputs = torch.stack(padded, dim=0)
            return padded_masks, padded_inputs, padded_outputs

        return padded_masks, padded_inputs
    

    def _run_model(
        self, 
        img: Tensor, 
        input: str,
        labels: Optional[str],
        batch_predict: bool=True
    ) -> dict:
        """Model does forward prediction"""

        pooled_tensor = self.img_model(img).pooler_output
        img_feature = self.img_projector(pooled_tensor)

        bos = self.tokenizer(Tokens.BOS.value, return_tensors='pt').input_ids
        bos_feature = self.word_embed(bos)

        tokens = self.tokenizer(input, truncation=True)

        if batch_predict:
            masks, inputs = self._batch_features(bos_feature, img_feature, tokens)
        else:
            masks, inputs = self._single_feature(img_feature, tokens)

        if labels:
            outputs = self.tokenizer(labels, truncation=True)
            attention_mask, input_embeds, labels = self._padding(masks, inputs, outputs.input_ids)
            return attention_mask, input_embeds, labels

        attention_mask, input_embeds = self._padding(masks, inputs, None)
        return attention_mask, input_embeds
    

    @torch.no_grad()
    def evaluate(
        self, 
        img: Tensor, 
        txt: Tensor,
        batch_predict: bool=True,
    ) -> dict:
        attention_mask, input_embeds = self._run_model(img, txt, None, batch_predict)
        return self.slm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True
        )
        
            
    def forward(
        self, 
        img: Tensor, 
        txt: Tensor,
        labels: Tensor,
        batch_predict: bool=True
    ) -> dict:
        attention_mask, input_embeds, labels = self._run_model(img, txt, labels, batch_predict)
        return self.slm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )