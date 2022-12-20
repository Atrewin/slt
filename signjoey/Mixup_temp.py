
#TODO models

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from itertools import groupby
from signjoey.initialization import initialize_model
from signjoey.embeddings import Embeddings, SpatialEmbeddings
from signjoey.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from signjoey.decoders import Decoder, RecurrentDecoder, TransformerDecoder
from signjoey.search import beam_search, greedy
from signjoey.vocabulary import (
    TextVocabulary,
    GlossVocabulary,
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
)
from signjoey.batch import Batch
from signjoey.helpers import freeze_params
from torch import Tensor
from typing import Union

# jinhui
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
import torch.nn as nn

class TextEmbeddings(nn.Module):

    """
    Simple Linear Projection Layer
    (For encoder outputs to predict glosses)
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        freeze: bool = False,
        norm_type: str = None,
        activation_type: str = None,
        scale: bool = False,
        scale_factor: float = None,
        max_length = 64,
        **kwargs
    ):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param input_size:
        :param freeze: freeze the embeddings during training
        """

        bert_path = "/apdcephfs/share_916081/shared_info/zhengshguo/jinhui/checkpoints/bert-base-german-cased"
        super().__init__()

        bert = BertForSequenceClassification.from_pretrained(
            bert_path,
            num_labels=embedding_dim)
        word_embedded = bert.base_model.embeddings
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        self.bert_word_embedding = word_embedded.word_embeddings # Embeddings
        self.vocab_size = self.bert_word_embedding.num_embeddings
        self.input_size = self.bert_word_embedding.embedding_dim
        self.embedding_dim = embedding_dim
        self.ln = nn.Linear(self.input_size, self.embedding_dim)

        self.norm_type = norm_type
        if self.norm_type:
            self.norm = MaskedNorm(
                norm_type=norm_type, num_groups=num_heads, num_features=embedding_dim
            )

        self.activation_type = activation_type
        if self.activation_type:
            self.activation = get_activation(activation_type)

        self.scale = scale
        if self.scale:
            if scale_factor:
                self.scale_factor = scale_factor
            else:
                self.scale_factor = math.sqrt(self.embedding_dim)

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(self, x: [], mask: Tensor) -> Tensor:
        """
        :param mask: frame masks
        :param x: input frame features
        :return: embedded representation for `x`
        """
        x = self.tokenize(x) # @jinhui 之后要放到外面取执行，并切to dievce
        x = self.bert_word_embedding(x)
        x = self.ln(x)

        if self.norm_type:
            x = self.norm(x, mask)

        if self.activation_type:
            x = self.activation(x)

        if self.scale:
            return x * self.scale_factor
        else:
            return x

    def __repr__(self):
        return "%s(embedding_dim=%d, vocab_size=%d)" % (
            self.__class__.__name__,
            self.embedding_dim,
            self.vocab_size,
        )

    def tokenize(self, raw_tokens):
        # token -> index
        # tokens = ['[CLS]']
        tokens = []
        for token in raw_tokens:
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.tokenizer.pad_token_id) #该用什么填充呢？
        indexed_tokens = indexed_tokens[:self.max_length]

        return torch.tensor(indexed_tokens, dtype=torch.long)


class SignMixupModel(nn.Module):
    """
    Base Model class
    """

    def __init__(
        self,
        encoder: Encoder,
        gloss_output_layer: nn.Module,
        decoder: Decoder,
        sgn_embed: SpatialEmbeddings,
        txt_embed: Embeddings,
        gloss_embed: TextEmbeddings,
        gls_vocab: GlossVocabulary,
        txt_vocab: TextVocabulary,
        do_recognition: bool = True,
        do_translation: bool = True,
    ):
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param sgn_embed: spatial feature frame embeddings
        :param txt_embed: spoken language word embedding
        :param gls_vocab: gls vocabulary
        :param txt_vocab: spoken language vocabulary
        :param do_recognition: flag to build the model with recognition output.
        :param do_translation: flag to build the model with translation decoder.
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.sgn_embed = sgn_embed
        self.gloss_embed = gloss_embed
        self.txt_embed = txt_embed

        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab

        self.txt_bos_index = self.txt_vocab.stoi[BOS_TOKEN]
        self.txt_pad_index = self.txt_vocab.stoi[PAD_TOKEN]
        self.txt_eos_index = self.txt_vocab.stoi[EOS_TOKEN]

        self.gloss_output_layer = gloss_output_layer
        self.do_recognition = do_recognition
        self.do_translation = do_translation

    # pylint: disable=arguments-differ
    def forward(
        self,
        sgn: Tensor,
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
        txt_input: Tensor,
        txt_mask: Tensor = None,
        forword_type = "sign",
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param sgn: source input
        :param sgn_mask: source mask
        :param sgn_lengths: length of source inputs
        :param txt_input: target input
        :param txt_mask: target mask
        :return: decoder outputs
        """

        sgn = self.embed(x=sgn, mask=sgn_mask, forword_type=forword_type)

        encoder_output, encoder_hidden = self.encode(
            sgn=sgn, sgn_mask=sgn_mask, sgn_length=sgn_lengths
        )

        if self.do_recognition:
            # Gloss Recognition Part
            # N x T x C
            gloss_scores = self.gloss_output_layer(encoder_output)
            # N x T x C
            gloss_probabilities = gloss_scores.log_softmax(2)
            # Turn it into T x N x C
            gloss_probabilities = gloss_probabilities.permute(1, 0, 2)
        else:
            gloss_probabilities = None

        if self.do_translation:
            unroll_steps = txt_input.size(1)
            decoder_outputs = self.decode(
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                sgn_mask=sgn_mask,
                txt_input=txt_input,
                unroll_steps=unroll_steps,
                txt_mask=txt_mask,
            )
        else:
            decoder_outputs = None

        return decoder_outputs, gloss_probabilities

    def embed(self, x, mask, forword_type):
        if forword_type == "sign":
            x = self.sgn_embed(x=x, mask=mask)
        elif forword_type == "gloss":
            x = self.gloss_embed(x=x, mask=mask)

        return x
    def encode(
        self, sgn: Tensor, sgn_mask: Tensor, sgn_length: Tensor
    ) -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param sgn:
        :param sgn_mask:
        :param sgn_length:
        :return: encoder outputs (output, hidden_concat)
        """

        # 需要更根据不同的情况调用不一样的
        return self.encoder(
            embed_src=sgn,
            src_length=sgn_length,
            mask=sgn_mask,
        )

    def decode(
        self,
        encoder_output: Tensor,
        encoder_hidden: Tensor,
        sgn_mask: Tensor,
        txt_input: Tensor,
        unroll_steps: int,
        decoder_hidden: Tensor = None,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param sgn_mask: sign sequence mask, 1 at valid tokens
        :param txt_input: spoken language sentence inputs
        :param unroll_steps: number of steps to unroll the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param txt_mask: mask for spoken language words
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=sgn_mask,
            trg_embed=self.txt_embed(x=txt_input, mask=txt_mask),
            trg_mask=txt_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
        )

    def get_loss_for_batch(
        self,
        batch: Batch,
        recognition_loss_function: nn.Module,
        translation_loss_function: nn.Module,
        recognition_loss_weight: float,
        translation_loss_weight: float,
    ) -> (Tensor, Tensor):
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param recognition_loss_function: Sign Language Recognition Loss Function (CTC)
        :param translation_loss_function: Sign Language Translation Loss Function (XEntropy)
        :param recognition_loss_weight: Weight for recognition loss
        :param translation_loss_weight: Weight for translation loss
        :return: recognition_loss: sum of losses over sequences in the batch
        :return: translation_loss: sum of losses over non-pad elements in the batch
        """
        # pylint: disable=unused-variable

        # Do a forward pass
        decoder_outputs, gloss_probabilities = self.forward(
            sgn=batch.sgn,
            sgn_mask=batch.sgn_mask,
            sgn_lengths=batch.sgn_lengths,
            txt_input=batch.txt_input,
            txt_mask=batch.txt_mask,
        )

        if self.do_recognition:
            assert gloss_probabilities is not None
            # Calculate Recognition Loss
            recognition_loss = (
                recognition_loss_function(
                    gloss_probabilities,
                    batch.gls,
                    batch.sgn_lengths.long(),
                    batch.gls_lengths.long(),
                )
                * recognition_loss_weight
            )
        else:
            recognition_loss = None

        if self.do_translation:
            assert decoder_outputs is not None
            word_outputs, _, _, _ = decoder_outputs
            # Calculate Translation Loss
            txt_log_probs = F.log_softmax(word_outputs, dim=-1)
            translation_loss = (
                translation_loss_function(txt_log_probs, batch.txt)
                * translation_loss_weight
            )
        else:
            translation_loss = None

        return recognition_loss, translation_loss

    def run_batch(
        self,
        batch: Batch,
        recognition_beam_size: int = 1,
        translation_beam_size: int = 1,
        translation_beam_alpha: float = -1,
        translation_max_output_length: int = 100,
    ) -> (np.array, np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param recognition_beam_size: size of the beam for CTC beam search
            if 1 use greedy
        :param translation_beam_size: size of the beam for translation beam search
            if 1 use greedy
        :param translation_beam_alpha: alpha value for beam search
        :param translation_max_output_length: maximum length of translation hypotheses
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """

        encoder_output, encoder_hidden = self.encode(
            sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_length=batch.sgn_lengths
        )

        if self.do_recognition:
            # Gloss Recognition Part
            # N x T x C
            gloss_scores = self.gloss_output_layer(encoder_output)
            # N x T x C
            gloss_probabilities = gloss_scores.log_softmax(2)
            # Turn it into T x N x C
            gloss_probabilities = gloss_probabilities.permute(1, 0, 2)
            gloss_probabilities = gloss_probabilities.cpu().detach().numpy()
            tf_gloss_probabilities = np.concatenate(
                (gloss_probabilities[:, :, 1:], gloss_probabilities[:, :, 0, None]),
                axis=-1,
            )

            assert recognition_beam_size > 0
            ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
                inputs=tf_gloss_probabilities,
                sequence_length=batch.sgn_lengths.cpu().detach().numpy(),
                beam_width=recognition_beam_size,
                top_paths=1,
            )
            ctc_decode = ctc_decode[0]
            # Create a decoded gloss list for each sample
            tmp_gloss_sequences = [[] for i in range(gloss_scores.shape[0])]
            for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
                tmp_gloss_sequences[dense_idx[0]].append(
                    ctc_decode.values[value_idx].numpy() + 1
                )
            decoded_gloss_sequences = []
            for seq_idx in range(0, len(tmp_gloss_sequences)):
                decoded_gloss_sequences.append(
                    [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
                )
        else:
            decoded_gloss_sequences = None

        if self.do_translation:
            # greedy decoding
            if translation_beam_size < 2:
                stacked_txt_output, stacked_attention_scores = greedy(
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    src_mask=batch.sgn_mask,
                    embed=self.txt_embed,
                    bos_index=self.txt_bos_index,
                    eos_index=self.txt_eos_index,
                    decoder=self.decoder,
                    max_output_length=translation_max_output_length,
                )
                # batch, time, max_sgn_length
            else:  # beam size
                stacked_txt_output, stacked_attention_scores = beam_search(
                    size=translation_beam_size,
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    src_mask=batch.sgn_mask,
                    embed=self.txt_embed,
                    max_output_length=translation_max_output_length,
                    alpha=translation_beam_alpha,
                    eos_index=self.txt_eos_index,
                    pad_index=self.txt_pad_index,
                    bos_index=self.txt_bos_index,
                    decoder=self.decoder,
                )
        else:
            stacked_txt_output = stacked_attention_scores = None

        return decoded_gloss_sequences, stacked_txt_output, stacked_attention_scores

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return (
            "%s(\n"
            "\tencoder=%s,\n"
            "\tdecoder=%s,\n"
            "\tsgn_embed=%s,\n"
            "\ttxt_embed=%s)"
            % (
                self.__class__.__name__,
                self.encoder,
                self.decoder,
                self.sgn_embed,
                self.txt_embed,
            )
        )


#TODO Embeddings

import math
import torch

from torch import nn, Tensor
import torch.nn.functional as F
from signjoey.helpers import freeze_params

def get_activation(activation_type):
    if activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "relu6":
        return nn.ReLU6()
    elif activation_type == "prelu":
        return nn.PReLU()
    elif activation_type == "selu":
        return nn.SELU()
    elif activation_type == "celu":
        return nn.CELU()
    elif activation_type == "gelu":
        return nn.GELU()
    elif activation_type == "sigmoid":
        return nn.Sigmoid()
    elif activation_type == "softplus":
        return nn.Softplus()
    elif activation_type == "softshrink":
        return nn.Softshrink()
    elif activation_type == "softsign":
        return nn.Softsign()
    elif activation_type == "tanh":
        return nn.Tanh()
    elif activation_type == "tanhshrink":
        return nn.Tanhshrink()
    else:
        raise ValueError("Unknown activation type {}".format(activation_type))


class MaskedNorm(nn.Module):
    """
        Original Code from:
        https://discuss.pytorch.org/t/batchnorm-for-different-sized-samples-in-batch/44251/8
    """

    def __init__(self, norm_type, num_groups, num_features):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == "batch":
            self.norm = nn.BatchNorm1d(num_features=num_features)
        elif self.norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=num_features)
        else:
            raise ValueError("Unsupported Normalization Layer")

        self.num_features = num_features

    def forward(self, x: Tensor, mask: Tensor):
        if self.training:
            reshaped = x.reshape([-1, self.num_features])
            reshaped_mask = mask.reshape([-1, 1]) > 0
            selected = torch.masked_select(reshaped, reshaped_mask).reshape(
                [-1, self.num_features]
            )
            batch_normed = self.norm(selected)
            scattered = reshaped.masked_scatter(reshaped_mask, batch_normed)
            return scattered.reshape([x.shape[0], -1, self.num_features])
        else:
            reshaped = x.reshape([-1, self.num_features])
            batched_normed = self.norm(reshaped)
            return batched_normed.reshape([x.shape[0], -1, self.num_features])






class BERTPAIRSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length):
        nn.Module.__init__(self)
        self.bert = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased",
            num_labels=2)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    def forward(self, inputs):
        x = self.bert(inputs['word'], token_type_ids=inputs['seg'], attention_mask=inputs['mask'])[0]
        return x

    def tokenize(self, raw_tokens):
        # token -> index
        # tokens = ['[CLS]']
        tokens = []
        for token in raw_tokens:
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.tokenizer.convert_tokens_to_ids("<pad>"))#该用什么填充呢？
        indexed_tokens = indexed_tokens[:self.max_length]

        return indexed_tokens
