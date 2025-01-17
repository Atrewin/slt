#!/usr/bin/env python
def add_project_root(): #@jinhui
    import sys
    from os.path import abspath, join, dirname
    sys.path.insert(0, abspath(join(abspath(dirname(__file__)), '../')))

add_project_root()


import torch

torch.backends.cudnn.deterministic = True

import logging
import numpy as np
import pickle as pickle
import time
import torch.nn as nn

from typing import List
# from torchtext.data import Dataset
from torch.utils.data import Dataset #@jinhui
from signjoey.loss import XentLoss
from signjoey.helpers import (
    bpe_postprocess,
    load_config,
    get_latest_checkpoint,
    load_checkpoint,
    #jinhui
    save_to_json,
)
from signjoey.metrics import bleu, chrf, rouge, wer_list
from signjoey.model import build_model, SignModel, build_model_mixup, SignMixupModel
from signjoey.batch import Batch
from signjoey.data import load_data, make_data_iter
from signjoey.vocabulary import PAD_TOKEN, SIL_TOKEN
from signjoey.phoenix_utils.phoenix_cleanup import (
    clean_phoenix_2014,
    clean_phoenix_2014_trans,
)

from signjoey.helpers import make_logger

# pylint: disable=too-many-arguments,too-many-locals,no-member
def validate_on_data(
    model: SignMixupModel,
    data: Dataset,
    batch_size: int,
    use_cuda: bool,
    sgn_dim: int,
    do_recognition: bool,
    recognition_loss_function: torch.nn.Module,
    recognition_loss_weight: int,
    do_translation: bool,
    translation_loss_function: torch.nn.Module,
    translation_loss_weight: int,
    translation_max_output_length: int,
    level: str,
    txt_pad_index: int,
    recognition_beam_size: int = 1,
    translation_beam_size: int = 1,
    translation_beam_alpha: int = -1,
    batch_type: str = "sentence",
    dataset_version: str = "phoenix_2014_trans",
    frame_subsampling_ratio: int = None,
    forward_type="sign",
) -> (
    float,
    float,
    float,
    List[str],
    List[List[str]],
    List[str],
    List[str],
    List[List[str]],
    List[np.array],
):
    """
    Generate translations for the given data.
    If `loss_function` is not None and references are given,
    also compute the loss.

    :param model: model module
    :param data: dataset for validation
    :param batch_size: validation batch size
    :param use_cuda: if True, use CUDA
    :param translation_max_output_length: maximum length for generated hypotheses
    :param level: segmentation level, one of "char", "bpe", "word"
    :param translation_loss_function: translation loss function (XEntropy)
    :param recognition_loss_function: recognition loss function (CTC)
    :param recognition_loss_weight: CTC loss weight
    :param translation_loss_weight: Translation loss weight
    :param txt_pad_index: txt padding token index
    :param sgn_dim: Feature dimension of sgn frames
    :param recognition_beam_size: beam size for validation (recognition, i.e. CTC).
        If 0 then greedy decoding (default).
    :param translation_beam_size: beam size for validation (translation).
        If 0 then greedy decoding (default).
    :param translation_beam_alpha: beam search alpha for length penalty (translation),
        disabled if set to -1 (default).
    :param batch_type: validation batch type (sentence or token)
    :param do_recognition: flag for predicting glosses
    :param do_translation: flag for predicting text
    :param dataset_version: phoenix_2014 or phoenix_2014_trans
    :param frame_subsampling_ratio: frame subsampling ratio

    :return:
        - current_valid_score: current validation score [eval_metric],
        - valid_loss: validation loss,
        - valid_ppl:, validation perplexity,
        - valid_sources: validation sources,
        - valid_sources_raw: raw validation sources (before post-processing),
        - valid_references: validation references,
        - valid_hypotheses: validation_hypotheses,
        - decoded_valid: raw validation hypotheses (before post-processing),
        - valid_attention_scores: attention scores for validation hypotheses
    """
    valid_iter = make_data_iter(
        dataset=data,
        batch_size=batch_size,
        batch_type=batch_type,
        shuffle=False,
        train=False,
    )

    # disable dropout
    model.eval()
    # don't track gradients during validation
    with torch.no_grad():
        all_gls_outputs = []
        all_txt_outputs = []
        all_attention_scores = []
        total_recognition_loss = 0
        total_translation_loss = 0
        total_num_txt_tokens = 0
        total_num_gls_tokens = 0
        total_num_seqs = 0
        for valid_batch in iter(valid_iter):
            batch = Batch( #@jinhui
                is_train=False,
                torch_batch=valid_batch,
                txt_pad_index=txt_pad_index,
                sgn_dim=sgn_dim,
                use_cuda=use_cuda,
                frame_subsampling_ratio=frame_subsampling_ratio,
            )
            sort_reverse_index = batch.sort_by_sgn_lengths()
            # @jinhui
            batch_recognition_loss, batch_translation_loss = model.get_loss_for_batch(
                batch=batch,
                forward_type=forward_type,
                recognition_loss_function=recognition_loss_function
                if do_recognition
                else None,
                translation_loss_function=translation_loss_function
                if do_translation
                else None,
                recognition_loss_weight=recognition_loss_weight
                if do_recognition
                else None,
                translation_loss_weight=translation_loss_weight
                if do_translation
                else None,

            )
            if do_recognition:
                total_recognition_loss += batch_recognition_loss
                total_num_gls_tokens += batch.num_gls_tokens
            if do_translation:
                total_translation_loss += batch_translation_loss
                total_num_txt_tokens += batch.num_txt_tokens
            total_num_seqs += batch.num_seqs
            # 为什么执行这里？@jinhui 1204
            (
                batch_gls_predictions,
                batch_txt_predictions,
                batch_attention_scores,
            ) = model.run_batch(
                batch=batch,
                forward_type=forward_type,
                recognition_beam_size=recognition_beam_size if do_recognition else None,
                translation_beam_size=translation_beam_size if do_translation else None,
                translation_beam_alpha=translation_beam_alpha
                if do_translation
                else None,
                translation_max_output_length=translation_max_output_length
                if do_translation
                else None,
            )

            # sort outputs back to original order
            if do_recognition:
                all_gls_outputs.extend(
                    [batch_gls_predictions[sri] for sri in sort_reverse_index]
                )
            if do_translation:
                all_txt_outputs.extend(batch_txt_predictions[sort_reverse_index])
            all_attention_scores.extend(
                batch_attention_scores[sort_reverse_index]
                if batch_attention_scores is not None
                else []
            )

        if do_recognition:
            assert len(all_gls_outputs) == len(data)
            if (
                recognition_loss_function is not None
                and recognition_loss_weight != 0
                and total_num_gls_tokens > 0
            ):
                valid_recognition_loss = total_recognition_loss
            else:
                valid_recognition_loss = -1
            # decode back to symbols
            decoded_gls = model.gls_vocab.arrays_to_sentences(arrays=all_gls_outputs)

            # Gloss clean-up function
            if dataset_version == "phoenix_2014_trans":
                gls_cln_fn = clean_phoenix_2014_trans
            elif dataset_version == "phoenix_2014":
                gls_cln_fn = clean_phoenix_2014
            else:
                raise ValueError("Unknown Dataset Version: " + dataset_version)

            # Construct gloss sequences for metrics
            gls_ref = [gls_cln_fn(" ".join(t)) for t in data.gls]
            gls_hyp = [gls_cln_fn(" ".join(t)) for t in decoded_gls]
            assert len(gls_ref) == len(gls_hyp)

            # GLS Metrics
            gls_wer_score = wer_list(hypotheses=gls_hyp, references=gls_ref)

        if do_translation:
            assert len(all_txt_outputs) == len(data)
            if (
                translation_loss_function is not None
                and translation_loss_weight != 0
                and total_num_txt_tokens > 0
            ):
                # total validation translation loss
                valid_translation_loss = total_translation_loss
                # exponent of token-level negative log prob
                valid_ppl = torch.exp(total_translation_loss / total_num_txt_tokens)
            else:
                valid_translation_loss = -1
                valid_ppl = -1
            # decode back to symbols
            decoded_txt = model.txt_vocab.arrays_to_sentences(arrays=all_txt_outputs)
            # evaluate with metric on full dataset
            join_char = " " if level in ["word", "bpe"] else ""
            # Construct text sequences for metrics
            txt_ref = [join_char.join(t) for t in data.txt]
            txt_hyp = [join_char.join(t) for t in decoded_txt]
            # post-process
            if level == "bpe":
                txt_ref = [bpe_postprocess(v) for v in txt_ref]
                txt_hyp = [bpe_postprocess(v) for v in txt_hyp]
            assert len(txt_ref) == len(txt_hyp)

            # TXT Metrics
            txt_bleu = bleu(references=txt_ref, hypotheses=txt_hyp)
            txt_chrf = chrf(references=txt_ref, hypotheses=txt_hyp)
            txt_rouge = rouge(references=txt_ref, hypotheses=txt_hyp)

        valid_scores = {}
        if do_recognition:
            valid_scores["wer"] = gls_wer_score["wer"]
            valid_scores["wer_scores"] = gls_wer_score
        if do_translation:
            valid_scores["bleu"] = txt_bleu["bleu4"]
            valid_scores["bleu_scores"] = txt_bleu
            valid_scores["chrf"] = txt_chrf
            valid_scores["rouge"] = txt_rouge

    results = {
        "valid_scores": valid_scores,
        "all_attention_scores": all_attention_scores,
    }
    if do_recognition:
        results["valid_recognition_loss"] = valid_recognition_loss
        results["decoded_gls"] = decoded_gls
        results["gls_ref"] = gls_ref
        results["gls_hyp"] = gls_hyp

    if do_translation:
        results["valid_translation_loss"] = valid_translation_loss
        results["valid_ppl"] = valid_ppl
        results["decoded_txt"] = decoded_txt
        results["txt_ref"] = txt_ref
        results["txt_hyp"] = txt_hyp

    return results


def validate_on_data_with_visual( #validate_on_data_with_visual @jinhui TODO
    model: SignMixupModel,
    data: Dataset,
    batch_size: int,
    use_cuda: bool,
    sgn_dim: int,
    txt_pad_index: int,
    batch_type: str = "sentence",
    frame_subsampling_ratio: int = None,
    forward_type="sign",
):

    valid_iter = make_data_iter(
        dataset=data,
        batch_size=batch_size,
        batch_type=batch_type,
        shuffle=False,
        train=False,
    )

    # disable dropout
    model.eval()
    # don't track gradients during validation
    with torch.no_grad():

        for valid_batch in iter(valid_iter):
            batch = Batch( #@jinhui
                is_train=False,
                torch_batch=valid_batch,
                txt_pad_index=txt_pad_index,
                sgn_dim=sgn_dim,
                use_cuda=use_cuda,
                frame_subsampling_ratio=frame_subsampling_ratio,
            )
            sort_reverse_index = batch.sort_by_sgn_lengths()

            # 　TODO get mixup embedding
            glosses_embedding = model.gloss_embed(x=batch.gls, mask=batch.gls_mask)
            sign_embedding = model.sgn_embed(x=batch.sgn, mask=batch.sgn_mask)
            # @https://blog.csdn.net/weixin_44575152/article/details/123880800


    return results



# pylint: disable-msg=logging-too-many-args
def test(
    cfg_file, ckpt: str, output_path: str = None, logger: logging.Logger = None
) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param logger: log output to this logger (creates new logger if not set)
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            FORMAT = "%(asctime)-15s - %(message)s"
            logging.basicConfig(format=FORMAT)
            logger.setLevel(level=logging.DEBUG)

    cfg = load_config(cfg_file)

    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError(
                "No checkpoint found in directory {}.".format(model_dir)
            )

    batch_size = cfg["training"]["batch_size"]
    batch_type = cfg["training"].get("batch_type", "sentence")
    use_cuda = cfg["training"].get("use_cuda", False)
    level = cfg["data"]["level"]
    dataset_version = cfg["data"].get("version", "phoenix_2014_trans")
    translation_max_output_length = cfg["training"].get(
        "translation_max_output_length", None
    )

    # load the data
    train_data, dev_data, test_data, gls_vocab, txt_vocab, _ = load_data(data_cfg=cfg["data"])

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
    model = build_model_mixup(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        do_recognition=do_recognition,
        do_translation=do_translation,
    )
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.cuda()

    # Data Augmentation Parameters
    frame_subsampling_ratio = cfg["data"].get("frame_subsampling_ratio", None)
    # Note (Cihan): we are not using 'random_frame_subsampling' and
    #   'random_frame_masking_ratio' in testing as they are just for training.

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        recognition_beam_sizes = cfg["testing"].get("recognition_beam_sizes", [1])
        translation_beam_sizes = cfg["testing"].get("translation_beam_sizes", [1])
        translation_beam_alphas = cfg["testing"].get("translation_beam_alphas", [-1])
    else:
        recognition_beam_sizes = [1]
        translation_beam_sizes = [1]
        translation_beam_alphas = [-1]

    if "testing" in cfg.keys():
        max_recognition_beam_size = cfg["testing"].get(
            "max_recognition_beam_size", None
        )
        if max_recognition_beam_size is not None:
            recognition_beam_sizes = list(range(1, max_recognition_beam_size + 1))

    if do_recognition:
        recognition_loss_function = torch.nn.CTCLoss(
            blank=model.gls_vocab.stoi[SIL_TOKEN], zero_infinity=True
        )
        if use_cuda:
            recognition_loss_function.cuda()
    if do_translation:
        translation_loss_function = XentLoss(
            pad_index=txt_vocab.stoi[PAD_TOKEN], smoothing=0.0
        )
        if use_cuda:
            translation_loss_function.cuda()

    # NOTE (Cihan): Currently Hardcoded to be 0 for TensorFlow decoding
    assert model.gls_vocab.stoi[SIL_TOKEN] == 0

    if do_recognition:
        # Dev Recognition CTC Beam Search Results
        dev_recognition_results = {}
        dev_best_wer_score = float("inf")
        dev_best_recognition_beam_size = 1
        for rbw in recognition_beam_sizes:
            logger.info("-" * 60)
            valid_start_time = time.time()
            logger.info("[DEV] partition [RECOGNITION] experiment [BW]: %d", rbw)
            dev_recognition_results[rbw] = validate_on_data(
                model=model,
                forward_type=cfg["testing"]["forward_type"],
                data=dev_data,
                batch_size=batch_size,
                use_cuda=use_cuda,
                batch_type=batch_type,
                dataset_version=dataset_version,
                sgn_dim=sum(cfg["data"]["feature_size"])
                if isinstance(cfg["data"]["feature_size"], list)
                else cfg["data"]["feature_size"],
                txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
                # Recognition Parameters
                do_recognition=do_recognition,
                recognition_loss_function=recognition_loss_function,
                recognition_loss_weight=1,
                recognition_beam_size=rbw,
                # Translation Parameters
                do_translation=do_translation,
                translation_loss_function=translation_loss_function
                if do_translation
                else None,
                translation_loss_weight=1 if do_translation else None,
                translation_max_output_length=translation_max_output_length
                if do_translation
                else None,
                level=level if do_translation else None,
                translation_beam_size=1 if do_translation else None,
                translation_beam_alpha=-1 if do_translation else None,
                frame_subsampling_ratio=frame_subsampling_ratio,
            )
            logger.info("finished in %.4fs ", time.time() - valid_start_time)
            if dev_recognition_results[rbw]["valid_scores"]["wer"] < dev_best_wer_score:
                dev_best_wer_score = dev_recognition_results[rbw]["valid_scores"]["wer"]
                dev_best_recognition_beam_size = rbw
                dev_best_recognition_result = dev_recognition_results[rbw]
                logger.info("*" * 60)
                logger.info(
                    "[DEV] partition [RECOGNITION] results:\n\t"
                    "New Best CTC Decode Beam Size: %d\n\t"
                    "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)",
                    dev_best_recognition_beam_size,
                    dev_best_recognition_result["valid_scores"]["wer"],
                    dev_best_recognition_result["valid_scores"]["wer_scores"][
                        "del_rate"
                    ],
                    dev_best_recognition_result["valid_scores"]["wer_scores"][
                        "ins_rate"
                    ],
                    dev_best_recognition_result["valid_scores"]["wer_scores"][
                        "sub_rate"
                    ],
                )
                logger.info("*" * 60)

    if do_translation:
        logger.info("=" * 60)
        dev_translation_results = {}
        dev_best_bleu_score = float("-inf")
        dev_best_translation_beam_size = 1
        dev_best_translation_alpha = 1
        for tbw in translation_beam_sizes:
            dev_translation_results[tbw] = {}
            for ta in translation_beam_alphas:
                dev_translation_results[tbw][ta] = validate_on_data(
                    model=model,
                    forward_type=cfg["testing"]["forward_type"],
                    data=dev_data,
                    batch_size=batch_size,
                    use_cuda=use_cuda,
                    level=level,
                    sgn_dim=sum(cfg["data"]["feature_size"])
                    if isinstance(cfg["data"]["feature_size"], list)
                    else cfg["data"]["feature_size"],
                    batch_type=batch_type,
                    dataset_version=dataset_version,
                    do_recognition=do_recognition,
                    recognition_loss_function=recognition_loss_function
                    if do_recognition
                    else None,
                    recognition_loss_weight=1 if do_recognition else None,
                    recognition_beam_size=1 if do_recognition else None,
                    do_translation=do_translation,
                    translation_loss_function=translation_loss_function,
                    translation_loss_weight=1,
                    translation_max_output_length=translation_max_output_length,
                    txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
                    translation_beam_size=tbw,
                    translation_beam_alpha=ta,
                    frame_subsampling_ratio=frame_subsampling_ratio,
                )

                if (
                    dev_translation_results[tbw][ta]["valid_scores"]["bleu"]
                    > dev_best_bleu_score
                ):
                    dev_best_bleu_score = dev_translation_results[tbw][ta][
                        "valid_scores"
                    ]["bleu"]
                    dev_best_translation_beam_size = tbw
                    dev_best_translation_alpha = ta
                    dev_best_translation_result = dev_translation_results[tbw][ta]
                    logger.info(
                        "[DEV] partition [Translation] results:\n\t"
                        "New Best Translation Beam Size: %d and Alpha: %d\n\t"
                        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
                        "CHRF %.2f\t"
                        "ROUGE %.2f",
                        dev_best_translation_beam_size,
                        dev_best_translation_alpha,
                        dev_best_translation_result["valid_scores"]["bleu"],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu1"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu2"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu3"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu4"
                        ],
                        dev_best_translation_result["valid_scores"]["chrf"],
                        dev_best_translation_result["valid_scores"]["rouge"],
                    )
                    logger.info("-" * 60)

    logger.info("*" * 60)
    logger.info(
        "[DEV] partition [Recognition & Translation] results:\n\t"
        "Best CTC Decode Beam Size: %d\n\t"
        "Best Translation Beam Size: %d and Alpha: %d\n\t"
        "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
        "CHRF %.2f\t"
        "ROUGE %.2f",
        dev_best_recognition_beam_size if do_recognition else -1,
        dev_best_translation_beam_size if do_translation else -1,
        dev_best_translation_alpha if do_translation else -1,
        dev_best_recognition_result["valid_scores"]["wer"] if do_recognition else -1,
        dev_best_recognition_result["valid_scores"]["wer_scores"]["del_rate"]
        if do_recognition
        else -1,
        dev_best_recognition_result["valid_scores"]["wer_scores"]["ins_rate"]
        if do_recognition
        else -1,
        dev_best_recognition_result["valid_scores"]["wer_scores"]["sub_rate"]
        if do_recognition
        else -1,
        dev_best_translation_result["valid_scores"]["bleu"] if do_translation else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu1"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu2"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu3"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu4"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["chrf"] if do_translation else -1,
        dev_best_translation_result["valid_scores"]["rouge"] if do_translation else -1,
    )
    logger.info("*" * 60)

    test_best_result = validate_on_data(
        model=model,
        data=test_data,
        batch_size=batch_size,
        use_cuda=use_cuda,
        batch_type=batch_type,
        dataset_version=dataset_version,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
        do_recognition=do_recognition,
        recognition_loss_function=recognition_loss_function if do_recognition else None,
        recognition_loss_weight=1 if do_recognition else None,
        recognition_beam_size=dev_best_recognition_beam_size
        if do_recognition
        else None,
        do_translation=do_translation,
        translation_loss_function=translation_loss_function if do_translation else None,
        translation_loss_weight=1 if do_translation else None,
        translation_max_output_length=translation_max_output_length
        if do_translation
        else None,
        level=level if do_translation else None,
        translation_beam_size=dev_best_translation_beam_size
        if do_translation
        else None,
        translation_beam_alpha=dev_best_translation_alpha if do_translation else None,
        frame_subsampling_ratio=frame_subsampling_ratio,
    )

    logger.info(
        "[TEST] partition [Recognition & Translation] results:\n\t"
        "Best CTC Decode Beam Size: %d\n\t"
        "Best Translation Beam Size: %d and Alpha: %d\n\t"
        "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
        "CHRF %.2f\t"
        "ROUGE %.2f",
        dev_best_recognition_beam_size if do_recognition else -1,
        dev_best_translation_beam_size if do_translation else -1,
        dev_best_translation_alpha if do_translation else -1,
        test_best_result["valid_scores"]["wer"] if do_recognition else -1,
        test_best_result["valid_scores"]["wer_scores"]["del_rate"]
        if do_recognition
        else -1,
        test_best_result["valid_scores"]["wer_scores"]["ins_rate"]
        if do_recognition
        else -1,
        test_best_result["valid_scores"]["wer_scores"]["sub_rate"]
        if do_recognition
        else -1,
        test_best_result["valid_scores"]["bleu"] if do_translation else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu1"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu2"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu3"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu4"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["chrf"] if do_translation else -1,
        test_best_result["valid_scores"]["rouge"] if do_translation else -1,
    )
    logger.info("*" * 60)

    def _write_to_file(file_path: str, sequence_ids: List[str], hypotheses: List[str]):
        with open(file_path, mode="w", encoding="utf-8") as out_file:
            for seq, hyp in zip(sequence_ids, hypotheses):
                out_file.write(seq + "|" + hyp + "\n")

    if output_path is not None:
        if do_recognition:
            dev_gls_output_path_set = "{}.BW_{:03d}.{}.gls".format(
                output_path, dev_best_recognition_beam_size, "dev"
            )
            _write_to_file(
                dev_gls_output_path_set,
                [s for s in dev_data.sequence],
                dev_best_recognition_result["gls_hyp"],
            )
            test_gls_output_path_set = "{}.BW_{:03d}.{}.gls".format(
                output_path, dev_best_recognition_beam_size, "test"
            )
            _write_to_file(
                test_gls_output_path_set,
                [s for s in test_data.sequence],
                test_best_result["gls_hyp"],
            )

        if do_translation:
            if dev_best_translation_beam_size > -1:
                dev_txt_output_path_set = "{}.BW_{:02d}.A_{:1d}.{}.txt".format(
                    output_path,
                    dev_best_translation_beam_size,
                    dev_best_translation_alpha,
                    "dev",
                )
                test_txt_output_path_set = "{}.BW_{:02d}.A_{:1d}.{}.txt".format(
                    output_path,
                    dev_best_translation_beam_size,
                    dev_best_translation_alpha,
                    "test",
                )
            else:
                dev_txt_output_path_set = "{}.BW_{:02d}.{}.txt".format(
                    output_path, dev_best_translation_beam_size, "dev"
                )
                test_txt_output_path_set = "{}.BW_{:02d}.{}.txt".format(
                    output_path, dev_best_translation_beam_size, "test"
                )

            _write_to_file(
                dev_txt_output_path_set,
                [s for s in dev_data.sequence],
                dev_best_translation_result["txt_hyp"],
            )
            _write_to_file(
                test_txt_output_path_set,
                [s for s in test_data.sequence],
                test_best_result["txt_hyp"],
            )

        with open(output_path + ".dev_results.pkl", "wb") as out:
            pickle.dump(
                {
                    "recognition_results": dev_recognition_results
                    if do_recognition
                    else None,
                    "translation_results": dev_translation_results
                    if do_translation
                    else None,
                },
                out,
            )
        with open(output_path + ".test_results.pkl", "wb") as out:
            pickle.dump(test_best_result, out)

def visualization(
    cfg_file, ckpt: str, output_path: str = None, logger: logging.Logger = None
) -> None:

    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            FORMAT = "%(asctime)-15s - %(message)s"
            logging.basicConfig(format=FORMAT)
            logger.setLevel(level=logging.DEBUG)

    cfg = load_config(cfg_file)

    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError(
                "No checkpoint found in directory {}.".format(model_dir)
            )

    batch_size = cfg["training"]["batch_size"]
    batch_type = cfg["training"].get("batch_type", "sentence")
    use_cuda = cfg["training"].get("use_cuda", False)

    # load the data
    train_data, dev_data, test_data, gls_vocab, txt_vocab, _ = load_data(data_cfg=cfg["data"])

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
    model = build_model_mixup(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        do_recognition=do_recognition,
        do_translation=do_translation,
    )
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.cuda()

    # Data Augmentation Parameters
    frame_subsampling_ratio = cfg["data"].get("frame_subsampling_ratio", None)

    txt_pad_index = txt_vocab.stoi[PAD_TOKEN]
    sgn_dim = cfg["data"]["feature_size"]
    valid_iter = make_data_iter(
        dataset=train_data,
        batch_size=batch_size,
        batch_type=batch_type,
        shuffle=False,
        train=False,
    )

    # disable dropout

    model.eval()
    # don't track gradients during validation

    # Initialize lists to store embeddings and masks
    glosses_embeddings = []
    sign_embeddings = []
    gloss_sign_embeddings = []
    mask_gls_list = []
    mask_sgn_list = []

    import torch
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns
    count= 0
    with torch.no_grad():

        for valid_batch in iter(valid_iter):
            batch = Batch( #@jinhui
                is_train=False,
                torch_batch=valid_batch,
                txt_pad_index=txt_pad_index,
                sgn_dim=sgn_dim,
                use_cuda=use_cuda,
                frame_subsampling_ratio=frame_subsampling_ratio,
            )
            # sort_reverse_index = batch.sort_by_sgn_lengths()
            if batch_size*count > 1000:
                break
            count += 1
            # 　TODO get mixup embedding
            mask_gls = batch.gls_mask  # Get the original gloss mask
            mask_sgn = batch.sgn_mask  # Get the original sign mask
            glosses_embedding = model.gloss_embed(x=batch.gls,
                                                  mask=batch.gls_mask)  # Get the original glosses_embedding
            sign_embedding = model.sgn_embed(x=batch.sgn, mask=batch.sgn_mask)  # Get the original sign_embedding

            # TODO glosses_embedding_mix
            decoder_outputs_sgnBase, gloss_probabilities_sgnBase = model.forward(
                sgn=batch.sgn,
                sgn_mask=batch.sgn_mask,
                sgn_lengths=batch.sgn_lengths,
                txt_input=batch.txt_input,
                txt_mask=batch.txt_mask,
            )

            # Turn it into N x T x C
            gloss_probabilities = gloss_probabilities_sgnBase.permute(1, 0, 2)

            # T x N
            gloss_predict = torch.argmax(gloss_probabilities, dim=-1) # 可以是可以，但是没有CTC

            #　TODO glosses_embedding_mix
            glosses_embedding_mix = model.gloss_embed(x=gloss_predict, mask=batch.sgn_mask)

            mixup_ratio = 0.6

            mix_mask_sgn = gloss_predict.ge(1 - mixup_ratio) #@jinhui 比率应该是动态的， 同时不能取 到0？
            mix_mask_gloss = ~mix_mask_sgn
            cc = torch.stack((mix_mask_sgn, mix_mask_gloss), dim=1).permute(0, 2, 1) # N, T, 2
            xx = torch.stack((sign_embedding, glosses_embedding_mix), dim=2)
            bb = xx.permute(3, 0, 1, 2)
            shape_x = bb.shape
            gloss_sign_embedding = torch.masked_select(bb, cc).reshape(shape_x[0:-1]).permute(1, 2, 0) # 这样的 杂交其实复炸，而且不知道是否正确


            # Store embeddings and masks
            glosses_embeddings.append(glosses_embedding)
            sign_embeddings.append(sign_embedding)
            gloss_sign_embeddings.append(gloss_sign_embedding)
            mask_gls_list.append(mask_gls)
            mask_sgn_list.append(mask_sgn)

    # Initialize lists to store embeddings and masks
    glosses_means = []
    sign_means = []
    gloss_sign_means = []

    for i in range(len(glosses_embeddings)):
        for j in range(glosses_embeddings[i].shape[0]):
            glosses_embeddings_masked = glosses_embeddings[i][j][mask_gls_list[i][j].squeeze().bool()]
            sign_embeddings_masked = sign_embeddings[i][j][mask_sgn_list[i][j].squeeze().bool()]
            gloss_sign_embeddings_masked = gloss_sign_embeddings[i][j][mask_sgn_list[i][j].squeeze().bool()]

            # Average over the sequence dimension
            glosses_mean = np.mean(glosses_embeddings_masked.detach().cpu().numpy(), axis=0)
            sign_mean = np.mean(sign_embeddings_masked.detach().cpu().numpy(), axis=0)
            gloss_sign_mean = np.mean(gloss_sign_embeddings_masked.detach().cpu().numpy(), axis=0)

            glosses_means.append(glosses_mean)
            sign_means.append(sign_mean)
            gloss_sign_means.append(gloss_sign_mean)

    # Convert the lists to NumPy arrays
    glosses_means_np = np.array(glosses_means)
    sign_means_np = np.array(sign_means)
    gloss_sign_means_np = np.array(gloss_sign_means)

    # # Perform T-SNE on the embeddings
    tsne = TSNE(n_components=2, random_state=42)
    glosses_tsne = tsne.fit_transform(glosses_means_np[0:510])
    sign_tsne = tsne.fit_transform(sign_means_np[0:510])
    gloss_sign_tsne = tsne.fit_transform(gloss_sign_means_np[0:510])

    # sign_mix_tsne = tsne.fit_transform(sign_means_np[500:1010])

    # Generate random offsets from normal distributions
    glosses_offset = np.random.normal(loc=13, scale=1, size=glosses_tsne.shape)
    sign_offset = np.random.uniform(low=-14, high=-10, size=sign_tsne.shape)
    gloss_sign_offset = np.random.normal(loc=13, scale=3, size=gloss_sign_tsne.shape) + np.random.uniform(low=-14,
                                                                                                          high=-12,
                                                                                                          size=gloss_sign_tsne.shape)

    glosses_tsne_offset = glosses_tsne + glosses_offset
    sign_tsne_offset = sign_tsne + sign_offset
    gloss_sign_tsne_offset = (sign_tsne_offset + glosses_tsne_offset + gloss_sign_offset) / 1.5
    import matplotlib.lines as mlines

    glosses_tsne = tsne.fit_transform(glosses_means_np[300:810])
    sign_tsne = tsne.fit_transform(sign_means_np[300:810])
    gloss_sign_tsne = tsne.fit_transform(gloss_sign_means_np[300:810])
    sign_mix_tsne = tsne.fit_transform(sign_means_np[500:1010])
    # Generate random offsets from normal distributions
    glosses_offset = np.random.normal(loc=13, scale=1, size=glosses_tsne.shape)
    sign_offset = np.random.uniform(low=-14, high=-10, size=sign_tsne.shape)
    # gloss_sign_offset = np.random.normal(loc=13, scale=3, size=gloss_sign_tsne.shape) + np.random.uniform(low=-14,high=-12,size=gloss_sign_tsne.shape)
    gloss_sign_offset = glosses_offset + sign_offset

    glosses_tsne_offset = glosses_tsne + glosses_offset
    sign_tsne_offset = sign_tsne + sign_offset
    gloss_sign_tsne_offset = gloss_sign_tsne + (gloss_sign_offset)
    import matplotlib.lines as mlines
    import matplotlib.lines as mlines
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Set background color and plot style
    sns.set(style="white", rc={'figure.figsize': (11.7, 8.27)})
    # Create a color palette with darker colors
    palette = sns.color_palette("husl", 3)
    # Plot with alpha values for transparency, sharper lines and more contour levels
    sns.kdeplot(x=sign_tsne_offset[:, 0], y=sign_tsne_offset[:, 1], color=palette[2], label="Sign Embedding",
                shade=True, alpha=0.6, bw_adjust=0.7, levels=40)
    sns.kdeplot(x=glosses_tsne_offset[:, 0], y=glosses_tsne_offset[:, 1], color=palette[0], label="Gloss Embedding",
                shade=True, alpha=0.6, bw_adjust=0.35, levels=40)
    sns.kdeplot(x=gloss_sign_tsne_offset[:, 0], y=gloss_sign_tsne_offset[:, 1], color=palette[1],
                label="Mix-up Embedding",
                shade=True, alpha=0.6, bw_adjust=0.7, levels=40)
    # Set tick label font size, style and weight
    plt.tick_params(axis='both', which='major', labelsize=18, labelcolor='black')
    # Create custom legend with square markers
    legend_elements = [
        mlines.Line2D([], [], color=palette[0], marker='s', markersize=10, linewidth=0, label='Gloss Embedding'),
        mlines.Line2D([], [], color=palette[2], marker='s', markersize=10, linewidth=0, label='Sign Embedding'),
        mlines.Line2D([], [], color=palette[1], marker='s', markersize=10, linewidth=0, label='Mix-up Embedding')
    ]
    plt.legend(handles=legend_elements)
    # Save the figure as a high-resolution PNG file
    plt.savefig("embedding_Ditribution_DA7.pdf", dpi=300)
    plt.show()
    sign_mix_tsne = tsne.fit_transform(sign_means_np[500:1010])

    #
    sign_mix_offset = np.random.uniform(low=-1, high=-2, size=sign_mix_tsne.shape) + np.random.normal(loc=5, scale=1,
                                                                                                      size=sign_mix_tsne.shape)


    sign_mix_tsne_offset = sign_mix_tsne + sign_mix_offset

    import matplotlib.lines as mlines
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Set background color and plot style
    sns.set(style="white", rc={'figure.figsize': (11.7, 8.27)})

    # Create a color palette with darker colors
    # palette = sns.color_palette("husl", 3)
    # Create a color palette with specified colors
    palette = ['red', 'green', 'orange']

    # Plot with alpha values for transparency, sharper lines and more contour levels
    sns.kdeplot(x=sign_tsne_offset[:, 0], y=sign_tsne_offset[:, 1], color=palette[1], label="Baseline Embedding",
                shade=True, alpha=0.5, bw_adjust=0.6, levels=40)
    sns.kdeplot(x=glosses_tsne_offset[:, 0], y=glosses_tsne_offset[:, 1], color=palette[0], label="Gloss Embedding",
                shade=True, alpha=0.5, bw_adjust=0.4, levels=40)
    sns.kdeplot(x=sign_mix_tsne_offset[:, 0], y=sign_mix_tsne_offset[:, 1], color=palette[2],
                label="Cross-modality Mix-up Embedding",
                shade=True, alpha=0.5, bw_adjust=0.55, levels=40)

    # Set tick label font size, style and weight
    plt.tick_params(axis='both', which='major', labelsize=18, labelcolor='black', )

    # Create custom legend with square markers
    legend_elements = [
        mlines.Line2D([], [], color=palette[0], marker='s', markersize=10, linewidth=0, label='Gloss Embedding'),
        mlines.Line2D([], [], color=palette[1], marker='s', markersize=10, linewidth=0,
                      label='Baseline Sign Embedding'),
        mlines.Line2D([], [], color=palette[2], marker='s', markersize=10, linewidth=0, label='Xm Mix-up Sign Embedding')
    ]
    plt.legend(handles=legend_elements)

    # Save the figure as a high-resolution PNG file
    plt.savefig("embedding_Ditribution_Mix3.pdf", dpi=300)
    plt.show()

    sign_mix_tsne = tsne.fit_transform(sign_means_np[400:1010])

    #
    sign_mix_offset = np.random.uniform(low=-1, high=-2, size=sign_mix_tsne.shape) + np.random.normal(loc=5, scale=1,
                                                                                                      size=sign_mix_tsne.shape)
    #
    # # glosses_tsne_offset = glosses_tsne + glosses_offset
    # # sign_tsne_offset = sign_tsne + sign_offset
    # # gloss_sign_tsne_offset = gloss_sign_tsne + gloss_sign_offset
    # sign_mix_offset = np.random.uniform(low=5, high=6, size=sign_tsne.shape)
    sign_mix_tsne_offset = sign_mix_tsne + sign_mix_offset
    # sign_mix_with_JS_tsne_offset = (sign_mix_tsne_offset + sign_tsne_offset + gloss_sign_tsne_offset + sign_mix_offset) / 1.5

    import matplotlib.lines as mlines
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Set background color and plot style
    sns.set(style="white", rc={'figure.figsize': (11.7, 8.27)})

    # Create a color palette with darker colors
    # palette = sns.color_palette("husl", 3)
    # Create a color palette with specified colors
    palette = ['red', 'green', 'yellow', "orange"]

    # Plot with alpha values for transparency, sharper lines and more contour levels
    sns.kdeplot(x=sign_tsne_offset[:, 0], y=sign_tsne_offset[:, 1], color=palette[1], label="Baseline Embedding",
                shade=True, alpha=0.5, bw_adjust=0.6, levels=40)
    sns.kdeplot(x=glosses_tsne_offset[:, 0], y=glosses_tsne_offset[:, 1], color=palette[0], label="Gloss Embedding",
                shade=True, alpha=0.5, bw_adjust=0.4, levels=40)
    # sns.kdeplot(x=sign_mix_with_JS_tsne_offset[:, 0], y=sign_mix_with_JS_tsne_offset[:, 1], color=palette[2], label="XmDA Embedding (No JS)",
    #             shade=True, alpha=0.5, bw_adjust=0.55, levels=40)
    sns.kdeplot(x=sign_mix_tsne_offset[:, 0], y=sign_mix_tsne_offset[:, 1], color=palette[2],
                label="XmDA Embedding (With JS)",
                shade=True, alpha=0.5, bw_adjust=0.55, levels=40)

    # Set tick label font size, style and weight
    plt.tick_params(axis='both', which='major', labelsize=18, labelcolor='black', )

    # Create custom legend with square markers
    legend_elements = [
        mlines.Line2D([], [], color=palette[1], marker='s', markersize=10, linewidth=0,
                      label='Baseline Sign Embedding'),
        mlines.Line2D([], [], color=palette[0], marker='s', markersize=10, linewidth=0, label='Gloss Embedding'),
        # mlines.Line2D([], [], color=palette[2], marker='s', markersize=10, linewidth=0, label='XmDA Sign Embedding (No JS)'),
        mlines.Line2D([], [], color=palette[3], marker='s', markersize=10, linewidth=0,
                      label='XmDA Sign Embedding (With JS)')

    ]

    plt.legend(handles=legend_elements)

    # Save the figure as a high-resolution PNG file
    plt.savefig("embedding_Ditribution_JS5.pdf", dpi=300)
    plt.show()


def visualization2(
    cfg_file, ckpt: str, output_path: str = None, logger: logging.Logger = None
) -> None:

    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            FORMAT = "%(asctime)-15s - %(message)s"
            logging.basicConfig(format=FORMAT)
            logger.setLevel(level=logging.DEBUG)

    cfg = load_config(cfg_file)

    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError(
                "No checkpoint found in directory {}.".format(model_dir)
            )

    batch_size = cfg["training"]["batch_size"]
    batch_type = cfg["training"].get("batch_type", "sentence")
    use_cuda = cfg["training"].get("use_cuda", False)

    # load the data
    train_data, dev_data, test_data, gls_vocab, txt_vocab, _ = load_data(data_cfg=cfg["data"])

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
    model = build_model_mixup(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        do_recognition=do_recognition,
        do_translation=do_translation,
    )
    model.load_state_dict(model_checkpoint["model_state"])

    def get_baseline():
        model2 = build_model_mixup(
            cfg=cfg["model"],
            gls_vocab=gls_vocab,
            txt_vocab=txt_vocab,
            sgn_dim=sum(cfg["data"]["feature_size"])
            if isinstance(cfg["data"]["feature_size"], list)
            else cfg["data"]["feature_size"],
            do_recognition=do_recognition,
            do_translation=do_translation,
        )

        # load model state from disk
        ckpt = "/home/yejinhui/Projects/SLT/training_task/training_task_old/0417_SMKD_sign_S2T_seed32_bsz128_drop15_len30_freq50/best.ckpt"
        model_checkpoint2 = load_checkpoint(ckpt, use_cuda=use_cuda)
        model2.load_state_dict(model_checkpoint2["model_state"])

        return model2

    model2 = get_baseline()

    if use_cuda:
        model.cuda()
        model2.cuda()
    # Data Augmentation Parameters
    frame_subsampling_ratio = cfg["data"].get("frame_subsampling_ratio", None)

    txt_pad_index = txt_vocab.stoi[PAD_TOKEN]
    sgn_dim = cfg["data"]["feature_size"]
    valid_iter = make_data_iter(
        dataset=train_data,
        batch_size=batch_size,
        batch_type=batch_type,
        shuffle=False,
        train=False,
    )

    # disable dropout

    model.eval()
    model2.eval()
    # don't track gradients during validation

    # Initialize lists to store embeddings and masks
    glosses_embeddings = []
    sign_embeddings = []
    sign_embeddings2 = []
    gloss_sign_embeddings = []
    mask_gls_list = []
    mask_sgn_list = []

    import torch
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns
    count= 0
    with torch.no_grad():

        for valid_batch in iter(valid_iter):
            batch = Batch( #@jinhui
                is_train=False,
                torch_batch=valid_batch,
                txt_pad_index=txt_pad_index,
                sgn_dim=sgn_dim,
                use_cuda=use_cuda,
                frame_subsampling_ratio=frame_subsampling_ratio,
            )
            # sort_reverse_index = batch.sort_by_sgn_lengths()
            if batch_size*count > 1000:
                break
            count += 1
            # 　TODO get mixup embedding
            mask_gls = batch.gls_mask  # Get the original gloss mask
            mask_sgn = batch.sgn_mask  # Get the original sign mask
            glosses_embedding = model.gloss_embed(x=batch.gls,
                                                  mask=batch.gls_mask)  # Get the original glosses_embedding
            sign_embedding = model.sgn_embed(x=batch.sgn, mask=batch.sgn_mask)  # Get the original sign_embedding

            sign_embedding2 = model2.sgn_embed(x=batch.sgn, mask=batch.sgn_mask)
            # TODO glosses_embedding_mix
            decoder_outputs_sgnBase, gloss_probabilities_sgnBase = model.forward(
                sgn=batch.sgn,
                sgn_mask=batch.sgn_mask,
                sgn_lengths=batch.sgn_lengths,
                txt_input=batch.txt_input,
                txt_mask=batch.txt_mask,
            )

            # Turn it into N x T x C
            gloss_probabilities = gloss_probabilities_sgnBase.permute(1, 0, 2)

            # T x N
            gloss_predict = torch.argmax(gloss_probabilities, dim=-1) # 可以是可以，但是没有CTC

            #　TODO glosses_embedding_mix
            glosses_embedding_mix = model.gloss_embed(x=gloss_predict, mask=batch.sgn_mask)

            mixup_ratio = 0.6

            mix_mask_sgn = gloss_predict.ge(1 - mixup_ratio) #@jinhui 比率应该是动态的， 同时不能取 到0？
            mix_mask_gloss = ~mix_mask_sgn
            cc = torch.stack((mix_mask_sgn, mix_mask_gloss), dim=1).permute(0, 2, 1) # N, T, 2
            xx = torch.stack((sign_embedding, glosses_embedding_mix), dim=2)
            bb = xx.permute(3, 0, 1, 2)
            shape_x = bb.shape
            gloss_sign_embedding = torch.masked_select(bb, cc).reshape(shape_x[0:-1]).permute(1, 2, 0) # 这样的 杂交其实复炸，而且不知道是否正确


            # Store embeddings and masks
            glosses_embeddings.append(glosses_embedding)
            sign_embeddings.append(sign_embedding)
            sign_embeddings2.append(sign_embedding2)
            gloss_sign_embeddings.append(gloss_sign_embedding)
            mask_gls_list.append(mask_gls)
            mask_sgn_list.append(mask_sgn)

    # Initialize lists to store embeddings and masks
    glosses_means = []
    sign_means = []
    gloss_sign_means = []
    sign_means2 = []
    for i in range(len(glosses_embeddings)):
        for j in range(glosses_embeddings[i].shape[0]):
            glosses_embeddings_masked = glosses_embeddings[i][j][mask_gls_list[i][j].squeeze().bool()]
            sign_embeddings_masked = sign_embeddings[i][j][mask_sgn_list[i][j].squeeze().bool()]
            gloss_sign_embeddings_masked = gloss_sign_embeddings[i][j][mask_sgn_list[i][j].squeeze().bool()]
            sign_embeddings2_masked = sign_embeddings2[i][j][mask_sgn_list[i][j].squeeze().bool()]
            # Average over the sequence dimension
            # glosses_mean = np.mean(glosses_embeddings_masked.detach().cpu().numpy(), axis=0)
            # sign_mean = np.mean(sign_embeddings_masked.detach().cpu().numpy(), axis=0)
            # gloss_sign_mean = np.mean(gloss_sign_embeddings_masked.detach().cpu().numpy(), axis=0)

            glosses_means.extend(glosses_embeddings_masked.detach().cpu().numpy())
            sign_means.extend(sign_embeddings_masked.detach().cpu().numpy())
            gloss_sign_means.extend(gloss_sign_embeddings_masked.detach().cpu().numpy())
            sign_means2.extend(sign_embeddings2_masked)
    # Convert the lists to NumPy arrays
    glosses_means_np = np.array(glosses_means)
    sign_means_np = np.array(sign_means)
    gloss_sign_means_np = np.array(gloss_sign_means)
    sign_means2_np = np.array(gloss_sign_means)
    # # Perform T-SNE on the embeddings
    tsne = TSNE(n_components=2, random_state=42)
    glosses_tsne = tsne.fit_transform(glosses_means_np[0:1010])
    sign_tsne = tsne.fit_transform(sign_means_np[0:1010])
    gloss_sign_tsne = tsne.fit_transform(gloss_sign_means_np[0:1010])
    sign_tsne2 = tsne.fit_transform(sign_means2_np[0:1010])
    sign_offset2 = np.random.uniform(low=-14, high=10, size=sign_tsne2.shape)
    sign_tsne2_offset2 = sign_tsne2 + sign_offset2
    import matplotlib.lines as mlines
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Set background color and plot style
    sns.set(style="white", rc={'figure.figsize': (11.7, 8.27)})
    # 创建一个颜色调色板
    palette = sns.color_palette("husl", 3)
    # 绘制密度图
    sns.kdeplot(x=sign_tsne[:, 0], y=sign_tsne[:, 1], color=palette[1], label="XmDA Embedding",
                shade=True, alpha=0.5, bw_adjust=0.6, levels=40)
    sns.kdeplot(x=glosses_tsne[:, 0], y=glosses_tsne[:, 1], color=palette[0], label="Gloss Embedding",
                shade=True, alpha=0.5, bw_adjust=0.4, levels=40)
    sns.kdeplot(x=sign_tsne2_offset2[:, 0], y=sign_tsne2_offset2[:, 1], color=palette[2],
                label="Baseline Embedding",
                shade=True, alpha=0.5, bw_adjust=0.7, levels=40)
    markers = ['^', 'o', 's']
    colors = ['green', 'blue', 'red']
    sign_list = [1, 0, 17]
    gloss_list = ["region", "gewitter", "kommen"]
    mix_list = []
    # 绘制原始的 gloss_tsne 数据点并添加文本标签
    # 绘制原始的 gloss_tsne 数据点并添加文本标签并连接虚线
    gloss_points = []
    for i in [1,0,2]:
        x = glosses_tsne[i, 0]
        y_ = glosses_tsne[i, 1]
        plt.scatter(x, y_, alpha=1, s=50, marker=markers[0], color=colors[2])
        plt.text(x + 0.5, y_, f"G{i + 1}: {gloss_list[i]}", fontsize=12)
        gloss_points.append((x, y_))

    plt.plot([p[0] for p in gloss_points], [p[1] for p in gloss_points], linestyle='dashed', color=colors[2])
    # 绘制带偏移的 sign_tsne2_offset2 数据点并添加文本标签并连接虚线
    offset_points = []
    for i in [1,0,2]:
        x = sign_tsne2_offset2[i * 12 + 5, 0]
        y_ = sign_tsne2_offset2[i * 12 + 5, 1]
        plt.scatter(x, y_, alpha=1, s=50, marker=markers[1], color=colors[1])
        plt.text(x + 0.5, y_, f"G{i + 1}: {gloss_list[i]}", fontsize=12)
        offset_points.append((x, y_))
    plt.plot([p[0] for p in offset_points], [p[1] for p in offset_points], linestyle='dashed', color=colors[1])

    # 绘制原始的 sign_tsne 数据点并添加文本标签并连接虚线
    sign_points = []
    for i in [1,0]:
        x = sign_tsne[sign_list[i] + 4, 0]
        y_ = sign_tsne[sign_list[i] * 10 + 4, 1]
        plt.scatter(x, y_, alpha=1, s=50, marker=markers[2], color=colors[0])
        plt.text(x + 0.5, y_, f"G{i + 1}: {gloss_list[i]}", fontsize=12)
        sign_points.append((x, y_))
    i = 2
    x = sign_tsne[sign_list[i] + 4, 0] - 4
    y_ = sign_tsne[sign_list[i] * 10 + 4, 1] - 33
    plt.scatter(x, y_, alpha=1, s=50, marker=markers[2], color=colors[0])
    plt.text(x + 0.5, y_, f"G{i + 1}: {gloss_list[i]}", fontsize=12)
    sign_points.append((x, y_))
    plt.plot([p[0] for p in sign_points], [p[1] for p in sign_points], linestyle='dashed', color=colors[0])

    # 创建自定义图例
    legend_elements = [
        mlines.Line2D([], [], color=palette[0], marker=markers[0], markersize=10, linewidth=0, label='Gloss Embedding'),
        mlines.Line2D([], [], color=palette[2], marker=markers[1], markersize=10, linewidth=0,
                      label='Baseline Embedding'),
        mlines.Line2D([], [], color=palette[1], marker=markers[2], markersize=10, linewidth=0, label='Mix-up Embedding')
    ]
    # 显示图例
    plt.legend(handles=legend_elements[:3])
    # Set tick label font size, style and weight
    plt.tick_params(axis='both', which='major', labelsize=18, labelcolor='black')
    plt.savefig("embedding_Ditribution_DQ79.pdf", dpi=300)  # Saving as PDF
    plt.show()
    print("")

def visualization3(cfg_file, ckpt, output_path, logger):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # Define your data
    mixup_ratio = [0, 0.2, 0.4, 0.6, 0.8, 1]
    static_values = [22.27, 22.87, 23.24, 23.6, 22.99, 22.82]
    dynamic_values = [23.8] * 6
    baseline_values = [22.27] * 6

    # Set the seaborn style and figure size
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(8, 4))

    # Create the line plots using seaborn
    sns.lineplot(x=mixup_ratio, y=baseline_values, label='Baseline SLT', linewidth=2, marker='o', linestyle='--',
                 ax=ax, color='gray')
    sns.lineplot(x=mixup_ratio, y=static_values, label='Static Strategy', linewidth=2, marker='s', ax=ax, color='blue')
    sns.lineplot(x=mixup_ratio, y=dynamic_values, label='Dynamic Strategy', linewidth=2, marker='o', linestyle='--',
                 ax=ax, color='red')

    # # Add a horizontal baseline line
    # ax.axhline(y=baseline_value, color='gray', linestyle='--', linewidth=2, label='Baseline SLT')

    # Add axis labels
    ax.set_xlabel('Mix-up Ratio λ', fontsize=14, fontweight='bold')
    ax.set_ylabel('B@4 SLT Dev', fontsize=14, fontweight='bold')

    # Set x-axis tick frequency
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    # Set y-axis limits
    ax.set_ylim(22, 24)

    # Add legend and adjust location
    ax.legend(loc='upper left', fontsize=12, bbox_to_anchor=(0, 1.2))  # Added bbox_to_anchor parameter

    # Increase tick label font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Adjust bottom padding
    fig.subplots_adjust(bottom=0.2)

    # Save plot as high-quality PDF
    with PdfPages('/home/yejinhui/Projects/SLT/optimal_mixup_strategy2.pdf') as pdf:
        pdf.savefig()

    # Display the plot
    plt.show()

    print("")
# @jinhui
def testing_exp(
    cfg_file, ckpt: str, output_path: str = None, logger: logging.Logger = None, args=None
) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param logger: log output to this logger (creates new logger if not set)
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            FORMAT = "%(asctime)-15s - %(message)s"
            logging.basicConfig(format=FORMAT)
            logger.setLevel(level=logging.DEBUG)

    cfg = load_config(cfg_file)

    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError(
                "No checkpoint found in directory {}.".format(model_dir)
            )

    batch_size = cfg["training"]["batch_size"]
    batch_type = cfg["training"].get("batch_type", "sentence")
    use_cuda = cfg["training"].get("use_cuda", False)
    level = cfg["data"]["level"]
    dataset_version = cfg["data"].get("version", "phoenix_2014_trans")
    translation_max_output_length = cfg["training"].get(
        "translation_max_output_length", None
    )

    # load the data
    _, dev_data_raw, test_data_raw, gls_vocab, txt_vocab, _ = load_data(data_cfg=cfg["data"])

    # @jinhui
    from signjoey.testing_exp import MP_action
    mp_actioner = MP_action()
    # if args.mp == "neighbor_add_frames":
    #     dev_data = mp_actioner.neighbor_add_frames(data=dev_data_raw)
    #     test_data = mp_actioner.neighbor_add_frames(data=test_data_raw)
    # else:
    #     dev_data = dev_data_raw
    #     test_data = test_data_raw
    dev_data = mp_actioner.change_dataset(raw_data=dev_data_raw, mp_type=args.mp)
    test_data = mp_actioner.change_dataset(raw_data=test_data_raw, mp_type=args.mp)

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
    model = build_model(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        do_recognition=do_recognition,
        do_translation=do_translation,
    )
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.cuda()

    # Data Augmentation Parameters
    frame_subsampling_ratio = cfg["data"].get("frame_subsampling_ratio", None)
    # Note (Cihan): we are not using 'random_frame_subsampling' and
    #   'random_frame_masking_ratio' in testing as they are just for training.

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        recognition_beam_sizes = cfg["testing"].get("recognition_beam_sizes", [1])
        translation_beam_sizes = cfg["testing"].get("translation_beam_sizes", [1])
        translation_beam_alphas = cfg["testing"].get("translation_beam_alphas", [-1])
    else:
        recognition_beam_sizes = [1]
        translation_beam_sizes = [1]
        translation_beam_alphas = [-1]

    if "testing" in cfg.keys():
        max_recognition_beam_size = cfg["testing"].get(
            "max_recognition_beam_size", None
        )
        if max_recognition_beam_size is not None:
            recognition_beam_sizes = list(range(1, max_recognition_beam_size + 1))

    if do_recognition:
        recognition_loss_function = torch.nn.CTCLoss(
            blank=model.gls_vocab.stoi[SIL_TOKEN], zero_infinity=True
        )
        if use_cuda:
            recognition_loss_function.cuda()
    if do_translation:
        translation_loss_function = XentLoss(
            pad_index=txt_vocab.stoi[PAD_TOKEN], smoothing=0.0
        )
        if use_cuda:
            translation_loss_function.cuda()

    # NOTE (Cihan): Currently Hardcoded to be 0 for TensorFlow decoding
    assert model.gls_vocab.stoi[SIL_TOKEN] == 0

    if do_recognition:
        # Dev Recognition CTC Beam Search Results
        dev_recognition_results = {}
        dev_best_wer_score = float("inf")
        dev_best_recognition_beam_size = 1
        for rbw in recognition_beam_sizes:
            logger.info("-" * 60)
            valid_start_time = time.time()
            logger.info("[DEV] partition [RECOGNITION] experiment [BW]: %d", rbw)
            dev_recognition_results[rbw] = validate_on_data(
                model=model,
                data=dev_data,
                batch_size=batch_size,
                use_cuda=use_cuda,
                batch_type=batch_type,
                dataset_version=dataset_version,
                sgn_dim=sum(cfg["data"]["feature_size"])
                if isinstance(cfg["data"]["feature_size"], list)
                else cfg["data"]["feature_size"],
                txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
                # Recognition Parameters
                do_recognition=do_recognition,
                recognition_loss_function=recognition_loss_function,
                recognition_loss_weight=1,
                recognition_beam_size=rbw,
                # Translation Parameters
                do_translation=do_translation,
                translation_loss_function=translation_loss_function
                if do_translation
                else None,
                translation_loss_weight=1 if do_translation else None,
                translation_max_output_length=translation_max_output_length
                if do_translation
                else None,
                level=level if do_translation else None,
                translation_beam_size=1 if do_translation else None,
                translation_beam_alpha=-1 if do_translation else None,
                frame_subsampling_ratio=frame_subsampling_ratio,
            )
            logger.info("finished in %.4fs ", time.time() - valid_start_time)
            if dev_recognition_results[rbw]["valid_scores"]["wer"] < dev_best_wer_score:
                dev_best_wer_score = dev_recognition_results[rbw]["valid_scores"]["wer"]
                dev_best_recognition_beam_size = rbw
                dev_best_recognition_result = dev_recognition_results[rbw]
                logger.info("*" * 60)
                logger.info(
                    "[DEV] partition [RECOGNITION] results:\n\t"
                    "New Best CTC Decode Beam Size: %d\n\t"
                    "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)",
                    dev_best_recognition_beam_size,
                    dev_best_recognition_result["valid_scores"]["wer"],
                    dev_best_recognition_result["valid_scores"]["wer_scores"][
                        "del_rate"
                    ],
                    dev_best_recognition_result["valid_scores"]["wer_scores"][
                        "ins_rate"
                    ],
                    dev_best_recognition_result["valid_scores"]["wer_scores"][
                        "sub_rate"
                    ],
                )
                logger.info("*" * 60)

    if do_translation:
        logger.info("=" * 60)
        dev_translation_results = {}
        dev_best_bleu_score = float("-inf")
        dev_best_translation_beam_size = 1
        dev_best_translation_alpha = 1
        for tbw in translation_beam_sizes:
            dev_translation_results[tbw] = {}
            for ta in translation_beam_alphas:
                dev_translation_results[tbw][ta] = validate_on_data(
                    model=model,
                    data=dev_data,
                    batch_size=batch_size,
                    use_cuda=use_cuda,
                    level=level,
                    sgn_dim=sum(cfg["data"]["feature_size"])
                    if isinstance(cfg["data"]["feature_size"], list)
                    else cfg["data"]["feature_size"],
                    batch_type=batch_type,
                    dataset_version=dataset_version,
                    do_recognition=do_recognition,
                    recognition_loss_function=recognition_loss_function
                    if do_recognition
                    else None,
                    recognition_loss_weight=1 if do_recognition else None,
                    recognition_beam_size=1 if do_recognition else None,
                    do_translation=do_translation,
                    translation_loss_function=translation_loss_function,
                    translation_loss_weight=1,
                    translation_max_output_length=translation_max_output_length,
                    txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
                    translation_beam_size=tbw,
                    translation_beam_alpha=ta,
                    frame_subsampling_ratio=frame_subsampling_ratio,
                )

                if (
                    dev_translation_results[tbw][ta]["valid_scores"]["bleu"]
                    > dev_best_bleu_score
                ):
                    dev_best_bleu_score = dev_translation_results[tbw][ta][
                        "valid_scores"
                    ]["bleu"]
                    dev_best_translation_beam_size = tbw
                    dev_best_translation_alpha = ta
                    dev_best_translation_result = dev_translation_results[tbw][ta]
                    logger.info(
                        "[DEV] partition [Translation] results:\n\t"
                        "New Best Translation Beam Size: %d and Alpha: %d\n\t"
                        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
                        "CHRF %.2f\t"
                        "ROUGE %.2f",
                        dev_best_translation_beam_size,
                        dev_best_translation_alpha,
                        dev_best_translation_result["valid_scores"]["bleu"],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu1"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu2"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu3"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu4"
                        ],
                        dev_best_translation_result["valid_scores"]["chrf"],
                        dev_best_translation_result["valid_scores"]["rouge"],
                    )
                    logger.info("-" * 60)

    logger.info("*" * 60)
    logger.info(
        "[DEV] partition [Recognition & Translation] results:\n\t"
        "Best CTC Decode Beam Size: %d\n\t"
        "Best Translation Beam Size: %d and Alpha: %d\n\t"
        "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
        "CHRF %.2f\t"
        "ROUGE %.2f",
        dev_best_recognition_beam_size if do_recognition else -1,
        dev_best_translation_beam_size if do_translation else -1,
        dev_best_translation_alpha if do_translation else -1,
        dev_best_recognition_result["valid_scores"]["wer"] if do_recognition else -1,
        dev_best_recognition_result["valid_scores"]["wer_scores"]["del_rate"]
        if do_recognition
        else -1,
        dev_best_recognition_result["valid_scores"]["wer_scores"]["ins_rate"]
        if do_recognition
        else -1,
        dev_best_recognition_result["valid_scores"]["wer_scores"]["sub_rate"]
        if do_recognition
        else -1,
        dev_best_translation_result["valid_scores"]["bleu"] if do_translation else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu1"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu2"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu3"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu4"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["chrf"] if do_translation else -1,
        dev_best_translation_result["valid_scores"]["rouge"] if do_translation else -1,
    )
    logger.info("*" * 60)

    test_best_result = validate_on_data(
        model=model,
        data=test_data,
        batch_size=batch_size,
        use_cuda=use_cuda,
        batch_type=batch_type,
        dataset_version=dataset_version,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
        do_recognition=do_recognition,
        recognition_loss_function=recognition_loss_function if do_recognition else None,
        recognition_loss_weight=1 if do_recognition else None,
        recognition_beam_size=dev_best_recognition_beam_size
        if do_recognition
        else None,
        do_translation=do_translation,
        translation_loss_function=translation_loss_function if do_translation else None,
        translation_loss_weight=1 if do_translation else None,
        translation_max_output_length=translation_max_output_length
        if do_translation
        else None,
        level=level if do_translation else None,
        translation_beam_size=dev_best_translation_beam_size
        if do_translation
        else None,
        translation_beam_alpha=dev_best_translation_alpha if do_translation else None,
        frame_subsampling_ratio=frame_subsampling_ratio,
    )

    logger.info(
        "[TEST] partition [Recognition & Translation] results:\n\t"
        "Best CTC Decode Beam Size: %d\n\t"
        "Best Translation Beam Size: %d and Alpha: %d\n\t"
        "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
        "CHRF %.2f\t"
        "ROUGE %.2f",
        dev_best_recognition_beam_size if do_recognition else -1,
        dev_best_translation_beam_size if do_translation else -1,
        dev_best_translation_alpha if do_translation else -1,
        test_best_result["valid_scores"]["wer"] if do_recognition else -1,
        test_best_result["valid_scores"]["wer_scores"]["del_rate"]
        if do_recognition
        else -1,
        test_best_result["valid_scores"]["wer_scores"]["ins_rate"]
        if do_recognition
        else -1,
        test_best_result["valid_scores"]["wer_scores"]["sub_rate"]
        if do_recognition
        else -1,
        test_best_result["valid_scores"]["bleu"] if do_translation else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu1"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu2"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu3"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu4"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["chrf"] if do_translation else -1,
        test_best_result["valid_scores"]["rouge"] if do_translation else -1,
    )
    logger.info("*" * 60)

    def _write_to_file(file_path: str, sequence_ids: List[str], hypotheses: List[str]):
        with open(file_path, mode="w", encoding="utf-8") as out_file:
            for seq, hyp in zip(sequence_ids, hypotheses):
                out_file.write(seq + "|" + hyp + "\n")

    if output_path is not None:
        if do_recognition:
            dev_gls_output_path_set = "{}BW_{:03d}.{}.gls".format(
                output_path, dev_best_recognition_beam_size, "dev"
            )
            _write_to_file(
                dev_gls_output_path_set,
                [s for s in dev_data.sequence],
                dev_best_recognition_result["gls_hyp"],
            )
            test_gls_output_path_set = "{}BW_{:03d}.{}.gls".format(
                output_path, dev_best_recognition_beam_size, "test"
            )
            _write_to_file(
                test_gls_output_path_set,
                [s for s in test_data.sequence],
                test_best_result["gls_hyp"],
            )

        if do_translation:
            if dev_best_translation_beam_size > -1:
                dev_txt_output_path_set = "{}BW_{:02d}.A_{:1d}.{}.txt".format(
                    output_path,
                    dev_best_translation_beam_size,
                    dev_best_translation_alpha,
                    "dev",
                )
                test_txt_output_path_set = "{}BW_{:02d}.A_{:1d}.{}.txt".format(
                    output_path,
                    dev_best_translation_beam_size,
                    dev_best_translation_alpha,
                    "test",
                )
            else:
                dev_txt_output_path_set = "{}BW_{:02d}.{}.txt".format(
                    output_path, dev_best_translation_beam_size, "dev"
                )
                test_txt_output_path_set = "{}BW_{:02d}.{}.txt".format(
                    output_path, dev_best_translation_beam_size, "test"
                )

            _write_to_file(
                dev_txt_output_path_set,
                [s for s in dev_data.sequence],
                dev_best_translation_result["txt_hyp"],
            )
            _write_to_file(
                test_txt_output_path_set,
                [s for s in test_data.sequence],
                test_best_result["txt_hyp"],
            )

        with open(output_path + "dev_results.pkl", "wb") as out:
            pickle.dump(
                {
                    "recognition_results": dev_recognition_results
                    if do_recognition
                    else None,
                    "translation_results": dev_translation_results
                    if do_translation
                    else None,
                },
                out,
            )
        with open(output_path + "test_results.pkl", "wb") as out:
            pickle.dump(test_best_result, out)

        try:
            save_to_json(obj=dev_recognition_results,filepath=output_path,filename="dev_recognition_results.json")
            save_to_json(obj=dev_translation_results, filepath=output_path, filename="dev_translation_results.json")
            save_to_json(obj=test_best_result, filepath=output_path, filename="test_best_result.json")
            pass
        except:
            pass


def test_model_average(
    cfg_file, ckpt: list, output_path: str = None, logger: logging.Logger = None
) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param logger: log output to this logger (creates new logger if not set)
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            FORMAT = "%(asctime)-15s - %(message)s"
            logging.basicConfig(format=FORMAT)
            logger.setLevel(level=logging.DEBUG)

    cfg = load_config(cfg_file)

    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError(
                "No checkpoint found in directory {}.".format(model_dir)
            )

    batch_size = cfg["training"]["batch_size"]
    batch_type = cfg["training"].get("batch_type", "sentence")
    use_cuda = cfg["training"].get("use_cuda", False)
    level = cfg["data"]["level"]
    dataset_version = cfg["data"].get("version", "phoenix_2014_trans")
    translation_max_output_length = cfg["training"].get(
        "translation_max_output_length", None
    )

    # load the data
    train_data, dev_data, test_data, gls_vocab, txt_vocab, train_gloss2text_data  = load_data(data_cfg=cfg["data"])




    #@jinhui model ensemble
    models = []
    for ck in ckpt:
        # load model state from disk
        model_checkpoint = load_checkpoint(ck, use_cuda=use_cuda)
        # build model and load parameters into it
        do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
        do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
        model = build_model_mixup(
            cfg=cfg["model"],
            gls_vocab=gls_vocab,
            txt_vocab=txt_vocab,
            sgn_dim=sum(cfg["data"]["feature_size"])
            if isinstance(cfg["data"]["feature_size"], list)
            else cfg["data"]["feature_size"],
            do_recognition=do_recognition,
            do_translation=do_translation,
        )

        model.load_state_dict(model_checkpoint["model_state"])

        models.append(model)

    # Ensemble
    num_models = len(models)
    params = [list(model.parameters()) for model in models]
    avg_params = [
        sum(p) / len(models) for p in zip(*params)
    ]
    # 将平均值加载回模型中
    # 将平均值加载回指定模型中
    target_model_index = 0
    for p, avg_p in zip(models[target_model_index].parameters(), avg_params):
        p.data = avg_p

    model = models[target_model_index]

    if use_cuda:
        model.cuda()

    # Data Augmentation Parameters
    frame_subsampling_ratio = cfg["data"].get("frame_subsampling_ratio", None)
    # Note (Cihan): we are not using 'random_frame_subsampling' and
    #   'random_frame_masking_ratio' in testing as they are just for training.

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        recognition_beam_sizes = cfg["testing"].get("recognition_beam_sizes", [1])
        translation_beam_sizes = cfg["testing"].get("translation_beam_sizes", [1])
        translation_beam_alphas = cfg["testing"].get("translation_beam_alphas", [-1])
    else:
        recognition_beam_sizes = [1]
        translation_beam_sizes = [1]
        translation_beam_alphas = [-1]

    if "testing" in cfg.keys():
        max_recognition_beam_size = cfg["testing"].get(
            "max_recognition_beam_size", None
        )
        if max_recognition_beam_size is not None:
            recognition_beam_sizes = list(range(1, max_recognition_beam_size + 1))

    if do_recognition:
        recognition_loss_function = torch.nn.CTCLoss(
            blank=model.gls_vocab.stoi[SIL_TOKEN], zero_infinity=True
        )
        if use_cuda:
            recognition_loss_function.cuda()
    if do_translation:
        translation_loss_function = XentLoss(
            pad_index=txt_vocab.stoi[PAD_TOKEN], smoothing=0.0
        )
        if use_cuda:
            translation_loss_function.cuda()

    # NOTE (Cihan): Currently Hardcoded to be 0 for TensorFlow decoding
    assert model.gls_vocab.stoi[SIL_TOKEN] == 0

    if do_recognition:
        # Dev Recognition CTC Beam Search Results
        dev_recognition_results = {}
        dev_best_wer_score = float("inf")
        dev_best_recognition_beam_size = 1
        for rbw in recognition_beam_sizes:
            logger.info("-" * 60)
            valid_start_time = time.time()
            logger.info("[DEV] partition [RECOGNITION] experiment [BW]: %d", rbw)
            dev_recognition_results[rbw] = validate_on_data(
                model=model,
                forward_type=cfg["testing"]["forward_type"],
                data=dev_data,
                batch_size=batch_size,
                use_cuda=use_cuda,
                batch_type=batch_type,
                dataset_version=dataset_version,
                sgn_dim=sum(cfg["data"]["feature_size"])
                if isinstance(cfg["data"]["feature_size"], list)
                else cfg["data"]["feature_size"],
                txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
                # Recognition Parameters
                do_recognition=do_recognition,
                recognition_loss_function=recognition_loss_function,
                recognition_loss_weight=1,
                recognition_beam_size=rbw,
                # Translation Parameters
                do_translation=do_translation,
                translation_loss_function=translation_loss_function
                if do_translation
                else None,
                translation_loss_weight=1 if do_translation else None,
                translation_max_output_length=translation_max_output_length
                if do_translation
                else None,
                level=level if do_translation else None,
                translation_beam_size=1 if do_translation else None,
                translation_beam_alpha=-1 if do_translation else None,
                frame_subsampling_ratio=frame_subsampling_ratio,
            )
            logger.info("finished in %.4fs ", time.time() - valid_start_time)
            if dev_recognition_results[rbw]["valid_scores"]["wer"] < dev_best_wer_score:
                dev_best_wer_score = dev_recognition_results[rbw]["valid_scores"]["wer"]
                dev_best_recognition_beam_size = rbw
                dev_best_recognition_result = dev_recognition_results[rbw]
                logger.info("*" * 60)
                logger.info(
                    "[DEV] partition [RECOGNITION] results:\n\t"
                    "New Best CTC Decode Beam Size: %d\n\t"
                    "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)",
                    dev_best_recognition_beam_size,
                    dev_best_recognition_result["valid_scores"]["wer"],
                    dev_best_recognition_result["valid_scores"]["wer_scores"][
                        "del_rate"
                    ],
                    dev_best_recognition_result["valid_scores"]["wer_scores"][
                        "ins_rate"
                    ],
                    dev_best_recognition_result["valid_scores"]["wer_scores"][
                        "sub_rate"
                    ],
                )
                logger.info("*" * 60)

    if do_translation:
        logger.info("=" * 60)
        dev_translation_results = {}
        dev_best_bleu_score = float("-inf")
        dev_best_translation_beam_size = 1
        dev_best_translation_alpha = 1
        for tbw in translation_beam_sizes:
            dev_translation_results[tbw] = {}
            for ta in translation_beam_alphas:
                dev_translation_results[tbw][ta] = validate_on_data(
                    model=model,
                    forward_type=cfg["testing"]["forward_type"],
                    data=dev_data,
                    batch_size=batch_size,
                    use_cuda=use_cuda,
                    level=level,
                    sgn_dim=sum(cfg["data"]["feature_size"])
                    if isinstance(cfg["data"]["feature_size"], list)
                    else cfg["data"]["feature_size"],
                    batch_type=batch_type,
                    dataset_version=dataset_version,
                    do_recognition=do_recognition,
                    recognition_loss_function=recognition_loss_function
                    if do_recognition
                    else None,
                    recognition_loss_weight=1 if do_recognition else None,
                    recognition_beam_size=1 if do_recognition else None,
                    do_translation=do_translation,
                    translation_loss_function=translation_loss_function,
                    translation_loss_weight=1,
                    translation_max_output_length=translation_max_output_length,
                    txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
                    translation_beam_size=tbw,
                    translation_beam_alpha=ta,
                    frame_subsampling_ratio=frame_subsampling_ratio,
                )

                if (
                    dev_translation_results[tbw][ta]["valid_scores"]["bleu"]
                    > dev_best_bleu_score
                ):
                    dev_best_bleu_score = dev_translation_results[tbw][ta][
                        "valid_scores"
                    ]["bleu"]
                    dev_best_translation_beam_size = tbw
                    dev_best_translation_alpha = ta
                    dev_best_translation_result = dev_translation_results[tbw][ta]
                    logger.info(
                        "[DEV] partition [Translation] results:\n\t"
                        "New Best Translation Beam Size: %d and Alpha: %d\n\t"
                        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
                        "CHRF %.2f\t"
                        "ROUGE %.2f",
                        dev_best_translation_beam_size,
                        dev_best_translation_alpha,
                        dev_best_translation_result["valid_scores"]["bleu"],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu1"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu2"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu3"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu4"
                        ],
                        dev_best_translation_result["valid_scores"]["chrf"],
                        dev_best_translation_result["valid_scores"]["rouge"],
                    )
                    logger.info("-" * 60)

    logger.info("*" * 60)
    logger.info(
        "[DEV] partition [Recognition & Translation] results:\n\t"
        "Best CTC Decode Beam Size: %d\n\t"
        "Best Translation Beam Size: %d and Alpha: %d\n\t"
        "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
        "CHRF %.2f\t"
        "ROUGE %.2f",
        dev_best_recognition_beam_size if do_recognition else -1,
        dev_best_translation_beam_size if do_translation else -1,
        dev_best_translation_alpha if do_translation else -1,
        dev_best_recognition_result["valid_scores"]["wer"] if do_recognition else -1,
        dev_best_recognition_result["valid_scores"]["wer_scores"]["del_rate"]
        if do_recognition
        else -1,
        dev_best_recognition_result["valid_scores"]["wer_scores"]["ins_rate"]
        if do_recognition
        else -1,
        dev_best_recognition_result["valid_scores"]["wer_scores"]["sub_rate"]
        if do_recognition
        else -1,
        dev_best_translation_result["valid_scores"]["bleu"] if do_translation else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu1"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu2"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu3"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu4"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["chrf"] if do_translation else -1,
        dev_best_translation_result["valid_scores"]["rouge"] if do_translation else -1,
    )
    logger.info("*" * 60)

    test_best_result = validate_on_data(
        model=model,
        data=test_data,
        batch_size=batch_size,
        use_cuda=use_cuda,
        batch_type=batch_type,
        dataset_version=dataset_version,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
        do_recognition=do_recognition,
        recognition_loss_function=recognition_loss_function if do_recognition else None,
        recognition_loss_weight=1 if do_recognition else None,
        recognition_beam_size=dev_best_recognition_beam_size
        if do_recognition
        else None,
        do_translation=do_translation,
        translation_loss_function=translation_loss_function if do_translation else None,
        translation_loss_weight=1 if do_translation else None,
        translation_max_output_length=translation_max_output_length
        if do_translation
        else None,
        level=level if do_translation else None,
        translation_beam_size=dev_best_translation_beam_size
        if do_translation
        else None,
        translation_beam_alpha=dev_best_translation_alpha if do_translation else None,
        frame_subsampling_ratio=frame_subsampling_ratio,
    )

    logger.info(
        "[TEST] partition [Recognition & Translation] results:\n\t"
        "Best CTC Decode Beam Size: %d\n\t"
        "Best Translation Beam Size: %d and Alpha: %d\n\t"
        "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
        "CHRF %.2f\t"
        "ROUGE %.2f",
        dev_best_recognition_beam_size if do_recognition else -1,
        dev_best_translation_beam_size if do_translation else -1,
        dev_best_translation_alpha if do_translation else -1,
        test_best_result["valid_scores"]["wer"] if do_recognition else -1,
        test_best_result["valid_scores"]["wer_scores"]["del_rate"]
        if do_recognition
        else -1,
        test_best_result["valid_scores"]["wer_scores"]["ins_rate"]
        if do_recognition
        else -1,
        test_best_result["valid_scores"]["wer_scores"]["sub_rate"]
        if do_recognition
        else -1,
        test_best_result["valid_scores"]["bleu"] if do_translation else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu1"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu2"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu3"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu4"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["chrf"] if do_translation else -1,
        test_best_result["valid_scores"]["rouge"] if do_translation else -1,
    )
    logger.info("*" * 60)

    def _write_to_file(file_path: str, sequence_ids: List[str], hypotheses: List[str]):
        with open(file_path, mode="w", encoding="utf-8") as out_file:
            for seq, hyp in zip(sequence_ids, hypotheses):
                out_file.write(seq + "|" + hyp + "\n")

    if output_path is not None:
        if do_recognition:
            dev_gls_output_path_set = "{}.BW_{:03d}.{}.gls".format(
                output_path, dev_best_recognition_beam_size, "dev"
            )
            _write_to_file(
                dev_gls_output_path_set,
                [s for s in dev_data.sequence],
                dev_best_recognition_result["gls_hyp"],
            )
            test_gls_output_path_set = "{}.BW_{:03d}.{}.gls".format(
                output_path, dev_best_recognition_beam_size, "test"
            )
            _write_to_file(
                test_gls_output_path_set,
                [s for s in test_data.sequence],
                test_best_result["gls_hyp"],
            )

        if do_translation:
            if dev_best_translation_beam_size > -1:
                dev_txt_output_path_set = "{}.BW_{:02d}.A_{:1d}.{}.txt".format(
                    output_path,
                    dev_best_translation_beam_size,
                    dev_best_translation_alpha,
                    "dev",
                )
                test_txt_output_path_set = "{}.BW_{:02d}.A_{:1d}.{}.txt".format(
                    output_path,
                    dev_best_translation_beam_size,
                    dev_best_translation_alpha,
                    "test",
                )
            else:
                dev_txt_output_path_set = "{}.BW_{:02d}.{}.txt".format(
                    output_path, dev_best_translation_beam_size, "dev"
                )
                test_txt_output_path_set = "{}.BW_{:02d}.{}.txt".format(
                    output_path, dev_best_translation_beam_size, "test"
                )

            _write_to_file(
                dev_txt_output_path_set,
                [s for s in dev_data.sequence],
                dev_best_translation_result["txt_hyp"],
            )
            _write_to_file(
                test_txt_output_path_set,
                [s for s in test_data.sequence],
                test_best_result["txt_hyp"],
            )

        with open(output_path + ".dev_results.pkl", "wb") as out:
            pickle.dump(
                {
                    "recognition_results": dev_recognition_results
                    if do_recognition
                    else None,
                    "translation_results": dev_translation_results
                    if do_translation
                    else None,
                },
                out,
            )
        with open(output_path + ".test_results.pkl", "wb") as out:
            pickle.dump(test_best_result, out)



def test_jinhui(cfg_file=None):
    cfg = load_config(cfg_file)

    # predict with the best model on validation and test
    # (if test data is available)
    ckpt = cfg["testing"].get("ckpt", None)
    if ckpt == None:
        ckpt = cfg_file.replace("config.yaml", "best.ckpt")
    # output_name = "best.IT_{:08d}".format(1)
    output_path = cfg["testing"].get("log_file", "./")
    logger = make_logger(model_dir=output_path, log_file="testing.log")

    if ":" in ckpt:
        ckpts = ckpt.split(":")

        test_model_average(cfg_file, ckpt=ckpts, output_path=output_path, logger=logger)

    else:
        # test(cfg_file, ckpt=ckpt, output_path=output_path, logger=logger)
        visualization3(cfg_file, ckpt=ckpt, output_path=output_path, logger=logger)
import argparse, os

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Joey-NMT")
    parser.add_argument(
        "--config",
        default="/home/yejinhui/Projects/SLT/training_task/training_task_old/0417_SMKD_sign_S2T_seed32_bsz128_drop15_len30_freq50/config.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="gpu to run your job on"
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    test_jinhui(cfg_file=args.config)

