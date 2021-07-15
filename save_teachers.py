#!/usr/bin/env python3
import sys
import torch
import logging
import numpy as np
from tqdm.contrib import tqdm
import h5py
import speechbrain as sb
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.edit_distance import wer_details_for_batch

"""
"""

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
    def __init__(self, tea_modules_list=None, hparams=None, run_opts=None):
        super(ASR, self).__init__(
            modules=None,
            opt_class=None,
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=None,
        )

        # Initialize teacher parameters
        tea_modules_list_ = []
        for tea_modules in tea_modules_list:
            tea_modules_ = torch.nn.ModuleList(tea_modules)
            tea_modules_ = tea_modules_.to(self.device)
            tea_modules_list_.append(tea_modules_)
        self.tea_modules_list = tea_modules_list_

    def compute_forward_tea(self, batch):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        ids = batch.id
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        tokens, tokens_lens = batch.tokens
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Forward pass
        feats = self.hparams.compute_features(wavs)
        feats = self.hparams.normalize(feats, wav_lens)
        apply_softmax = torch.nn.Softmax(dim=-1)

        # run inference to each teacher model
        tea_dict_list = []
        for num in range(self.hparams.num_tea):
            tea_dict = {}
            self.tea_modules_list[num].eval()
            with torch.no_grad():
                x_tea = tea_enc_list[num](feats)
                ctc_logits_tea = tea_ctc_lin_list[num](x_tea)

                # output layer for ctc log-probabilities
                p_ctc_tea = self.hparams.log_softmax(
                    ctc_logits_tea / self.hparams.temperature
                )

                e_in_tea = tea_emb_list[num](tokens_bos)
                h_tea, _ = tea_dec_list[num](e_in_tea, x_tea, wav_lens)

                # output layer for seq2seq log-probabilities
                seq_logits_tea = tea_seq_lin_list[num](h_tea)
                p_seq_tea = apply_softmax(
                    seq_logits_tea / self.hparams.temperature
                )

                # WER from output layer of CTC
                sequence_ctc = sb.decoders.ctc_greedy_decode(
                    p_ctc_tea, wav_lens, blank_id=self.hparams.blank_index
                )
                predicted_words = self.tokenizer(sequence_ctc, task="decode_from_list")

                target_words = undo_padding(tokens, tokens_lens)
                target_words = self.tokenizer(target_words, task="decode_from_list")

                wer_stats_ctc = wer_details_for_batch(ids, target_words, predicted_words)

                wer_ctc_tea = []
                for item in wer_stats_ctc:
                    wer_ctc_tea.append(item["WER"])

                wer_ctc_tea = exclude_wer(wer_ctc_tea)
                wer_ctc_tea = np.expand_dims(wer_ctc_tea, axis=0)

                # WER from output layer of CE
                _, predictions = p_seq_tea.max(dim=-1)
                sequence_ce = sb.decoders.seq2seq.batch_filter_seq2seq_output(
                    predictions, eos_id=self.hparams.eos_index
                )
                predicted_words_ce = self.tokenizer(sequence_ce, task="decode_from_list")

                wer_stats_ce = wer_details_for_batch(ids, target_words, predicted_words_ce)

                wer_tea = []
                for item in wer_stats_ce:
                    wer_tea.append(item["WER"])

                wer_tea = exclude_wer(wer_tea)
                wer_tea = np.expand_dims(wer_tea, axis=0)

            # save the variables into dict
            tea_dict["p_ctc_tea"] = p_ctc_tea.cpu().numpy()
            tea_dict["p_seq_tea"] = p_seq_tea.cpu().numpy()
            tea_dict["wer_ctc_tea"] = wer_ctc_tea
            tea_dict["wer_tea"] = wer_tea
            tea_dict_list.append(tea_dict)

        return tea_dict_list

    def def_tea_name(self):
        # define teacher variable name
        tea_name = []
        for tea_num in range(self.hparams.num_tea):
            tea = "t{}".format(tea_num)
            tea_name.append(tea)
        return tea_name

    def fit_save(self, train_set, valid_set=None, test_set=None):
        data_sets = [train_set, valid_set, test_set]
        stage = self.hparams.stage
        tea_name = self.def_tea_name()

        # define output file name
        f_name = "/tea_infer_{}batch.hdf5".format(self.hparams.batch_size)
        f = h5py.File(self.hparams.output_folder + f_name, "w")
        for num in range(len(stage)):
            # create group for each set (train, valid, test).
            g_sets = f.create_group(stage[num])

            with tqdm(
                data_sets[num], initial=self.step, dynamic_ncols=True,
            ) as t:
                for batch in t:
                    self.step += 1
                    # create group for each batch
                    g_batch = g_sets.create_group(str(self.step))

                    # run inference to each teacher
                    tea_dict_list = self.compute_forward_tea(batch)

                    for tea_num in range(self.hparams.num_tea):
                        # create group for each teacher
                        g_tea = g_batch.create_group(tea_name[tea_num])
                        g_tea.create_dataset(
                            "p_ctc_tea",
                            data=tea_dict_list[tea_num]["p_ctc_tea"],
                        )
                        g_tea.create_dataset(
                            "p_seq_tea",
                            data=tea_dict_list[tea_num]["p_seq_tea"],
                        )
                        g_tea.create_dataset(
                            "wer_ctc_tea",
                            data=tea_dict_list[tea_num]["wer_ctc_tea"][0],
                        )
                        g_tea.create_dataset(
                            "wer_tea", data=tea_dict_list[tea_num]["wer_tea"][0]
                        )
            self.step = 0
        f.close()


def exclude_wer(wer):
    """
    This function is used to exclude the
    wer values which is more than 100.
    """
    wer_list = []
    for item in wer:
        if item > 100:
            item = 100
        wer_list.append(item)
    return np.array(wer_list)


# Define custom data procedure
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder},
    )

    # We also sort the validation data so it is faster to validate
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        if info.num_channels > 1:
            sig = torch.mean(sig, dim=1)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_data, tokenizer


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset preparation (parsing CommonVoice)
    from common_voice_prepare import prepare_common_voice  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Due to DDP, we do the preparation ONLY on the main python process
    run_on_main(
        prepare_common_voice,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "train_tsv_file": hparams["train_tsv_file"],
            "dev_tsv_file": hparams["dev_tsv_file"],
            "test_tsv_file": hparams["test_tsv_file"],
            "accented_letters": hparams["accented_letters"],
            "language": hparams["language"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data, test_data, tokenizer = dataio_prepare(hparams)

    # initialise teacher model variables
    tea_enc_list = []
    tea_emb_list = []
    tea_dec_list = []
    tea_ctc_lin_list = []
    tea_seq_lin_list = []
    for i in range(hparams["num_tea"]):
        exec ("tea_enc_list.append(hparams['tea{}_enc'])".format(i))
        exec ("tea_emb_list.append(hparams['tea{}_emb'])".format(i))
        exec ("tea_dec_list.append(hparams['tea{}_dec'])".format(i))
        exec ("tea_ctc_lin_list.append(hparams['tea{}_ctc_lin'])".format(i))
        exec ("tea_seq_lin_list.append(hparams['tea{}_seq_lin'])".format(i))

    # create ModuleList
    for i in range(hparams["num_tea"]):
        exec (
            "tea{}_modules = torch.nn.ModuleList([tea_enc_list[i], tea_emb_list[i], tea_dec_list[i], tea_ctc_lin_list[i], tea_seq_lin_list[i]])".format(
                i
            )
        )  # i denotes the index of teacher models

    tea_modules_list = []
    for i in range(hparams["num_tea"]):
        exec ("tea_modules_list.append(tea{}_modules)".format(i))

    # Trainer initialization
    asr_brain = ASR(
        tea_modules_list=tea_modules_list, hparams=hparams, run_opts=run_opts
    )

    # Adding objects to trainer.
    asr_brain.tokenizer = tokenizer

    # load pre-trained weights of teacher models
    with open(hparams["tea_models_dir"], "r") as f:
        enter_token = "\n"
        for i, path in enumerate(f.readlines()):
            print(i)
            exec (
                "tea{}_modules.load_state_dict(torch.load(path.strip(enter_token)))".format(
                    i
                )
            )

    # make dataloaders
    train_set = sb.dataio.dataloader.make_dataloader(
        train_data, **hparams["dataloader_options"]
    )
    valid_set = sb.dataio.dataloader.make_dataloader(
        valid_data, **hparams["dataloader_options"]
    )
    test_set = sb.dataio.dataloader.make_dataloader(
        test_data, **hparams["test_dataloader_options"]
    )

    # run inference and save results
    asr_brain.fit_save(train_set, valid_set, test_set)
