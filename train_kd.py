#!/usr/bin/env python3
import sys
import torch
import logging
import h5py
import speechbrain as sb
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main

import time
from enum import Enum, auto
from torch.utils.data import DataLoader
from tqdm.contrib import tqdm

logger = logging.getLogger(__name__)

class Stage(Enum):
    """Simple enum to track stage of experiments."""

    TRAIN = auto()
    VALID = auto()
    TEST = auto()

# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Forward pass
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        ## Add augmentation if specified
        if stage == Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)

        x = self.modules.enc(feats.detach())
        e_in = self.modules.emb(tokens_bos)  # y_in bos + tokens
        h, _ = self.modules.dec(e_in, x, wav_lens)
        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        if stage == Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                # Output layer for ctc log-probabilities
                logits = self.modules.ctc_lin(x)
                p_ctc = self.hparams.log_softmax(logits)
                return p_ctc, p_seq, wav_lens
            else:
                return p_seq, wav_lens
        else:
            p_tokens, scores = self.hparams.beam_searcher(x, wav_lens)
            return p_seq, wav_lens, p_tokens

    def def_tea_name(self):
        # define teacher variable name
        tea_name = []
        for tea_num in range(self.hparams.num_tea):
            tea = "t{}".format(tea_num)
            tea_name.append(tea)
        return tea_name

    def re_format(self, data_dict):
        item_tea_list = [None, None, None, None]
        tea_name = self.def_tea_name()
        for tea_num in range(self.hparams.num_tea):
            for i in range(4):
                item_tea = data_dict[str(self.step)][tea_name[tea_num]][
                    self.hparams.tea_keys[i]
                ][()]

                if self.hparams.tea_keys[i].startswith("wer"):
                    item_tea = torch.tensor(item_tea)
                else:
                    item_tea = torch.from_numpy(item_tea)

                item_tea = item_tea.to(self.device)
                item_tea = torch.unsqueeze(item_tea, 0)
                if tea_num == 0:
                    item_tea_list[i] = item_tea
                else:
                    item_tea_list[i] = torch.cat(
                        [item_tea_list[i], item_tea], 0
                    )
        return item_tea_list

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        current_epoch = self.hparams.epoch_counter.current
        if stage == Stage.TRAIN:
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                p_ctc, p_seq, wav_lens = predictions
            else:
                p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        loss_seq_nor = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

        # if stage == sb.Stage.TRAIN:

        # Add ctc loss if necessary
        if (
            stage == Stage.TRAIN
            and current_epoch <= self.hparams.number_of_ctc_epochs
        ):
            loss_ctc_nor = self.hparams.ctc_cost(
                p_ctc, tokens, wav_lens, tokens_lens
            )

        # load teacher inference results
        # data_dict = (
        #     self.train_dict
        #     if stage == sb.Stage.TRAIN
        #     else self.valid_dict
        #     if stage == sb.Stage.VALID
        #     else self.test_dict
        # )
        data_dict = (
            self.train_dict
            if stage == Stage.TRAIN
            else self.valid_dict
            if stage == Stage.VALID
            else self.test_dict
        )

        item_tea_list = self.re_format(data_dict)
        p_ctc_tea, p_seq_tea, wer_ctc_tea, wer_tea = [
            item for item in item_tea_list
        ]

        # Stategy "average": average losses of teachers when doing distillation.
        # Stategy "top-1": choosing the best teacher based on WER.
        # Stategy "top-k": choosing all the best teacher based on WER if they have equal WER.
        # Stategy "weighted": assigning weights to teachers based on WER.
        if self.hparams.strategy == "top-1":
            # tea_ce for kd
            wer_scores, indx = torch.min(wer_tea, dim=0)
            indx = list(indx.cpu().numpy())

            # select the best teacher for each sentence
            tea_seq2seq_pout = None
            for stn_indx, tea_indx in enumerate(indx):
                s2s_one = p_seq_tea[tea_indx][stn_indx]
                s2s_one = torch.unsqueeze(s2s_one, 0)
                if stn_indx == 0:
                    tea_seq2seq_pout = s2s_one
                else:
                    tea_seq2seq_pout = torch.cat([tea_seq2seq_pout, s2s_one], 0)

        apply_softmax = torch.nn.Softmax(dim=0)

        if (
            self.hparams.strategy == "top-1"
            or self.hparams.strategy == "weighted"
            or self.hparams.strategy == "top-k"
        ):
            if (
                    stage == Stage.TRAIN
                    and current_epoch <= self.hparams.number_of_ctc_epochs
            ):
                # mean wer for ctc
                tea_wer_ctc_mean = wer_ctc_tea.mean(1)
                tea_acc_main = 100 - tea_wer_ctc_mean

                # normalise weights via Softmax function
                tea_acc_softmax = apply_softmax(tea_acc_main)

        if self.hparams.strategy == "weighted":
            # mean wer for ce
            tea_wer_mean = wer_tea.mean(1)
            tea_acc_ce_main = 100 - tea_wer_mean

            # normalise weights via Softmax function
            tea_acc_ce_softmax = apply_softmax(tea_acc_ce_main)

        if self.hparams.strategy == "top-k":
            p_seq_tea_new = None
            p_seq_new = None
            rel_length_new = None
            for stn_num in range(wer_tea.shape[1]):
                wer_one_stn = wer_tea[:, stn_num]
                min_wer_one_stn = torch.min(wer_one_stn)
                min_wer_indx_list = []
                for tea_indx, wer_ in enumerate(wer_one_stn):
                    if wer_ == min_wer_one_stn:
                        min_wer_indx_list.append(tea_indx)

                p_seq_tea_one_list = None
                for i in range(len(min_wer_indx_list)):
                    p_seq_tea_one = p_seq_tea[min_wer_indx_list[i], stn_num]
                    p_seq_tea_one = torch.unsqueeze(p_seq_tea_one, 0)
                    if i == 0:
                        p_seq_tea_one_list = p_seq_tea_one
                    else:
                        p_seq_tea_one_list = torch.cat([p_seq_tea_one_list, p_seq_tea_one])

                p_seq_rep = None
                for i in range(len(min_wer_indx_list)):
                    p_seq_one = torch.unsqueeze(p_seq[stn_num], 0)
                    if i == 0:
                        p_seq_rep = p_seq_one
                    else:
                        p_seq_rep = torch.cat([p_seq_rep, p_seq_one])

                rel_length_rep = None
                for i in range(len(min_wer_indx_list)):
                    rel_length_one = torch.unsqueeze(tokens_eos_lens[stn_num], 0)
                    if i == 0:
                        rel_length_rep = rel_length_one
                    else:
                        rel_length_rep = torch.cat([rel_length_rep, rel_length_one])

                if stn_num == 0:
                    p_seq_tea_new = p_seq_tea_one_list
                    p_seq_new = p_seq_rep
                    rel_length_new = rel_length_rep
                else:
                    p_seq_tea_new = torch.cat([p_seq_tea_new, p_seq_tea_one_list])
                    p_seq_new = torch.cat([p_seq_new, p_seq_rep])
                    rel_length_new = torch.cat([rel_length_new, rel_length_rep])

        # kd loss
        ctc_loss_list = None
        ce_loss_list = None
        for tea_num in range(self.hparams.num_tea):
            if (
                    stage == Stage.TRAIN
                    and current_epoch <= self.hparams.number_of_ctc_epochs
            ):
                # ctc
                p_ctc_tea_one = p_ctc_tea[tea_num]
                # calculate CTC distillation loss of one teacher
                loss_ctc_one = self.hparams.ctc_cost_kd(
                    p_ctc, p_ctc_tea_one, wav_lens, device=self.device
                )
                loss_ctc_one = torch.unsqueeze(loss_ctc_one, 0)
                if tea_num == 0:
                    ctc_loss_list = loss_ctc_one
                else:
                    ctc_loss_list = torch.cat([ctc_loss_list, loss_ctc_one])

            # ce
            p_seq_tea_one = p_seq_tea[tea_num]
            # calculate CE distillation loss of one teacher
            loss_seq_one = self.hparams.seq_cost_kd(
                p_seq, p_seq_tea_one, tokens_eos_lens
            )
            loss_seq_one = torch.unsqueeze(loss_seq_one, 0)
            if tea_num == 0:
                ce_loss_list = loss_seq_one
            else:
                ce_loss_list = torch.cat([ce_loss_list, loss_seq_one])

        # kd loss
        if self.hparams.strategy == "average":
            # get average value of losses from all teachers (CTC and CE loss)
            if (
                    stage == Stage.TRAIN
                    and current_epoch <= self.hparams.number_of_ctc_epochs
            ):
                ctc_loss_kd = ctc_loss_list.mean(0)

            seq2seq_loss_kd = ce_loss_list.mean(0)
        else:
            # assign weights to different teachers (CTC loss)
            if (
                    stage == Stage.TRAIN
                    and current_epoch <= self.hparams.number_of_ctc_epochs
            ):
                ctc_loss_kd = (tea_acc_softmax * ctc_loss_list).sum(0)

            if self.hparams.strategy == "top-1":
                # only use the best teacher to compute CE loss
                seq2seq_loss_kd = self.hparams.seq_cost_kd(
                    p_seq, tea_seq2seq_pout, tokens_eos_lens
                )
            if self.hparams.strategy == "weighted":
                # assign weights to different teachers (CE loss)
                seq2seq_loss_kd = (tea_acc_ce_softmax * ce_loss_list).sum(0)
            if self.hparams.strategy == "top-k":
                seq2seq_loss_kd = self.hparams.seq_cost_kd(p_seq_new, p_seq_tea_new, rel_length_new)

        # total loss
        # combine normal supervised training
        loss_seq = (
                self.hparams.temperature
                * self.hparams.temperature
                * self.hparams.alpha
                * seq2seq_loss_kd
                + (1 - self.hparams.alpha) * loss_seq_nor
        )

        if self.hparams.ceonly:
            loss = loss_seq
        else:
            if (
                    stage == Stage.TRAIN
                    and current_epoch <= self.hparams.number_of_ctc_epochs
            ):
                loss_ctc = (
                        self.hparams.temperature
                        * self.hparams.temperature
                        * self.hparams.alpha
                        * ctc_loss_kd
                        + (1 - self.hparams.alpha) * loss_ctc_nor
                )

                loss = (
                        self.hparams.ctc_weight * loss_ctc
                        + (1 - self.hparams.ctc_weight) * loss_seq
                )
            else:
                loss = loss_seq

        # else:
        #     loss = loss_seq_nor


        if stage != Stage.TRAIN:
            # Decode token terms to words
            predicted_words = self.tokenizer(
                predicted_tokens, task="decode_from_list"
            )

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of multiple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:

        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``update_average()``

        If the initialization was done with distributed_count > 0 and the
        distributed_backend is ddp, this will generally handle multiprocess
        logic, like splitting the training data into subsets for each device and
        only saving a checkpoint on the main process.

        Arguments
        ---------
        epoch_counter : iterable
            Each call should return an integer indicating the epoch count.
        train_set : Dataset, DataLoader
            A set of data to use for training. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        valid_set : Dataset, DataLoader
            A set of data to use for validation. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        train_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the train_loader
            (if train_set is a Dataset, not DataLoader).
            E.G. batch_size, num_workers.
            DataLoader kwargs are all valid.
        valid_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the valid_loader
            (if valid_set is a Dataset, not DataLoader).
            E.g., batch_size, num_workers.
            DataLoader kwargs are all valid.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        """

        if not isinstance(train_set, DataLoader):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not isinstance(valid_set, DataLoader):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Iterate epochs
        for epoch in epoch_counter:

            # Training stage
            self.on_stage_start(Stage.TRAIN, epoch)
            self.modules.train()

            # Reset nonfinite count to 0 each epoch
            self.nonfinite_count = 0

            if self.train_sampler is not None and hasattr(
                self.train_sampler, "set_epoch"
            ):
                self.train_sampler.set_epoch(epoch)

            # Time since last intra-epoch checkpoint
            last_ckpt_time = time.time()

            # Only show progressbar if requested and main_process
            enable = progressbar and sb.utils.distributed.if_main_process()
            with tqdm(
                train_set,
                initial=self.step,
                dynamic_ncols=True,
                disable=not enable,
            ) as t:
                for batch in t:
                    self.step += 1
                    loss = self.fit_batch(batch)
                    self.avg_train_loss = self.update_average(
                        loss, self.avg_train_loss
                    )
                    t.set_postfix(train_loss=self.avg_train_loss)

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                    if (
                        self.checkpointer is not None
                        and self.ckpt_interval_minutes > 0
                        and time.time() - last_ckpt_time
                        >= self.ckpt_interval_minutes * 60.0
                    ):
                        run_on_main(self._save_intra_epoch_ckpt)
                        last_ckpt_time = time.time()

            # Run train "on_stage_end" on all processes
            self.on_stage_end(Stage.TRAIN, self.avg_train_loss, epoch)
            self.avg_train_loss = 0.0
            self.step = 0

            # Validation stage
            if valid_set is not None:
                self.on_stage_start(Stage.VALID, epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(
                        valid_set, dynamic_ncols=True, disable=not enable
                    ):
                        self.step += 1
                        loss = self.evaluate_batch(batch, stage=Stage.VALID)
                        avg_valid_loss = self.update_average(
                            loss, avg_valid_loss
                        )

                        # Debug mode only runs a few batches
                        if self.debug and self.step == self.debug_batches:
                            break

                    # Only run validation "on_stage_end" on main process
                    self.step = 0
                    run_on_main(
                        self.on_stage_end,
                        args=[Stage.VALID, avg_valid_loss, epoch],
                    )

            # Debug mode only runs a few epochs
            if self.debug and epoch == self.debug_epochs:
                break

    def evaluate(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).

        Returns
        -------
        average test loss
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not isinstance(test_set, DataLoader):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1
                loss = self.evaluate_batch(batch, stage=Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            # Only run evaluation "on_stage_end" on main process
            run_on_main(
                self.on_stage_end, args=[Stage.TEST, avg_test_loss, None]
            )
        self.step = 0


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


def load_teachers(hparams):
    """
    Load results of inference of teacher models stored on disk.
    Note: Run experiment_save_teachers.py beforehand to generate .hdf5 files.
    """
    path = hparams["tea_infer_dir"] + "/tea_infer_{}batch.hdf5".format(
        hparams["batch_size"]
    )
    f = h5py.File(path, "r")
    train_dict = f["train"]
    valid_dict = f["valid"]
    test_dict = f["test"]

    return train_dict, valid_dict, test_dict


def st_load(hparams, asr_brain):
    """
    load pre-trained student model and remove decoder layer.
    """
    print("loading pre-trained student model...")
    chpt_path = hparams["pretrain_st_dir"] + "/model.ckpt"
    weight_dict = torch.load(chpt_path)
    # del the decoder layer
    key_list = []
    for k in weight_dict.keys():
        key_list.append(k)
    for k in key_list:
        if not k.startswith("0"):
            del weight_dict[k]

    # loading weights
    asr_brain.hparams.model.load_state_dict(weight_dict, strict=False)


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

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        opt_class=hparams["opt_class"],
        checkpointer=hparams["checkpointer"],
    )

    # Adding objects to trainer.
    asr_brain.tokenizer = tokenizer

    # load teacher models
    train_dict, valid_dict, test_dict = load_teachers(hparams)
    asr_brain.train_dict = train_dict
    asr_brain.valid_dict = valid_dict
    asr_brain.test_dict = test_dict

    if hparams["pretrain"]:
        # load pre-trained student model except last layer
        if hparams["epoch_counter"].current == 0:
            st_load(hparams, asr_brain)

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Test
    asr_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test.txt"
    asr_brain.evaluate(
        test_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
