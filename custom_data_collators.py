import os.path
import random
import re
from dataclasses import dataclass
from typing import List, Union, Dict, Set, Tuple

import torch

from transformers import DataCollatorForWholeWordMask, BatchEncoding
from transformers.data.data_collator import _collate_batch, tolist


@dataclass
class DataCollatorForPMIMasking(DataCollatorForWholeWordMask):
    """
    Data collator used for language modeling.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    ngrams_vocab_set: Set[str] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.mlm
        assert self.ngrams_vocab_set is not None and isinstance(self.ngrams_vocab_set, set) and len(self.ngrams_vocab_set) > 0

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        batch_input = _collate_batch(input_ids, self.tokenizer)

        mask_labels = []
        indexer = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            # ref tokens is a list of strings with the original tokens (including the [SEP], [CLS] etc. However, it contains splitted tokens with prefix of ## for tokens that are attached to the previous tokens
            mask_labels_for_sample, indexer_for_sample = self._pmi_word_mask(ref_tokens)
            mask_labels.append(mask_labels_for_sample)
            indexer.append(indexer_for_sample)
        batch_mask = _collate_batch(mask_labels, self.tokenizer)
        batch_indexer = _collate_batch(indexer, self.tokenizer)
        inputs, labels = self.mask_tokens(batch_input, batch_mask, batch_indexer)
        return {"input_ids": inputs, "labels": labels}

    def _pmi_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        whole_words_indexes = []
        whole_words_lists = [[]]
        whole_words = whole_words_lists[0]
        for (i, token) in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                whole_words_lists.append([])  # to separate parts as we don't want them to be considered part of an ngram
                whole_words = whole_words_lists[-1]
                continue
            # now, we mark the indices of token that start a whole word that can be masked
            if len(whole_words_indexes) >= 1 and token.startswith("##"):
                whole_words_indexes[-1].append(i)
                whole_words[-1] = whole_words[-1] + token[2:]
            else:
                whole_words_indexes.append([i])
                whole_words.append(token)

        offset = 0
        covered_indices = set()
        segments_to_merge = []
        for whole_words in whole_words_lists:
            if len(whole_words) == 0:
                continue
            added_offset = len(whole_words)
            whole_words = ' '.join(whole_words)
            start_inds = [0] + [m.start()+1 for m in re.finditer(' ', whole_words)] + [len(whole_words)+1]
            for n_grams in range(5, 1, -1):
                self._merge_segments(offset, n_grams, covered_indices, whole_words, start_inds, segments_to_merge)
            offset += added_offset
        segments_to_merge.extend([i] for i in set(range(len(whole_words_indexes))).difference(covered_indices))

        candidates = []
        for seg_to_merge in segments_to_merge:
            candidates.append(sum([whole_words_indexes[i] for i in seg_to_merge], []))

        random.shuffle(candidates)  # whole_words_indexes is a list of lists of ints, each likst is the indices part of the whole word to be masked, i.e. the segment to be considered for masking
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        # aux list that index every token to the parent segment index to make sure later the entire segment is masked correctly
        indexer = list(range(len(input_tokens)))
        covered_indexes = set()
        mask_labels = [0] * len(input_tokens)  # list of 0/1 in the length of the input tokens, 1 means the token should be masked
        for index_set in candidates:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidates.
            if len(covered_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:  # not sure how is it possible, the sets should be disjoint
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            head_index = min(index_set)
            mask_labels[head_index] = 1  # we will later draw only out of the heads of each segment
            for index in index_set:
                covered_indexes.add(index)
                indexer[index] = head_index

        return mask_labels, indexer

    def _merge_segments(self, offset, n_grams, covered_indices, whole_words, start_inds, segments_to_merge):
        possible_merges = []
        for i in range(len(start_inds)-n_grams):
            segment = whole_words[start_inds[i]:start_inds[i+n_grams]-1]
            if segment in self.ngrams_vocab_set:
                possible_merges.append(list(range(offset+i, offset+i+n_grams)))

        random.shuffle(possible_merges)
        for seg_inds in possible_merges:
            if len(set(seg_inds).intersection(covered_indices)) > 0:
                continue
            covered_indices.update(seg_inds)
            segments_to_merge.append(seg_inds)

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor, indexer: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()


        replace_strategy = torch.zeros_like(inputs)

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        replace_strategy[indices_replaced] = 1

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        replace_strategy[indices_random] = 2

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        # now we fix each segment to have the same strategy (0=Orig, 1=Mask, 2=Random), making sure that random/orig will be different
        # between tokens in the same segment
        replace_strategy = torch.gather(replace_strategy, 1, indexer)
        # we need to correct the labels of the non-head indices of the segments as well
        masked_indices = torch.gather(masked_indices, 1, indexer)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        inputs[replace_strategy == 1] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[replace_strategy == 2] = random_words[replace_strategy == 2]

        return inputs, labels

