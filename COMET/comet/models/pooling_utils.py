# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch


def partial_average_pooling(
        tokens: torch.Tensor,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        padding_index: int,
) -> torch.Tensor:
    """Average pooling function for only the last sentence.
    :param tokens: Word ids [batch_size x seq_length]
    :param embeddings: Word embeddings [batch_size x seq_length x hidden_size]
    :param mask: Padding mask [batch_size x seq_length]
    :param padding_index: Padding value.
    """
    # the start index of the last sentence
    start_index = []
    new_mask = mask
    for i, sent in enumerate(tokens):
        # find all separator tokens in the sequence
        separators = (sent == 2).nonzero()
        if len(separators) > 1:
            # if there are more than one find where the last sentence starts
            ind = separators[-2].cpu().numpy().item()
            start_index.append(ind)
            new_mask[i, 1:ind+1] = 0
        else:
            start_index.append(0)
    wordemb = mask_fill_index(0.0, tokens, embeddings, start_index, padding_index)
    sentemb = torch.sum(wordemb, 1)
    # print(new_mask)
    sum_mask = new_mask.unsqueeze(-1).expand(embeddings.size()).float().sum(1)
    return sentemb / sum_mask


def average_pooling(
        tokens: torch.Tensor,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        padding_index: int,
) -> torch.Tensor:
    """Average pooling function.
    :param tokens: Word ids [batch_size x seq_length]
    :param embeddings: Word embeddings [batch_size x seq_length x hidden_size]
    :param mask: Padding mask [batch_size x seq_length]
    :param padding_index: Padding value.
    """
    wordemb = mask_fill(0.0, tokens, embeddings, padding_index)
    sentemb = torch.sum(wordemb, 1)
    sum_mask = mask.unsqueeze(-1).expand(embeddings.size()).float().sum(1)
    return sentemb / sum_mask


def max_pooling(
        tokens: torch.Tensor, embeddings: torch.Tensor, padding_index: int
) -> torch.Tensor:
    """Max pooling function.
    :param tokens: Word ids [batch_size x seq_length]
    :param embeddings: Word embeddings [batch_size x seq_length x hidden_size]
    :param padding_index: Padding value.
    """
    return mask_fill(float("-inf"), tokens, embeddings, padding_index).max(dim=1)[0]


def mask_fill(
        fill_value: float,
        tokens: torch.Tensor,
        embeddings: torch.Tensor,
        padding_index: int,
) -> torch.Tensor:
    """
    Function that masks embeddings representing padded elements.
    :param fill_value: the value to fill the embeddings belonging to padded tokens.
    :param tokens: The input sequences [bsz x seq_len].
    :param embeddings: word embeddings [bsz x seq_len x hiddens].
    :param padding_index: Index of the padding token.
    """
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)


def mask_fill_index(
        fill_value: float,
        tokens: torch.Tensor,
        embeddings: torch.Tensor,
        start_index: list,
        padding_index: int,
) -> torch.Tensor:
    """
    Function that masks embeddings representing padded elements and previous sentences for multi-sentence sequences.
    :param fill_value: the value to fill the embeddings belonging to padded tokens.
    :param tokens: The input sequences [bsz x seq_len].
    :param embeddings: word embeddings [bsz x seq_len x hiddens].
    :param padding_index: Index of the padding token.
    """
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    padding_maks2 = torch.zeros(tokens.shape, dtype=torch.bool, device=padding_mask.device)
    for i, start in enumerate(start_index):
        padding_maks2[i, 1: start+1] = True
    padding_mask = torch.logical_or(padding_mask, padding_maks2.unsqueeze(-1))
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)
