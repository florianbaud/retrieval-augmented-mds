from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter


class FeedForwardLayer(nn.Module):

    def __init__(self, embed_dim, ff_embed_dim, dropout):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_embed_dim)
        self.fc2 = nn.Linear(ff_embed_dim, embed_dim)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.gelu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0., weights_dropout: bool = True):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads  # 1
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * \
            num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.weights_dropout = weights_dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        need_weights: str = None,
        attn_bias: torch.Tensor = None,
    ) -> List[torch.Tensor]:
        """ Input shape: Time x Batch x Channel
            key_padding_mask: Time x batch
            attn_mask:  tgt_len x src_len
        """
        # query shape = (seq_len, batch_size, hidden_size)
        # key,value shape = (seq_len, batch_size*topk, hidden_size)
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # query shape = (batch_size, seq_len, hidden_size)
        # key,value shape = (batch_size, seq_len*topk, hidden_size)

        src_len = k.size(1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        # attn shape = (batch_size, q_seq_len, k_seq_len*topk)

        assert list(attn_weights.size()) == [
            bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            attn_bias = attn_bias.transpose(0, 1).unsqueeze(1).unsqueeze(2)
            # attn_bias = attn_bias.unsqueeze(1).unsqueeze(1)
            # attn bias shape = (batch_size, 1, 1, k_seq_len*topk)
            attn_weights = attn_weights + attn_bias
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        # fill_value = torch.tensor(torch.finfo(query.dtype).min)
        fill_value = float('-inf')

        if attn_mask is not None:
            attn_weights.masked_fill_(
                attn_mask.unsqueeze(0),
                fill_value
            )

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            key_padding_mask_ = key_padding_mask.transpose(
                0, 1).unsqueeze(1).unsqueeze(2)
            attn_weights.masked_fill_(
                mask=key_padding_mask_,
                value=fill_value,
            )
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.weights_dropout:
            attn_weights = F.dropout(
                attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        if not self.weights_dropout:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        assert list(attn.size()) == [
            bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # maximum attention weight over heads
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            if need_weights == 'max':
                attn_weights, _ = attn_weights.max(dim=1)
            elif need_weights == "one":
                attn_weights = attn_weights[:, 0, :, :]
            else:
                assert False, "need weights?"
            attn_weights = attn_weights.transpose(0, 1)
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


class CopyTokenDecoder(nn.Module):
    def __init__(self, vocabs, tgt_embed, label_smoothing, embed_dim, ff_embed_dim, dropout):
        super(CopyTokenDecoder, self).__init__()
        self.output_projection = nn.Linear(
            tgt_embed.weight.shape[1],
            tgt_embed.weight.shape[0],
            bias=False,
        )
        self.alignment_layer = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=1,
            dropout=dropout,
            weights_dropout=False,
        )
        self.alignment_layer_norm = nn.LayerNorm(embed_dim)
        self.ff_layer = FeedForwardLayer(embed_dim, ff_embed_dim, dropout)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)
        self.diverter = nn.Linear(2*embed_dim, 2)
        self.output_projection.weight = tgt_embed.weight
        self.vocabs = vocabs
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.diverter.weight, std=0.02)
        nn.init.constant_(self.diverter.bias, 0.)

    def forward(
            self,
            outs: torch.Tensor,
            mem: torch.Tensor,
            mem_mask: torch.Tensor,
            mem_bias: torch.Tensor,
            copy_seq: torch.Tensor,
            data,
            work=False,
    ) -> torch.Tensor:
        ####################
        ### Transformers ###
        attn_mask = None
        attn, alignment_weight = self.alignment_layer(
            outs,
            mem,
            mem,
            key_padding_mask=mem_mask,
            need_weights='one',
            attn_mask=attn_mask,
            attn_bias=mem_bias,
        )
        # attn => target_n-1 + source + memory signals
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        outs = self.alignment_layer_norm(outs + attn)
        outs = self.ff_layer(outs)
        outs = F.dropout(outs, p=self.dropout, training=self.training)
        outs = self.ff_layer_norm(outs)
        ### Transformers ###
        ####################

        #######################
        ### Gates computing ###
        attn_normalized = self.alignment_layer_norm(attn)
        gates = F.softmax(
            self.diverter(
                torch.cat([outs, attn_normalized], -1)
            ), -1)

        gen_gate, copy_gate = gates.chunk(2, dim=-1)
        ### Gates computing ###
        #######################

        seq_len, bsz, _ = outs.size()
        probs = gen_gate * F.softmax(self.output_projection(outs), -1)

        # copy_seq: src_len x bsz
        # copy_gate: tgt_len x bsz
        # alignment_weight: tgt_len x bsz x src_len
        # index: tgt_len x bsz
        index = copy_seq.transpose(0, 1).contiguous().view(
            1, bsz, -1).expand(seq_len, -1, -1)
        # -> tgt_len x bsz x src_len
        copy_probs = (copy_gate * alignment_weight).view(seq_len, bsz, -1)
        # -> tgt_len x bsz x src_len
        probs = probs.scatter_add_(-1, index, copy_probs)
        # lprobs = torch.log(probs + 1e-12)
        # Increase eps because of half precision training
        # if probs.isnan().any().item():
        #     print("Probs", probs)
        lprobs = torch.log(probs + 1e-7)
        # print(probs.shape)
        # lprobs = F.log_softmax(probs + 1e-7, -1)
        # print(lprobs)
        # lprobs = probs

        # exit()
        return lprobs
        # if work:
        #     return lprobs
        # loss, nll_loss = label_smoothed_nll_loss(
        #     lprobs, data['tgt_tokens_out'], self.label_smoothing, ignore_index=self.vocabs['tgt'].padding_idx, sum=True)
        # top1 = torch.argmax(lprobs, -1)
        # acc = torch.eq(top1, data['tgt_tokens_out']).float().sum().item()
        # loss = loss / data['tgt_num_tokens']
        # return loss, acc
