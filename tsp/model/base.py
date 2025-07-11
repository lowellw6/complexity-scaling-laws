import torch
import torch.nn as nn
import torch.nn.functional as F

from tsp.model.submodules import TspEncoder, PositionalEncoding, TspCritic
from tsp.utils import generate_padding_mask, reset_pads
from tsp.select import greedy_select
from tsp.datagen import SENTINEL


class TspDecoder(nn.Module):
    """
    Node encodings and selections are input as
    target and memory, respectively.

    Self-attention is applied to the node
    encodings like in the TspEncoder.

    For the encoder-decoder attention step,
    node encodings are the queries and
    the selected coordinates thus far are
    used to generate keys and values.

    This results in a decoder-stack output shape
    of (S, N, d), or (seq dim, batch dim, tsp dimensions).

    A linear layer then condenses the coordinate
    embedding dim 'd' and we perform a masked softmax
    over the sequence dim 'S' (number of nodes in TSP
    problem).
    """

    def __init__(self, dim_model=128, num_layers=3, tsp_dimensions=2):
        super().__init__()
        self.dim_model = dim_model
        self.num_layers = num_layers
        self.tsp_dimensions = tsp_dimensions

        self.embed = nn.Linear(tsp_dimensions, dim_model)
        self.pos_enc = PositionalEncoding(dim_model)

        self.to_dist = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.ReLU(),
            nn.Linear(dim_model, 1),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            dim_model, nhead=8, dim_feedforward=(4 * dim_model), dropout=0.0
        )

        self.trf_dec = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

    def forward(
        self, node_encodings, selections, output_mask, node_enc_padding_mask, eos_mask
    ):
        """
        Generate selection distribution for the next node using
        self-attention over node encodings and encoder-decoder
        attention over the coordinates selected so far.

        Args:
            node_encodings - encoded rep of problem coordinates, shape (N, S, d)
            selections - ordered problem coordinates selected so far, shape (N, ?, d)
            output_mask - boolean mask asserted at indices already selected, shape (N, S)
            node_enc_padding_mask - boolean mask asserted at indices of padding encodings, shape (N, S)
            eos_mask - boolean mask asserted at batch indices which passed EOS, shape (N,)

        Returns log_probs over next selection, shape (N, S).

        Note '?' is a wildcard of selections. For the first iteration of decoding
        this should just be size 1 for the start token.
        """
        select_pad_mask = generate_padding_mask(selections)

        node_encodings_t = torch.transpose(node_encodings, 0, 1)
        selections_t = torch.transpose(selections, 0, 1)
        output_mask_t = torch.transpose(output_mask, 0, 1)

        sel_input_sym = selections_t * 2 - 1  # [0, 1] --> [-1, 1]
        sel_input_emb = self.embed(sel_input_sym)
        sel_input_pe = self.pos_enc(sel_input_emb)

        dec_out = self.trf_dec(
            node_encodings_t,
            sel_input_pe,
            tgt_key_padding_mask=node_enc_padding_mask,
            memory_key_padding_mask=select_pad_mask,
        )

        logits = self.to_dist(dec_out).squeeze(-1)

        with torch.no_grad():
            logits[output_mask_t] = float("-inf")  # make re-selection impossible

            node_enc_padding_mask_t = torch.transpose(node_enc_padding_mask, 0, 1)
            logits[node_enc_padding_mask_t] = float("-inf")  # mask pad queries

        # make subsequent log_probs eos distributions uniform rather than all "nan"
        logits = reset_pads(torch.transpose(logits, 0, 1), eos_mask)

        return F.log_softmax(logits, dim=1)


class TspModel(nn.Module):
    def __init__(self, dim_model=128, num_enc_layers=3, num_dec_layers=3, tsp_dimensions=2):
        super().__init__()
        self.dim_model = dim_model
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.tsp_dimensions = tsp_dimensions

        self.encoder = TspEncoder(self.dim_model, num_layers=num_enc_layers, tsp_dimensions=tsp_dimensions)
        self.decoder = TspDecoder(self.dim_model, num_layers=num_dec_layers, tsp_dimensions=tsp_dimensions)

    def forward(self, problems, select_fn):
        selections, select_idxs, log_probs, _, _, _ = self._rollout(problems, select_fn)
        selections = selections[:, 1:]  # remove start token before returning

        return selections, select_idxs, log_probs

    def _rollout(self, problems, select_fn):
        """
        TODO
        """
        batch_size, problem_size, _ = problems.shape
        pad_mask = generate_padding_mask(problems)

        node_encodings = self.encoder(problems)

        selections = torch.zeros(batch_size, 1, self.tsp_dimensions).to(
            problems.device
        )  # zeros as start token
        select_idxs = torch.empty((batch_size, problem_size), dtype=torch.int64).to(
            problems.device
        )
        log_probs = torch.empty(batch_size, problem_size, problem_size).to(
            problems.device
        )
        output_mask = torch.zeros((batch_size, problem_size), dtype=torch.bool).to(
            problems.device
        )

        for step_idx in range(problem_size):
            selections, step_sel_idx, step_log_probs, output_mask = self.step(
                problems,
                node_encodings,
                selections,
                output_mask,
                pad_mask,
                select_fn,
                step_idx,
            )

            select_idxs[:, step_idx] = step_sel_idx  # save integer index of last choice
            log_probs[
                :, step_idx
            ] = step_log_probs  # save log prob dist from last choice

        return selections, select_idxs, log_probs, node_encodings, output_mask, pad_mask

    def step(
        self,
        problems,
        node_encodings,
        selections,
        output_mask,
        pad_mask,
        select_fn,
        step_idx,
    ):
        """
        TODO

        Detach next_sel --> don't want grads to propagate through past selections
        Though I think this is unnecessary since it depends on the problems and
        index tensors, both of which are usually leaf tensors (for the index tensor,
        because we use argmax or a sample).
        """
        problem_size = problems.shape[1]

        joint_invalid_mask = torch.logical_or(output_mask, pad_mask)
        eos_mask = torch.all(joint_invalid_mask, dim=1)  # (N,)

        log_probs = self.decoder(
            node_encodings, selections, output_mask, pad_mask, eos_mask
        )

        next_idx, next_sel = select_fn(
            problems, log_probs, step_idx
        )  # makes invalid selections after EOS, needs reset
        next_sel = reset_pads(next_sel, eos_mask, val=SENTINEL)
        next_idx = reset_pads(next_idx, eos_mask, val=int(SENTINEL))

        selections = torch.cat((selections, next_sel.detach()), dim=1)

        output_mask[~eos_mask] = torch.logical_or(
            output_mask[~eos_mask],
            F.one_hot(next_idx[~eos_mask], num_classes=problem_size),
        )

        return selections, next_idx, log_probs, output_mask


class TspAcModel(TspModel):
    """
    TSP actor-crtic model.
    Simply adds a value head over selections.
    """

    def __init__(
        self, dim_model=128, num_enc_layers=3, num_dec_layers=3, num_crt_layers=3, tsp_dimensions=2
    ):
        """Initialize actor model and decoupled critic model."""
        super().__init__(
            dim_model=dim_model,
            num_enc_layers=num_enc_layers,
            num_dec_layers=num_dec_layers,
            tsp_dimensions=tsp_dimensions
        )
        self.num_crt_layers = num_crt_layers

        self.critic = TspCritic(self.dim_model, num_layers=num_crt_layers, tsp_dimensions=tsp_dimensions)

    def forward(self, problems, select_fn):
        """
        Call TspModel to get solections and log probs.
        Then compute values for each selection sequence; values
        are based on all selections up to their sequence index
        (inclusive).
        """
        (
            selections,
            select_idxs,
            log_probs,
            node_encodings,
            _,
            node_enc_pad_mask,
        ) = self._rollout(problems, select_fn)

        select_pad_mask = generate_padding_mask(selections)

        values = self.critic(
            node_encodings, selections, node_enc_pad_mask, select_pad_mask
        )  # selections include start token

        selections = selections[:, 1:]  # remove start token before returning

        return selections, select_idxs, log_probs, values


class TspGreedyBaselineModel(nn.Module):
    """
    Actor-critic model where the critic is a seperate policy. 
    The baseline can be computed as the greedy rollout from this
    policy, which is updated only when the actor
    policy performs better on a held-out set.

    NOTE spur_critic() must be called to receive baseline rollouts
    """

    def __init__(self, dim_model=128, num_enc_layers=3, num_dec_layers=3, tsp_dimensions=2):
        super().__init__()
        self.dim_model = dim_model
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.tsp_dimensions = tsp_dimensions

        self.actor = TspModel(dim_model, num_enc_layers, num_dec_layers, tsp_dimensions)
        self.critic = TspModel(dim_model, num_enc_layers, num_dec_layers, tsp_dimensions)

        self.critic_active = False

    def forward(self, problems, select_fn):
        """
        NOTE defaults to returning empty tensor for critic selections (values replacement)
        To receive critic selections, call spur_critic() before the forward pass

        This is a bit hacky, but lets us avoid expensive, redundant critic forward
        passes during minibatching, while making minimal changes to existing code
        """
        selections, select_idxs, log_probs = self.actor(problems, select_fn)

        if self.critic_active:
            with torch.no_grad():
                self.critic.eval()
                crt_sel, _, _ = self.critic(problems, greedy_select)
            
            self.critic_active = False  # NOTE critic is deactivated automatically
        else:
            crt_sel = torch.zeros(0)

        return selections, select_idxs, log_probs, crt_sel

    def sync_baseline(self):
        self.critic.load_state_dict(self.actor.state_dict())

    def spur_critic(self):
        """Activate baseline policy critic selection for the next forward pass (autoregressive and computationally expensive)"""
        self.critic_active = True
