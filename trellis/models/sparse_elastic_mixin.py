from contextlib import contextmanager
from typing import *
import math
from ..modules import sparse as sp
from ..utils.elastic_utils import ElasticModuleMixin


class SparseTransformerElasticMixin(ElasticModuleMixin):
    def _get_input_size(self, x: sp.SparseTensor, *args, **kwargs):
        return x.feats.shape[0]
    
    @contextmanager
    def with_mem_ratio(self, mem_ratio=1.0):
        if mem_ratio == 1.0:
            yield 1.0
            return
        # num_blocks = len(self.blocks)
        # num_checkpoint_blocks = min(math.ceil((1 - mem_ratio) * num_blocks) + 1, num_blocks)
        # exact_mem_ratio = 1 - (num_checkpoint_blocks - 1) / num_blocks
        # for i in range(num_blocks):
        #     self.blocks[i].use_checkpoint = i < num_checkpoint_blocks
        # yield exact_mem_ratio
        # for i in range(num_blocks):
        #     self.blocks[i].use_checkpoint = False
        # all_blocks = self.blocks + self.cross_blocks
        all_blocks = self.blocks
        num_total_blocks = len(all_blocks)

        original_states = [block.use_checkpoint for block in all_blocks]
        try:
            # --- Apply new checkpointing strategy ---
            # Calculate the number of blocks to checkpoint based on the total number
            num_checkpoint_blocks = min(math.ceil((1 - mem_ratio) * num_total_blocks) + 1, num_total_blocks)
            
            # Calculate the exact memory ratio achieved with the discrete number of blocks
            exact_mem_ratio = 1 - (num_checkpoint_blocks - 1) / num_total_blocks if num_total_blocks > 1 else 1.0

            # Enable checkpointing for the determined number of blocks
            for i in range(num_total_blocks):
                all_blocks[i].use_checkpoint = (i < num_checkpoint_blocks)
            
            yield exact_mem_ratio

        finally:
            # --- Restore original states ---
            # This ensures the model is in its original state after the context manager exits.
            for i, block in enumerate(all_blocks):
                block.use_checkpoint = original_states[i]
