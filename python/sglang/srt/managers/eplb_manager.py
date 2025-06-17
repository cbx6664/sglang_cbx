import logging
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, List

import torch.cuda

from sglang.srt.managers.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.managers.expert_location import ExpertLocationMetadata

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

from lb_analysis.analyzer import calculate_gpu_loads, plot_hotness_heatmap
from lb_analysis.visualization import LoadBalanceVisualizer

logger = logging.getLogger(__name__)


class EPLBManager:
    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()
        self._model_runner = model_runner
        self._server_args = model_runner.server_args
        self._rebalance_layers_per_chunk = (
            self._server_args.eplb_rebalance_layers_per_chunk
        )
        self._rebalance_num_iterations = self._server_args.eplb_rebalance_num_iterations

        # Otherwise, the circular buffer will contain stale data. If the case is needed, it can be implemented.
        assert (
            self._server_args.eplb_rebalance_num_iterations
            >= self._server_args.expert_distribution_recorder_buffer_size
        ), "eplb_rebalance_num_iterations must be greater than expert_distribution_recorder_buffer_size"

        if not get_global_expert_distribution_recorder().recording:
            get_global_expert_distribution_recorder().start_record()

        logger.info(
            f"[EPLBManager] system started, will rebalance per {self._rebalance_num_iterations} iterations."
        )

        # Add rebalance counter for visualization folder naming
        self._rebalance_counter = 0
        self._main_generator = self._entrypoint()

    def on_forward_pass_end(self):
        next(self._main_generator)

    # can be more complex if needed
    def _entrypoint(self):
        while True:
            for _ in range(self._rebalance_num_iterations):
                yield

            yield from self.rebalance()

    def rebalance(self):
        self._rebalance_counter += 1
        logger.info(f"[EPLBManager] rebalance start (iteration {self._rebalance_counter})")

        enable_timing = self._rebalance_layers_per_chunk is None

        if enable_timing:
            torch.cuda.synchronize()
            time_start = time.time()

        logical_count = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )["logical_count"]
        
        # Sum up the logical counts and print
        summed_logical_count = logical_count.sum(dim=0)  # Shape: [num_layer, num_experts]
        if self._model_runner.tp_rank == 0:  # Only print on rank 0
            logger.info(f"[EPLBManager] Expert distribution data shape: {summed_logical_count.shape}")
            torch.set_printoptions(threshold=float('inf'))
            logger.info(f"[EPLBManager] Expert distribution data content (summed across {self._rebalance_num_iterations} forward passes):\n{summed_logical_count}")
            
            # Call lb_analysis visualization functions
            if os.getenv("PLOT_TOKEN_DISTRIBUTION", "false").lower() in ("true", "1", "yes"):
                try:
                    self._visualize_expert_distribution(summed_logical_count, self._rebalance_counter)
                except Exception as e:
                    logger.error(f"[EPLBManager] Visualization failed: {e}")

        expert_location_metadata = ExpertLocationMetadata.init_by_eplb(
            self._server_args, self._model_runner.model_config, logical_count
        )

        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        for chunk_index, update_layer_ids in enumerate(update_layer_ids_chunks):
            if len(update_layer_ids_chunks) > 1:
                yield
            self._model_runner.update_expert_location(
                expert_location_metadata,
                update_layer_ids=update_layer_ids,
            )

        msg = f"[EPLBManager] rebalance end"
        if enable_timing:
            torch.cuda.synchronize()
            time_end = time.time()
            msg += f" time={time_end - time_start:.3f}s"
        logger.info(msg)

    def _compute_update_layer_ids_chunks(self) -> List[List[int]]:
        all_layer_ids = sorted(
            list(self._model_runner.model.routed_experts_weights_of_layer.keys())
        )
        chunk_size = self._rebalance_layers_per_chunk or 1000000
        return list(_chunk_list(all_layer_ids, chunk_size=chunk_size))

    def _visualize_expert_distribution(self, expert_token_counts: torch.Tensor, rebalance_iteration: int):
        """
        Visualize expert distribution using lb_analysis functions.
        
        Args:
            expert_token_counts: Tensor of shape [num_layers, num_experts] containing token counts
            rebalance_iteration: Current rebalance iteration number
        """
        logger.info(f"[EPLBManager] Starting visualization of expert distribution (iteration {rebalance_iteration})")
        
        # Create output directory from environment variable or use default
        plot_dir = Path(os.getenv("PLOT_DIR", "/tmp/eplb_visualization"))
        plot_dir.mkdir(parents=True, exist_ok=True)
        # Create subfolder for this rebalance iteration
        subfolder_name = f"eplb-rebalance-{rebalance_iteration}"
        output_dir = Path(plot_dir) / subfolder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[EPLBManager] Saving visualizations to: {output_dir}")
        
        num_layers, num_experts = expert_token_counts.shape
        logger.info(f"[EPLBManager] Visualizing {num_layers} layers with {num_experts} experts each")
        
        try:
            # 1. Expert hotness heatmap
            heatmap_path = output_dir / f"expert_hotness_heatmap_{time.strftime('%Y%m%d_%H%M%S')}.png"
            plot_hotness_heatmap(
                expert_token_counts,
                str(heatmap_path),
                title="Expert Token Distribution Heatmap"
            )
            logger.info(f"[EPLBManager] Saved expert hotness heatmap to: {heatmap_path}")
            
            # 2. GPU load analysis 
            num_gpus = getattr(self._server_args, 'tp_size')
            if num_experts % num_gpus == 0:
                visualizer = LoadBalanceVisualizer(output_dir)
                gpu_loads = calculate_gpu_loads(expert_token_counts, num_gpus)
                
                # Save GPU load heatmap and boxplot using analyzer functions
                gpu_heatmap_path = output_dir / f"gpu_loads_heatmap_{time.strftime('%Y%m%d_%H%M%S')}.png"
                gpu_boxplot_path = output_dir / f"gpu_loads_boxplot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                
                visualizer.plot_gpu_loads_analysis(
                    gpu_loads,
                    gpu_heatmap_path,
                    gpu_boxplot_path,
                    title_prefix=""
                )

                logger.info(f"[EPLBManager] Saved GPU load visualizations to: {gpu_heatmap_path}, {gpu_boxplot_path}")


        except Exception as e:
            logger.error(f"[EPLBManager] Error during visualization: {e}")
            import traceback
            logger.error(f"[EPLBManager] Traceback: {traceback.format_exc()}")
        
        logger.info("[EPLBManager] Visualization completed")


def _chunk_list(items: List, chunk_size):
    for start_index in range(0, len(items), chunk_size):
        yield items[start_index : start_index + chunk_size]
