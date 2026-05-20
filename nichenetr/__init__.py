"""
nichenetr — Python port of the R nichenetr package.

NicheNet: modeling intercellular communication by linking ligands to
target genes.
"""

__version__ = "2.2.1.1"
__r_commit__ = "2d5c1ab"

from .utils import (
    scaling_zscore,
    scaling_modified_zscore,
    scale_quantile,
    scale_quantile_adapted,
)
from .prediction import (
    predict_ligand_activities,
    predict_single_cell_ligand_activities,
    normalize_single_cell_ligand_activities,
    single_ligand_activity_score_regression,
)
from .symbols import (
    get_expressed_genes,
    alias_to_symbol_anndata,
    convert_alias_to_symbols,
    assign_ligands_to_celltype,
    get_lfc_celltype,
)
from .targets import (
    get_weighted_ligand_target_links,
    prepare_ligand_target_visualization,
    get_weighted_ligand_receptor_links,
    prepare_ligand_receptor_visualization,
    get_ligand_target_links_oi,
)
from .networks import (
    get_ligand_signaling_path,
    format_signaling_graph,
    infer_supporting_datasources,
)
from .evaluation import (
    assess_rf_class_probabilities,
    classification_evaluation_continuous_pred_wrapper,
    calculate_fraction_top_predicted,
    calculate_fraction_top_predicted_fisher,
    get_top_predicted_genes,
    convert_settings_ligand_prediction,
)
from .prioritization import (
    calculate_de,
    get_exprs_avg,
    process_table_to_ic,
    generate_prioritization_tables,
    generate_info_tables,
)
from .wrappers import (
    nichenet_seuratobj_aggregate,
    nichenet_seuratobj_cluster_de,
)
from .visualization import (
    make_heatmap_ggplot,
    make_threecolor_heatmap_ggplot,
    make_line_plot,
    make_mushroom_plot,
    make_circos_plot,
    make_circos_lr,
    prepare_circos_visualization,
)
from .datasets import (
    NamedMatrix,
    load_lr_network,
    load_ligand_target_matrix,
    load_weighted_networks,
    load_sig_network,
    load_gr_network,
    load_ligand_tf_matrix,
    load_seurat_obj,
    load_hnscc_expression,
    load_pemt_signature,
    load_source_weights_df,
    load_hyperparameter_list,
    load_geneinfo,
    load_geneinfo_alias,
    load_optimized_source_weights_df,
)

__all__ = [
    # utils
    "scaling_zscore",
    "scaling_modified_zscore",
    "scale_quantile",
    "scale_quantile_adapted",
    # prediction
    "predict_ligand_activities",
    "predict_single_cell_ligand_activities",
    "normalize_single_cell_ligand_activities",
    "single_ligand_activity_score_regression",
    # symbols
    "get_expressed_genes",
    "alias_to_symbol_anndata",
    "convert_alias_to_symbols",
    "assign_ligands_to_celltype",
    "get_lfc_celltype",
    # targets
    "get_weighted_ligand_target_links",
    "prepare_ligand_target_visualization",
    "get_weighted_ligand_receptor_links",
    "prepare_ligand_receptor_visualization",
    "get_ligand_target_links_oi",
    # networks
    "get_ligand_signaling_path",
    "format_signaling_graph",
    "infer_supporting_datasources",
    # evaluation
    "assess_rf_class_probabilities",
    "classification_evaluation_continuous_pred_wrapper",
    "calculate_fraction_top_predicted",
    "calculate_fraction_top_predicted_fisher",
    "get_top_predicted_genes",
    "convert_settings_ligand_prediction",
    # prioritization
    "calculate_de",
    "get_exprs_avg",
    "process_table_to_ic",
    "generate_prioritization_tables",
    "generate_info_tables",
    # wrappers
    "nichenet_seuratobj_aggregate",
    "nichenet_seuratobj_cluster_de",
    # visualization
    "make_heatmap_ggplot",
    "make_threecolor_heatmap_ggplot",
    "make_line_plot",
    "make_mushroom_plot",
    "make_circos_plot",
    "make_circos_lr",
    "prepare_circos_visualization",
    # datasets
    "NamedMatrix",
    "load_lr_network",
    "load_ligand_target_matrix",
    "load_weighted_networks",
    "load_sig_network",
    "load_gr_network",
    "load_ligand_tf_matrix",
    "load_seurat_obj",
    "load_hnscc_expression",
    "load_pemt_signature",
    "load_source_weights_df",
    "load_hyperparameter_list",
    "load_geneinfo",
    "load_geneinfo_alias",
    "load_optimized_source_weights_df",
]
