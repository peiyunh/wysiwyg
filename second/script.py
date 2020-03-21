import os
import fire
from second.pytorch.train import train, evaluate
from google.protobuf import text_format
from second.protos import pipeline_pb2
from pathlib import Path
from second.utils import config_tool
from second.data.all_dataset import get_dataset_class

def _div_up(a, b):
    return (a + b - 1) // b

def _get_cfg(cfg_path):
    cfg = pipeline_pb2.TrainEvalPipelineConfig()
    with open(cfg_path, 'r') as f:
        proto_str = f.read()
        text_format.Merge(proto_str, cfg)
    return cfg

def _fix_kitti_dir(cfg, kitti_dir):
    cfg.train_input_reader.dataset.kitti_info_path = f'{kitti_dir}/kitti_infos_train.pkl'
    cfg.train_input_reader.dataset.kitti_root_path = f'{kitti_dir}'
    cfg.train_input_reader.preprocess.database_sampler.database_info_path = f'{kitti_dir}/dbinfos_train.pkl'
    cfg.eval_input_reader.dataset.kitti_info_path = f'{kitti_dir}/kitti_infos_val.pkl'
    cfg.eval_input_reader.dataset.kitti_root_path = f'{kitti_dir}'

def _modify_kitti_dataset(cfg, new_kitti_dir, new_dataset_class, new_input_dim):
    # overwrite dataset directories
    cfg.train_input_reader.dataset.kitti_info_path = f'{new_kitti_dir}/kitti_infos_train.pkl'
    cfg.train_input_reader.dataset.kitti_root_path = f'{new_kitti_dir}'
    cfg.train_input_reader.preprocess.database_sampler.database_info_path = f'{new_kitti_dir}/dbinfos_train.pkl'
    cfg.eval_input_reader.dataset.kitti_info_path = f'{new_kitti_dir}/kitti_infos_val.pkl'
    cfg.eval_input_reader.dataset.kitti_root_path = f'{new_kitti_dir}'
    #
    cfg.train_input_reader.dataset.dataset_class_name = new_dataset_class
    cfg.eval_input_reader.dataset.dataset_class_name = new_dataset_class
    #
    cfg.model.second.num_point_features = new_input_dim
    cfg.model.second.voxel_feature_extractor.num_input_features = new_input_dim  # VFE
    cfg.model.second.middle_feature_extractor.num_input_features = new_input_dim-1  # MFE

def _fix_nusc(cfg, nusc_dir, sweep_db):
    train_info_name = 'infos_train_with_sweep_info' if sweep_db else 'infos_train'
    cfg.train_input_reader.dataset.kitti_info_path = f'{nusc_dir}/{train_info_name}.pkl'
    cfg.train_input_reader.dataset.kitti_root_path = f'{nusc_dir}'
    db_name = 'dbinfos_train_with_sweep_info' if sweep_db else 'dbinfos_train'
    cfg.train_input_reader.preprocess.database_sampler.database_info_path = f'{nusc_dir}/{db_name}.pkl'
    val_info_name = 'infos_val_with_sweep_info' if sweep_db else 'infos_val'
    cfg.eval_input_reader.dataset.kitti_info_path = f'{nusc_dir}/{val_info_name}.pkl'
    cfg.eval_input_reader.dataset.kitti_root_path = f'{nusc_dir}'

def _fix_nusc_step(cfg,
                   epochs,
                   eval_epoch,
                   data_sample_factor,
                   gpus,
                   num_examples=28130):
    input_cfg = cfg.train_input_reader
    train_cfg = cfg.train_config
    batch_size = input_cfg.batch_size
    data_sample_factor_to_name = {
        1: "NuScenesDataset",
        2: "NuScenesDatasetD2",
        3: "NuScenesDatasetD3",
        4: "NuScenesDatasetD4",
        5: "NuScenesDatasetD5",
        6: "NuScenesDatasetD6",
        7: "NuScenesDatasetD7",
        8: "NuScenesDatasetD8",
    }
    dataset_name = data_sample_factor_to_name[data_sample_factor]
    input_cfg.dataset.dataset_class_name = dataset_name
    ds = get_dataset_class(dataset_name)(
        root_path=input_cfg.dataset.kitti_root_path,
        info_path=input_cfg.dataset.kitti_info_path,
    )
    num_examples_after_sample = len(ds)
    step_per_epoch = _div_up(num_examples_after_sample, batch_size)
    step_per_eval = _div_up(step_per_epoch * eval_epoch, len(gpus))
    total_step = _div_up(step_per_epoch * epochs, len(gpus))
    train_cfg.steps = total_step
    train_cfg.steps_per_eval = step_per_eval


def train_kitti(
    kitti_dir = '/data/kitti/object',
    base_cfg  = 'all.fhd.config',
    ext_input = '',
    resume    = False,
):
    cfg_path = Path(f'configs/{base_cfg}')
    cfg_name = base_cfg.replace('.config', '').replace('.', '_')
    cfg = _get_cfg(cfg_path)
    if ext_input == '':
        _fix_kitti_dir(cfg, kitti_dir)
    else:
        if ext_input == 'gt_fgm':
            _modify_kitti_dataset(cfg, '/data/kitti/object_gt_fgm', 'KittiFGMDataset', 5)
        cfg_name = f'{cfg_name}_{ext_input}'
    model_dir = Path('models/kitti') / cfg_name
    train(cfg, model_dir, resume=resume)

def train_nuscenes(
    nusc_dir      = '/data/nuscenes',
    base_cfg      = 'all.fhd.config',
    sample_factor = 8,
    epochs        = 50,
    eval_epoch    = 5,
    sweep_db      = False,
    label         = "", 
    resume        = False,
):
    cfg_path = Path(f'configs/nuscenes/{base_cfg}')
    cfg_name = base_cfg.replace('.config', '').replace('.', '_')
    cfg_name += "_swpdb" if sweep_db else ""
    cfg_name += f"_d{sample_factor}_ep{epochs}_ev{eval_epoch}"
    cfg_name += "" if label == "" else f"_{label}"
    cfg = _get_cfg(cfg_path)
    gpus = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    _fix_nusc(cfg, nusc_dir, sweep_db)
    _fix_nusc_step(cfg, epochs, eval_epoch, sample_factor, gpus)
    model_dir = Path('models/nuscenes') / cfg_name
    train(cfg, model_dir, multi_gpu=len(gpus)>1, resume=resume)

def test_nuscenes(
    nusc_dir      = '/data/nuscenes',
    model_name    = 'all_pp_mhead_vpn_swpdb_d1_ep20_ev2_vp_pp_oa_ta_ots_early_fusion',
    eval_step     = 187540, 
):
    model_dir = f'models/nuscenes/{model_name}'
    ckpt_path = f'{model_dir}/voxelnet-{eval_step}.tckpt'

    cfg_path = f'{model_dir}/pipeline.config'
    cfg = _get_cfg(cfg_path)
    cfg.eval_input_reader.dataset.kitti_info_path = f'{nusc_dir}/infos_test.pkl'
    cfg.eval_input_reader.dataset.kitti_root_path = f'{nusc_dir}'

    evaluate(cfg, model_dir, ckpt_path=ckpt_path, clean_after=False)

if __name__ == "__main__":
    fire.Fire()
