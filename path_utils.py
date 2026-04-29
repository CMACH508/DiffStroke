import os


def default_path(preferred, legacy=None):
    """Return the preferred path, falling back to a legacy path if it exists."""
    if legacy and (not os.path.exists(preferred)) and os.path.exists(legacy):
        return legacy
    return preferred


def join_path(*parts):
    return os.path.normpath(os.path.join(*parts))


DEFAULT_DATA_ROOT = default_path("data", "Dataset")
DEFAULT_CHECKPOINT_ROOT = default_path("checkpoints", "models")
DEFAULT_OUTPUT_ROOT = default_path("outputs", "experiments")


def apply_checkpoint_config(config, checkpoint_root):
    clip_path = default_path(
        join_path(checkpoint_root, "clip-vit-large-patch14"),
        "./models/clip-vit-large-patch14",
    )
    cond_params = config.model.params.cond_stage_config.params
    if "version" in cond_params:
        cond_params.version = clip_path
    return config


def checkpoint_path(checkpoint_root, parts, legacy):
    return default_path(join_path(checkpoint_root, *parts), legacy)


def resolve_common_paths(
    opt,
    dataset_parts=None,
    legacy_dataset=None,
    fusion_parts=None,
    legacy_fusion=None,
    init_fusion_parts=None,
    legacy_init_fusion=None,
    need_landmark=False,
):
    if getattr(opt, "data_path", None) is None and dataset_parts:
        opt.data_path = default_path(join_path(DEFAULT_DATA_ROOT, *dataset_parts), legacy_dataset)
    if getattr(opt, "ckpt", None) is None:
        opt.ckpt = checkpoint_path(
            opt.checkpoint_root,
            ["stable-diffusion-v1-5", "v1-5-pruned-emaonly.ckpt"],
            "models/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt",
        )
    if getattr(opt, "pidinet_ckpt", None) is None:
        opt.pidinet_ckpt = checkpoint_path(
            opt.checkpoint_root,
            ["pidinet", "table5_pidinet.pth"],
            "models/table5_pidinet.pth",
        )
    if getattr(opt, "adapter_ckpt", None) is None:
        opt.adapter_ckpt = checkpoint_path(
            opt.checkpoint_root,
            ["t2i-adapter", "t2iadapter_sketch_sd15v2.pth"],
            "models/t2iadapter_sketch_sd15v2.pth",
        )
    if hasattr(opt, "fusion_ckpt") and opt.fusion_ckpt is None and fusion_parts:
        opt.fusion_ckpt = checkpoint_path(opt.checkpoint_root, fusion_parts, legacy_fusion)
    if hasattr(opt, "init_fusion_ckpt") and opt.init_fusion_ckpt is None and init_fusion_parts:
        opt.init_fusion_ckpt = checkpoint_path(opt.checkpoint_root, init_fusion_parts, legacy_init_fusion)
    if need_landmark and getattr(opt, "landmark_model", None) is None:
        opt.landmark_model = checkpoint_path(
            opt.checkpoint_root,
            ["dlib", "shape_predictor_68_face_landmarks.dat"],
            "./models/shape_predictor_68_face_landmarks.dat",
        )
    return opt
