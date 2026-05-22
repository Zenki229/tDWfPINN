from omegaconf import OmegaConf


def float_token(value):
    return f"{float(value):.2f}".replace(".", "p")


def configured(value):
    if value is None:
        return False
    if str(value).strip().lower() in {"", "none", "null", "???"}:
        return False
    return True


def method_parts(method):
    pieces = str(method).split("-", maxsplit=1)
    quad = pieces[0]
    type_token = pieces[1] if len(pieces) > 1 else "NA"
    return quad, type_token


def _get(node, key, default=None):
    if node is None:
        return default
    return getattr(node, key, default)


def quadrature_points(cfg):
    quad, _ = method_parts(cfg.pde.method)
    quad_cfg = _get(cfg.pde, quad)
    return _get(quad_cfg, "nums")


def build_run_metadata(cfg):
    case = str(cfg.pde.name)
    alpha = float(cfg.pde.al)
    alpha_tag = f"alpha{float_token(alpha)}"
    method = str(cfg.pde.method)
    quad, type_token = method_parts(method)
    quad_points = quadrature_points(cfg)

    name_tokens = [case, alpha_tag, method]
    if quad_points is not None:
        name_tokens.append(f"{quad}{int(quad_points)}")

    if case == "forward":
        name_tokens.extend([
            f"k{int(cfg.pde.k)}",
            f"lam{float_token(cfg.pde.lam)}",
        ])

    name_tokens.extend([
        f"steps{int(cfg.training.max_steps)}",
        f"seed{int(cfg.seed)}",
    ])

    tags = [
        "jax",
        case,
        alpha_tag,
        method,
        quad,
        f"type{type_token}",
    ]

    return {
        "name": "_".join(name_tokens),
        "group": f"{case}_{alpha_tag}",
        "tags": tags,
        "job_type": "train",
    }


def prepare_wandb_run(cfg):
    metadata = build_run_metadata(cfg)
    if not configured(_get(cfg.wandb, "name")):
        cfg.wandb.name = metadata["name"]
    if not configured(_get(cfg.wandb, "group")):
        cfg.wandb.group = metadata["group"]
    if not configured(_get(cfg.wandb, "job_type")):
        cfg.wandb.job_type = metadata["job_type"]

    existing_tags = list(_get(cfg.wandb, "tags", []) or [])
    for tag in metadata["tags"]:
        if tag not in existing_tags:
            existing_tags.append(tag)
    cfg.wandb.tags = existing_tags
    return metadata


def wandb_init_kwargs(cfg):
    return {
        "project": cfg.wandb.project,
        "entity": cfg.wandb.entity,
        "mode": cfg.wandb.mode,
        "name": cfg.wandb.name,
        "group": cfg.wandb.group,
        "tags": list(cfg.wandb.tags or []),
        "job_type": cfg.wandb.job_type,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
