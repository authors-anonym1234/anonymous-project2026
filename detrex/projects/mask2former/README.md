# Mask2Former (vendored) for detrex

This project vendors the upstream `Mask2Former/mask2former` package into `detrex/projects/mask2former/mask2former` so it can be used without relying on an external installation. The top-level `__init__.py` prepends this project to `sys.path`, so `import mask2former` resolves to the vendored copy.

Next steps to make it trainable via detrex LazyConfig:
- Add LazyConfig files under `configs/` (e.g., copy from `projects/maskdino/configs` and swap in Mask2Former components/params).
- Optionally add a thin `train_net.py` entry that delegates to `detrex/tools/train_net.py`.
- Ensure datasets are registered or guard duplicate registrations inside the vendored data loaders if needed.

During development, set `PYTHONPATH` to include `detrex/projects/mask2former` (or rely on detrex tools that add `projects/`).
