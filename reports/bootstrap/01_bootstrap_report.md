# 01 Bootstrap Report

Status: **PASS_WITH_DOCUMENTED_ENVIRONMENT_DEVIATION**

## Repository

- URL: `https://github.com/THU-MIG/yolov10.git`
- Upstream commit: `453c6e38a51e9d1d5a2aa5fb7f1014a711913397`
- Working branch: `feat/vehicle-pose`
- Local path: `/Users/caius/Documents/yolopose/yolov10-vehicle-pose`
- Remote path: `/home/caius/projects/yolopose/yolov10-vehicle-pose`

## Remote environment

- OS: Ubuntu 22.04 under WSL2, Linux 5.15.146.1
- Python: 3.10.12
- PyTorch: 2.6.0+cu124
- torchvision: 0.21.0+cu124
- CUDA runtime reported by PyTorch: 12.4
- cuDNN: 9.1.0
- GPU: NVIDIA GeForce RTX 3090 Ti, 24,564 MiB
- NVIDIA driver: 560.94
- OpenCV: 4.11.0
- NumPy: 1.26.4

The official `requirements.txt` pins PyTorch 2.0.1/torchvision 0.15.2. Reinstalling began at approximately 22 KB/s and would replace an already working CUDA 12.4 stack. The environment was therefore rebuilt as project `.venv` with `--system-site-packages`, then the repository was installed editable using `--no-build-isolation --no-deps`. All imported versions are recorded above. This is an explicit reproducibility deviation, not an unreported dependency change.

## Official pretrained weight

- URL: `https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt`
- File: `weights/yolov10n.pt`
- Size: 11,448,431 bytes
- SHA256: `61b91ffc99b284792dca49bf40216945833cc2a515e1a742954e6e9327cfc19e`

## Verification

- Official checkpoint load on PyTorch 2.6: PASS after a tested compatibility fix for the changed `torch.load(weights_only)` default.
- Upstream YOLOv10n Detect 640×640 zero-image inference: PASS; all eight returned tensors finite.
- Vehicle Pose pretrained transfer: 493/679 state-dict items transferred; new pose towers retained initialized values.
- Vehicle Pose CUDA forward, `one2one_only=True`: PASS; output `(1, 47, 8400)`, all finite.
- Vehicle Pose parameter count: 4,005,986.

## Commands actually executed

```bash
git clone https://github.com/THU-MIG/yolov10.git yolov10-vehicle-pose
git switch -c feat/vehicle-pose
python3 -m venv --clear --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m pip install pytest pytest-cov
curl -L --fail --retry 3 --retry-delay 2 -o weights/yolov10n.pt <official-release-url>
shasum -a 256 weights/yolov10n.pt
.venv/bin/python -m pytest tests/test_vehicle_pose.py -q
```

## Failures and resolution

1. Full pinned dependency installation was stopped because of a measured 22 KB/s source and unnecessary CUDA stack downgrade; see deviation above.
2. PyTorch 2.6 rejected the official module checkpoint under its new `weights_only=True` default. A regression-tested compatibility loader now explicitly uses `weights_only=False` for this existing project checkpoint format, with a fallback for older PyTorch.

