# 07 Profiler Analysis

Status: **SCRIPT READY, FORMAL PROFILE NOT RUN**

Code inspection and the timing smoke identify the unused one-to-many eval towers as an execution bottleneck. `profile_torch.py` is ready to emit CPU/CUDA top-30 operator tables and TensorBoard traces, but the formal profile is deferred until a trained checkpoint and fixed real evaluation list exist.

