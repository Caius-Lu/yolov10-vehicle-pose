# 05 CarFusion 14-Point Index

The official converter exposes numeric indices 0–13 but does not provide sufficient semantic left/right names in its README. The original visualization code has been audited for connectivity, but the final semantic names and `flip_idx` require human confirmation against at least 100 authorized images.

The real COCO metadata confirms 14 numeric names and supplies the skeleton edges. `scripts/visualize_carfusion.py` now creates a deterministic 100-image review set covering views, small vehicles, occlusion, multi-vehicle scenes, and random samples; visible and occluded points use different colors and every point is numbered.

The official converter's `Visualize.py` provides part-level semantics:

- 0–3: wheels;
- 4–5: front lights;
- 6–7: rear lights;
- 8–9: side mirrors;
- 10–13: top/roof corners.

It does not name each side consistently enough to enable augmentation without visual confirmation. The likely pair structure is visible in its connectivity, but no `flip_idx` is activated from inference alone.

Until that review is complete:

- indices remain `0..13`;
- `flip_idx: []`;
- horizontal flip is forced off with `fliplr=0.0` in all training scripts.
