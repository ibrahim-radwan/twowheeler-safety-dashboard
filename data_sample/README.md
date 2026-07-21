# data_sample

Default dataset folder for the dashboard (`KITTI_BASE_PATH` → `./data_sample`).

## Layout

```text
data_sample/
  image_2/00000.png …
  calib/00000.txt …
  label_2/00000.txt …
  radar/00000.bin …
```

IDs must match across folders.

## Git

Binary files here are ignored by `.gitignore`. Copy your annotated sequence into this folder before running the app, or set `KITTI_BASE_PATH` to another root.
