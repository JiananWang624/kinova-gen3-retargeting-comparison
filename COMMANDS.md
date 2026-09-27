# TIAGo Pro 常用指令

以下命令均在项目根目录 `F:\code\sew_mimic_repro` 的 PowerShell 中执行，使用已安装依赖的 `.venv`。当前配置以 [config.yaml](config.yaml) 为准；这些命令不会修改 placement 或人体坐标约定。

## 只看装配，不运行 IK

```powershell
.venv\Scripts\python.exe scripts\show_tiago_pro_mounting.py --frame 0
```

`--frame` 选择要显示的人体目标帧；机械臂保持 `q=0`。只打印 J1、arm_base、tool、肩点和目标点的世界坐标，不打开 MuJoCo 窗口：

```powershell
.venv\Scripts\python.exe scripts\show_tiago_pro_mounting.py --frame 0 --no-viewer
```

## 计算指定数量的帧

`--start-frame` 是从 0 开始的 CSV 帧索引，`--count` 是连续计算的帧数。以下均计算**前 N 帧**。输出目录建议按帧数区分；使用已有目录会覆盖其中同名结果文件。

```powershell
# 前 100 帧
.venv\Scripts\python.exe scripts\benchmark_tiago_pro.py --start-frame 0 --count 100 --output output\tiago_pro_first100

# 前 500 帧
.venv\Scripts\python.exe scripts\benchmark_tiago_pro.py --start-frame 0 --count 500 --manual-long-run --output output\tiago_pro_first500

# 前 1000 帧
.venv\Scripts\python.exe scripts\benchmark_tiago_pro.py --start-frame 0 --count 1000 --manual-long-run --output output\tiago_pro_first1000

# 前 2000 帧
.venv\Scripts\python.exe scripts\benchmark_tiago_pro.py --start-frame 0 --count 2000 --manual-long-run --output output\tiago_pro_first2000
```

计算指定区间，例如从帧 246 起连续计算 100 帧：

```powershell
.venv\Scripts\python.exe scripts\benchmark_tiago_pro.py --start-frame 246 --count 100 --output output\tiago_pro_246_345
```

每次输出包含 `window_frames.csv`（逐帧关节值、状态及误差）和 `summary.json`（汇总及当次 placement 快照）。超过 100 帧必须显式加 `--manual-long-run`；这只是人工运行开关，不会改变 solver，也不会自动跑 oracle。

## 在 MuJoCo 中播放结果

计算完成后，`--results` 指向对应输出目录的 `window_frames.csv`；`--max-frames` 指定本次最多播放的行数。

```powershell
# 播放前 100 帧结果
.venv\Scripts\python.exe scripts\replay_tiago_pro.py --results output\tiago_pro_first100\window_frames.csv --max-frames 100

# 播放当前保留的前 500 帧结果
.venv\Scripts\python.exe scripts\replay_tiago_pro.py --results output\tiago_pro_first500\window_frames.csv --max-frames 500 --manual-long-run

# 播放前 1000 帧结果
.venv\Scripts\python.exe scripts\replay_tiago_pro.py --results output\tiago_pro_first1000\window_frames.csv --max-frames 1000 --manual-long-run

# 播放前 2000 帧结果
.venv\Scripts\python.exe scripts\replay_tiago_pro.py --results output\tiago_pro_first2000\window_frames.csv --max-frames 2000 --manual-long-run
```

播放超过 100 帧同样需要 `--manual-long-run`。可加 `--fps 30` 设置播放速率。若只想校验结果而不打开 Viewer，在 replay 命令末尾加 `--no-viewer`；它会用 MuJoCo FK 复核已保存的成功帧。失败帧在 Viewer 中保持上一个有效机械臂姿态，并显示失败目标，不能把静止姿态当作该帧的解。

查看当前任务定义、坐标链和注意事项：[README.md](README.md)；迁移背景：[HANDOFF.md](HANDOFF.md)。
