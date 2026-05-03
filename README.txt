论文提交代码包（按类别整理）

目录说明：
- 01_train: 训练入口脚本与yaml配置
- 02_infer_top3_main: 论文主推理代码（Task1使用Top3版本）
- 02_infer_optional_top1_runtime: 当前运行链路相关（保持top1，不改总控）
- 03_eval: 三个任务评估代码
- 04_data_construction_optional: 数据构造代码（如论文包含数据构造贡献建议提交）

说明：
- 已将 infer_task1_withtop3.py 作为主提交推理脚本。
- 总控 run_qwen_infer_all_gpu3.sh 保持原样（top1链路），放在 optional 目录。
