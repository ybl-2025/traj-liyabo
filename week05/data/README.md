# 数据来源与独立复现

本目录附带完整课程SI源数据：`NGSIM_I80_20050413_1600-1615_SI.csv.gz`，约26.24 MB。它是原始CSV的gzip无损压缩版，脚本直接读取，无需解压或重新下载。原始CSV为99,340,985字节，共1,226,650条记录、2,052个车辆编号。

- 来源：[U.S. DOT NGSIM车辆轨迹](https://data.transportation.gov/d/8ect-6jqj)。
- 地点：I-80, Emeryville, California。
- 时段：2005-04-13当地时间16:00:00≤t<16:15:00，PDT（UTC−7）。
- 课程数据下载日期：2026-10-08；官方API按分钟提取、合并并换算为SI，未平滑或插补。
- 单位：US survey foot = 1200/3937 m，SI数值保留6位小数，时间为相对秒数，0.1 s采样。
- 解压内容SHA-256：`df86a98e490daba1bf6a2ff3a1ca2e346be1cf2bdaf66a037910b74f40695255`。

`package_manifest.json`保存压缩文件和解压内容的大小及SHA-256；`source_and_checks.json`保留课程源数据的官方API查询、转换和原始校验记录。官方数据入口标注许可：[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。数据署名为U.S. DOT / Federal Highway Administration；课程版本进行了字段选择和单位转换，后续筛选及小幅去噪已在项目README说明。

所需字段：`vehicle_id, frame_id, time_s, lane_id, local_y_m, local_x_m, vehicle_length_m, vehicle_class, speed_m_s`。官方英尺数据不能直接冒充SI输入。

在 `week05/` 下运行：

```bash
python run_experiments.py
```

脚本从本目录源文件重建候选池，按`selection_config.json`固定规则选出21段训练和13段测试，每段30秒。源数据只读；原始与处理后的34段信号同时保存于`results/following_observations.csv`，全部模型预测、图表和审计记录也包含在仓库中。筛选规则不使用模型误差或排名，训练/测试前后车编号无交叉。

压缩源数据明确提交Git；未压缩的大CSV、ZIP、缓存和旧归档仍由`.gitignore`排除。整个week05可独立运行，不依赖本机week4目录。
