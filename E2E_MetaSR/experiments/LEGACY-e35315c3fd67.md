# LEGACY-e35315c3fd67

- 状态：**incomplete（本次回传快照）**。配置计划300轮，但仅收到21个完整epoch；日志继续到第22轮第46/176张训练WSI，iteration=3742后结束。没有正常完成标记，不能记为300轮完成，也不能仅由本地快照判断服务器是否仍在运行或为何停止。
- 模型：Native512 + RDN-D1（C8、growth64、输出64通道）+ Region-aware Adaptive SPP V1（3200维）+ Gated ABMIL + Linear。
- 配置：scale16、lambda_sr=0、Microbatch8、BF16、lr=1e-4、weight_decay=1e-4、seed1、无增强；epochs=300、early_stopping_patience=301。
- Fold：启动命令指定0；21个完整epoch的训练WSI集合与本地真实Fold0逐项一致。验证成员列表、标签和四项来源Hash未随本次文件回传，不能声称全部重新核验。
- GPU型号、原始实验ID、Git Commit、开始/结束时间：本次文件未记录。日志含CUDA显存统计，但不能据此补填GPU型号。
- 本记录ID采用既有summarize工具的LEGACY回退规则，由本地run目录Hash生成，不冒充服务器原始实验ID。
- 本地目录：`D:\metasr-e2e\E2E_MetaSR\runs\native512_d1_region_gate_sr0_noaug_m8_300`；归档日期：2026-10-23。

## 实验目标与训练数据流

在保持Native512/RDN-D1/ABMIL/Linear与原优化目标的基础上，以Region级动态Max/Avg融合代替原Max SPP。本次关闭SR，单独观察分类表现；不增加额外Loss。

每次最多8个Region：LR `[m,3,512,512]` → 共享RDN `[m,64,512,512]`。分别计算levels=(1,2,3,6)的Max/Avg SPP，使用128维全局Max/Avg描述向量，经LayerNorm→Linear(128,32)→ReLU→Linear(32,1)→Sigmoid产生每个Region自己的alpha，融合为 `[m,3200]`。拼接全WSI `[N,3200]` 后一次ABMIL、一次WSI交叉熵、一次backward/Adam更新。

lambda_sr=0时训练不抽取SR Crop、不读取HR像素、不调用SR Decoder；RDN仍由分类梯度更新。配置保留scale16和sr_train_crop256，它们不表示这次执行了8K重建。启动时的小尺寸Decoder/checkpoint等价性自检仍会执行。

## 最佳AUC轮与最后完整轮

按照入口的严格Val AUC改善规则，当前history最佳轮为2；本次未提供best.pth/last.pth，没有检查真实checkpoint内容或Gate参数。summary只追加第2轮整行，不拼接其他轮的最好指标。

| 指标 | 最佳AUC轮：2 | 最后完整轮：21 |
|---|---:|---:|
| train_loss_cls | 0.696322 | 0.482007 |
| train_loss_sr | 0.000000 | 0.000000 |
| train_loss_total | 0.696322 | 0.482007 |
| train_eval_loss | 0.682395 | 0.398190 |
| train_auc | 0.681373 | 0.917263 |
| train_acc | 0.579545 | 0.823864 |
| train_bacc | 0.500000 | 0.820217 |
| val_loss | 0.686834 | 1.090308 |
| val_auc | 0.545045 | 0.348967 |
| val_acc | 0.579545 | 0.386364 |
| val_bacc | 0.500000 | 0.377848 |

本次没有best_val_predictions.csv。若验证集使用真实Fold0的51 Normal/37 Tumor，由同轮ACC与BACC可反推唯一的整数混淆矩阵：

| 轮次 | TN | FP | FN | TP | Tumor Recall |
|---|---:|---:|---:|---:|---:|
| 最佳AUC轮2 | 51 | 0 | 37 | 0 | 0/37 |
| 最后完整轮21 | 22 | 29 | 25 | 12 | 12/37 = 0.324324 |

以上是类别总数与汇总指标的数学推导，不是对逐WSI预测文件的独立复算。无法给出p_tumor范围、分布或重算AUC。

## 训练趋势

| Epoch | Train AUC | Train eval CE | Val AUC | Val BACC | Val CE |
|---|---:|---:|---:|---:|---:|
| 1 | 0.630498 | 0.707941 | 0.489931 | 0.500000 | 0.711115 |
| 2 | 0.681373 | 0.682395 | 0.545045 | 0.500000 | 0.686834 |
| 3 | 0.682962 | 0.670554 | 0.499735 | 0.500000 | 0.680575 |
| 4 | 0.659579 | 0.670399 | 0.484367 | 0.453895 | 0.693692 |
| 5 | 0.723105 | 0.661055 | 0.483837 | 0.500000 | 0.681286 |
| 6 | 0.677266 | 0.638250 | 0.488076 | 0.432962 | 0.706587 |
| 7 | 0.727676 | 0.635010 | 0.480392 | 0.471913 | 0.691695 |
| 8 | 0.729730 | 0.640570 | 0.476683 | 0.468203 | 0.689187 |
| 9 | 0.635864 | 0.829974 | 0.421569 | 0.419714 | 1.119234 |
| 10 | 0.743906 | 0.618896 | 0.441441 | 0.468468 | 0.717085 |
| 11 | 0.747615 | 0.614430 | 0.448596 | 0.505034 | 0.737527 |
| 12 | 0.716746 | 0.602975 | 0.423423 | 0.454955 | 0.805376 |
| 13 | 0.767488 | 0.584595 | 0.407260 | 0.397191 | 0.751265 |
| 14 | 0.796635 | 0.548934 | 0.404081 | 0.402226 | 0.802657 |
| 15 | 0.804650 | 0.551616 | 0.375199 | 0.408585 | 0.847579 |
| 16 | 0.808824 | 0.553689 | 0.377318 | 0.452305 | 0.789745 |
| 17 | 0.845456 | 0.501909 | 0.360095 | 0.369104 | 0.847723 |
| 18 | 0.854001 | 0.476875 | 0.385533 | 0.396131 | 0.949668 |
| 19 | 0.858506 | 0.462296 | 0.333863 | 0.398516 | 1.014543 |
| 20 | 0.897787 | 0.425793 | 0.354001 | 0.415739 | 1.006264 |
| 21 | 0.917263 | 0.398190 | 0.348967 | 0.377848 | 1.090308 |

21轮Val AUC平均0.429972，最后5轮平均0.356492，20/21轮低于0.5。第2→21轮Train AUC从0.681373升至0.917263，而Val AUC从0.545045降至0.348967，Val CE从0.686834升至1.090308。**训练拟合增强，但验证排序明显恶化；当前快照没有显示V1改善分类泛化。**

最高Val BACC是第11轮0.505034，同轮AUC0.448596；最小Val CE是第3轮0.680575，同轮AUC0.499735。它们不能与第2轮AUC合成一个最佳结果。

后期Val AUC低于0.5表示验证排序出现反向倾向，但不证明标签被写反，更不能通过翻转验证分数来宣称模型改善；需要逐WSI概率、标签及来源核查。

## 梯度、SR关闭与异常波动

3742条逐WSI记录中，SR Loss全部为0，total与分类Loss逐条完全一致；374条诊断的P2W梯度全部为0。Gate、RDN、Classifier的诊断梯度均有限且非零，说明Gate没有整体断开训练路径；这不能证明各层或每个Region都没有饱和，也不能证明alpha随Region有效变化。

| 模块 | 最小范数 | 最大范数 | 零范数次数 |
|---|---:|---:|---:|
| rdn | 0.660305 | 142.732 | 0 / 374 |
| region_encoder | 0.0027059 | 10.9489 | 0 / 374 |
| abmil | 0 | 36.1421 | 5 / 374 |
| classifier | 0.583715 | 541.946 | 0 / 374 |
| p2w | 0 | 0 | 374 / 374 |

5次ABMIL零梯度均对应N=1的WSI。单Region时注意力softmax恒为1，其参数不影响聚合结果，零梯度符合数学定义，不是断梯度证据。

第9轮iteration1580、tumor_104（N=28）出现CLS=20.25的单WSI尖峰；同次RDN/Gate/ABMIL/Classifier梯度范数约142.732/10.949/36.142/541.946。均有限，但明显高于一般记录，值得结合该步logits、embedding范数、attention与alpha追查。没有权重/中间量，不能将它直接定性为Gate饱和、BF16故障或已证实的梯度爆炸根因。

日志没有Traceback、CUDA OOM、Non-finite或KeyboardInterrupt；开头的torch.cpu.amp.autocast FutureWarning是弃用提示，不是本次训练失败的异常证据。

## 与原Max SPP、SR0、Microbatch4对照

两份原始YAML只在classification_encoder和region_microbatch_size上不同：spp→spp_region_gate、4→8，其余字段一致。因此不是严格单变量消融；BF16形状变化与Gate初始化后的随机数消耗也会影响轨迹。实际首轮176张WSI的排列已经不同。

| 指标（共同前21轮） | 原Max SPP / micro4 | V1 / micro8 |
|---|---:|---:|
| 最高Val AUC | 0.556969（epoch1） | 0.545045（epoch2） |
| 对应Val BACC | 0.500000 | 0.500000 |
| 第21轮Train AUC | 0.993111 | 0.917263 |
| 第21轮Train eval CE | 0.166640 | 0.398190 |
| 第21轮Val AUC | 0.479862 | 0.348967 |
| 第21轮Val BACC | 0.488341 | 0.377848 |
| 第21轮Val CE | 1.415788 | 1.090308 |
| 前21轮平均Val AUC | 0.510447 | 0.429972 |
| 快照峰值allocated GiB | 14.928083 | 23.428203 |
| 快照峰值reserved GiB | 16.218750 | 25.607422 |

当前V1拟合训练集较慢，但验证表现也更差，不能把训练AUC较低解释成抗过拟合成功。显存变化也不能全部归因于Gate：本次Microbatch翻倍；没有耗时或时间戳，不能计算每轮速度。

相关原实验：[Max SPP / SR0 / micro4](EXP-20261022-141941-1e1390df1608.md)。

## 数据覆盖与证据限制

逐条解析3742次WSI更新，iteration1–3742连续。前21轮各覆盖176张唯一训练WSI、2784个Region；成员集合与D:/c16_metadata的真实Fold0完全一致，所有逐WSI Region数均与详细manifest匹配；第22轮只有46张、655个Region。21条history与21条[epoch]日志逐字段完全一致，每轮在线CLS均值也与逐WSI日志重算完全一致。

本次只提供config、history和log，没有run_info.json、data_provenance.json、best/last checkpoint、逐WSI验证概率或alpha记录。没有补造这些元数据，GPU/训练Git Commit/起止时间留空。不能重算本组来源Hash或AUC、检查Gate权重与alpha、复现A100显存、判断病灶覆盖或确认真实图片的像素对应。所有结论限Fold0/seed1当前快照。

## 下一步建议

1. 当前趋势不支持仅为凑满300轮继续投入。优先补回best/last checkpoint和逐WSI验证概率，核查标签关联、反向排序集中在哪些WSI。
2. 从checkpoint离线提取alpha分布、Gate logits及attention；按Region数、类别和正确/错误WSI分组，区分常数门、饱和、视觉捷径与聚合异常；重点查看第9轮尖峰对应的tumor_104。
3. 结合真实肿瘤标注检查保留Region的病灶覆盖；若再做模型对照，固定micro8及随机序列，比较原Max、固定alpha=0.5与动态Gate，而不是直接再叠加模块。

本轮只归档、分析并上传实验记录，没有修改模型/训练逻辑，没有启动训练。

## 归档范围与原始文件

沿用五文件发布范围：本Markdown、results/summary.csv追加一行、本组config.yaml、history.csv与匹配log。源码、权重、预测和JSON不上传；此前六个正式新增源码/配置文件保持原状。

原始回传SHA256（本地复制保持原始字节；Git可规范化换行）：

- config.yaml：`113583b14eeb3113c56bd0e348db935c07a7a27f9fa722bbafa75da090a13b17`
- history.csv：`5f2c3a37876e2c8ca721770d371b83503f22635ea0ba00d70d78dc67f125dbd8`
- log：`49b280092bc59391e16f4cb75ccfb46360c6b7c2bc3b5dfa2fed011e722a57d2`

原始文件：[配置](../runs/native512_d1_region_gate_sr0_noaug_m8_300/config.yaml)、[逐轮指标](../runs/native512_d1_region_gate_sr0_noaug_m8_300/history.csv)、[终端日志](../logs/native512_d1_region_gate_sr0_noaug_m8_300.log)。
