---
experiment_id: LEGACY-d4f246190ad6
recorded_at: 2026-10-27
status: incomplete
data_mode: native512
classification_encoder: spp
mil_pooling: abmil
fold: 0
seed: 1
stain_augmentation: RandStainNA-LAB
snapshot_complete_epochs: 23
snapshot_partial_epoch: 24
snapshot_partial_epoch_wsis: 24
---

# LEGACY-d4f246190ad6

## 快照与来源

- 状态：**incomplete（训练中回传快照）**。计划300 Epoch，history和轮末日志仅包含23个完整Epoch；第24轮记录到24/176张训练WSI，iteration=4072、normal_135/N=19，没有第24轮评估或正常完成标记。不能将本快照记为300轮完成，也不能仅由日志末尾判断服务器是否仍在运行。
- 本次原始材料只有用户提供的config(2).yaml、history(3).csv和对应log。三份文件逐字节保存，未补造checkpoint、逐WSI预测、run_info或data_provenance。
- 实验ID使用仓库既有的LEGACY路径Hash规则生成，**是本地归档ID，不是服务器原始experiment_id**。服务器训练Git Commit、准确启动时间、GPU型号及软件版本未随本次材料提供，保持未记录。日志文件名的时间不作为训练启动时间证明。
- 本地目录：`D:\metasr-e2e\E2E_MetaSR\runs\native512_d1_randstainna_lab_sr0_m4_300_f0_s1_20261027_101751`。
- 模型：Native512 + RDN-D1/C8/G64 + 原Max SPP(1,2,3,6)/3200维 + Gated ABMIL + Linear；SR=0、microbatch4、BF16、lr=1e-4、weight_decay=1e-4、seed1、300轮/patience301、无几何增强、freeze_meanshift=false。
- RandStainNA：LAB/normal、probability=0.5、std_hyper=-0.3；日志确认enabled=true及独立NumPy RandomState(seed)。统计文件SHA256为`c6c222547aa329d9a32f13310f8f452f603b4b861fc43bd4ddd1677d312a81db`，与本地已审查的fold0统计文件逐字节一致。

## 最佳AUC轮与最后完整轮

summary追加最高Val AUC对应的**第18轮整行**，不拼接其他Epoch的最好指标。

| 指标 | 最佳AUC轮：18 | 最后完整轮：23 |
|---|---:|---:|
| train_loss_cls | 0.409626 | 0.301720 |
| train_loss_sr | 0.000000 | 0.000000 |
| train_loss_total | 0.409626 | 0.301720 |
| train_eval_loss | 0.442239 | 0.227312 |
| train_auc | 0.908585 | 0.974960 |
| train_acc | 0.806818 | 0.903409 |
| train_bacc | 0.822205 | 0.903683 |
| val_loss | 0.924891 | 1.248826 |
| val_auc | 0.543190 | 0.523582 |
| val_acc | 0.534091 | 0.545455 |
| val_bacc | 0.546105 | 0.522523 |

23轮Val AUC范围0.469528–0.543190，12/23轮低于0.5；最后5轮平均Val AUC=0.508850、BACC=0.499205、Val CE=1.135174。第1→23轮Train AUC从0.594595升至0.974960，Train Eval CE从0.676358降至0.227312，而Val CE从0.681641升至1.248826。增强没有阻止训练集继续拟合，验证排序仍接近随机。

在线train_loss_cls来自随机染色增强图像；Train AUC、Train Eval CE和Val指标来自无随机增强的轮末评估。二者不是相同输入、相同时间点的Loss，不直接用它们的差值判断异常。

## 分类偏置与指标推导

本次没有best_val_predictions.csv，无法独立重算概率AUC、查看各WSI概率或检验分数校准。下表仅根据Fold0验证集51 Normal/37 Tumor及history中的ACC、BACC，穷举整数混淆矩阵后得到唯一解，**属于指标推导，不是逐WSI预测文件复算**。

| Epoch | TN | FP | FN | TP | 预测Normal/Tumor | Tumor Recall | Normal Specificity |
|---:|---:|---:|---:|---:|---|---:|---:|
| 18 | 24 | 27 | 14 | 23 | 38 / 50 | 62.16% | 47.06% |
| 23 | 34 | 17 | 23 | 14 | 57 / 31 | 37.84% | 66.67% |

相对原Baseline最佳轮全部预测Normal，本组第18轮检出更多Tumor，但51张Normal中误报27张，BACC仅0.546105；不能只用召回提高认定整体有效。第23轮召回又降至37.84%，分类偏置并不稳定。

## 单因素对照：共同前23轮

对照：`runs/native512_d1_micro4_noaug_300`。解析后的YAML仅新增stain_augmentation块；模型、Native512、D1、scale16、SR0、micro4、BF16、LR、WD、seed、几何增强关闭、训练轮数和patience均相同。当前日志前4072条(epoch, iteration, slide_id, N)与Baseline对应前缀完全相同，未发现WSI顺序被染色RNG改变。

当前没有服务器源码快照、checkpoint和data_provenance，不能进一步证明两次运行的软件环境、初始权重及全部服务器数据Hash完全一致。

| 指标：共同前23轮 | 无增强Baseline | RandStainNA-LAB |
|---|---:|---:|
| 最佳Val AUC | 0.556969 | 0.543190 |
| 23轮平均Val AUC | 0.509470 | 0.498445 |
| 23轮平均Val BACC | 0.467685 | 0.480024 |
| 23轮平均Val CE | 0.963177 | 0.876773 |
| 第23轮Train AUC | 0.998940 | 0.974960 |
| 第23轮Train Eval CE | 0.072940 | 0.227312 |
| 第23轮Val AUC | 0.487546 | 0.523582 |
| 第23轮Val BACC | 0.469793 | 0.522523 |
| 第23轮Val CE | 1.622780 | 1.248826 |

RandStainNA有7/23轮AUC高于同轮Baseline，16/23轮较低。第23轮AUC和BACC较Baseline高、Val CE较低，但最高AUC和23轮平均AUC均未提高。**当前证据支持“减慢拟合、缓解部分后期CE恶化”，不支持“已经解决验证集泛化”。** 只有fold0/seed1及未完成快照；相关Epoch不是独立重复，不进行显著性或最终优越性宣称。

## 日志、数值与数据清单检查

- history的23行与23条轮末日志所有字段一致；每轮在线Loss均值、显存峰值逐项复算一致。
- 4072次训练更新连续；每个完整Epoch恰有176个唯一WSI，全部ID和Region数量与本地Fold0元数据一致。Region数范围1–42，没有从microbatch大小推断或删除Region。
- 本地Fold0元数据：训练176 WSI（102 Normal/74 Tumor）、2784 Region；验证88 WSI（51/37）、1559 Region；train/val病例无交叉。上述是本地清单与日志验证，不是本机完整图片验证，也不代替服务器data_provenance。
- 所有逐WSI Loss有限，SR Loss恒0、total=cls；407条梯度诊断均有限，RDN/ABMIL/Classifier有非零梯度。3次ABMIL零梯度均为N=1；Max SPP无参数，因此region_encoder=0正常；SR关闭，因此P2W=0正常。
- 没有Traceback、CUDA OOM、Non-finite或KeyboardInterrupt。autocast FutureWarning属于弃用提示，不作为训练失败证据。
- 回传日志峰值allocated=14.928082 GiB、reserved=16.218750 GiB；不是本机GPU重测，不能外推其他WSI或配置的显存需求。
- 依据已审查的本地训练入口：仅训练LR执行随机染色，train_eval/val_eval使用原RGB；全WSI Region embedding拼齐后一次ABMIL/CE/backward/Adam。SR=0训练循环不读HR、不调用SR Decoder；启动阶段独立CPU合成Decoder自检仍执行并已由日志确认通过。

## 当前结论与下一步

1. 当前没有稳定的Val AUC改善，不能因单轮峰值、较低Val CE或较高Tumor Recall就认定染色增强有效；也不能据此证明染色差异完全无关。
2. 信息价值最高的补充材料是第18轮best_val_predictions及同一保存时刻的checkpoint，用于成对比较误报、漏报和概率排序。本次未提供，未伪造这些结果。
3. 优先检查误判WSI的病灶覆盖、Region内容和颜色分布；后续对照继续固定模型、LR/WD、microbatch和SR，不同时更换池化、开启SR或添加模块。是否继续训练由用户决定，本次归档没有启动、恢复或干扰训练。

## 复现与五文件发布范围

本地RandStainNA实现、配置模板和统计文件目前尚未被Git跟踪；本次仅发布实验记录，不夹带源码或统计YAML。公开仓库中仅有本记录和运行配置，**不能据此声称下载main即可直接复现RandStainNA训练**。

本次发布五个文件：本Markdown、results/summary.csv追加行、本运行config.yaml、history.csv、training.log。未上传权重、预测CSV、JSON元数据或实现源码；旧实验原始文件不改写。

| 原始归档文件 | Bytes | SHA256 |
|---|---:|---|
| config.yaml | 719 | `a59b4df8cafadb96b394231354b0c3019901a941755f340a2e30b5cb04769bc4` |
| history.csv | 5173 | `5ba5d54b5fe8a6c04257f9595191248aa04f9f81c608ddabb668c3ab0126971a` |
| training.log | 934261 | `1df103ae547c582ee47060b767ef9811bc5e7998f4b98714203501046a6dc0ed` |

原始来源路径：

- `D:\Backup\Downloads\config(2).yaml`
- `D:\Backup\Downloads\history(3).csv`
- `D:\Backup\Downloads\native512_d1_randstainna_lab_sr0_m4_300_f0_s1_20261027_101751.log`

原始归档：[配置](../runs/native512_d1_randstainna_lab_sr0_m4_300_f0_s1_20261027_101751/config.yaml)、[逐轮指标](../runs/native512_d1_randstainna_lab_sr0_m4_300_f0_s1_20261027_101751/history.csv)、[日志](../runs/native512_d1_randstainna_lab_sr0_m4_300_f0_s1_20261027_101751/training.log)。Git可能规范化换行，以上SHA256针对本地逐字节归档原件。
