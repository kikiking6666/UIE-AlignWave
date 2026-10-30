# UIE-AlignWave 上传说明

这个文件夹只保留训练和测试需要的源码、配置和说明文件。

## 已保留

- `train.py`：训练入口
- `test.py`：测试入口
- `config.yml`：训练/测试配置
- `DWT.py`：训练脚本直接依赖，用于子带监督
- `models/`：模型定义与实现（具体类名见源代码）
- `data/`：数据读取逻辑
- `metrics/`：UCIQE 指标
- `utils/`：随机种子、权重保存/加载等工具
- `config/`：YAML 配置加载
- `README.md`、`LICENSE`
- `requirements.txt`、`.gitignore`

## 已排除

- `Dataset/`
- `result/`
- `log/`
- `best weight/`
- `my_result.zip`
- `archive.zip`
- `__pycache__/`
- `.ipynb_checkpoints/`
- 当前训练/测试链路没有 import 的 `CFC.py`

## 使用前需要改

`config.yml` 里仍然保留服务器上的路径。换机器或换目录后，请按实际位置修改：

- `TRAINING.TRAIN_DIR`
- `TRAINING.VAL_DIR`
- `TRAINING.SAVE_DIR`
- `TESTING.VAL_DIR`
- `TESTING.WEIGHT`
- `TESTING.RESULT_DIR`

训练：

```bash
python train.py
```

测试：

```bash
python test.py
```
