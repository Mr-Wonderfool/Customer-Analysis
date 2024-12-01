### 数据集选择
- [Kaggle顾客数据集](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data)

### 代码运行
- 用脚本进行环境配置
```bash
# 如果使用WSL或者Linux子系统，授予权限
chmod +x config.sh
# windows程序直接运行脚本，请确保安装了git bash或其他bash工具
./config.sh
```

### 文件说明
在`models`文件夹下：
- `Density`中储存了基于密度聚类的文件，`density_clustering.ipynb`中调库实现，并储存了运行结果；`density_model.py`中采用手写实现DBSCAN。
- `Hierarchical`中储存了层次聚类相关文件，运行`clustering.py`将聚类结果和分析结果储存在`images`文件夹下。
- `KMeans`中`KMeans.py`运行得到手写Kmeans和调库实现的聚类结果。
- `utils`中储存了辅助文件，包括数据处理和可视化。