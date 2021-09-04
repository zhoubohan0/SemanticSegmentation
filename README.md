

# Semantic segmentation

## Introduction

​	基于`mediapipe`的人像语义分割，可输入人像数据集，输出完成语义分割后的图像数据集；亦可通过内置摄像头实时录制视频，完成背景填充、背景模糊虚化、虚拟背景替换等。

## 第三方库

| 第三方库名 | 版本     |
| :--------- | :------- |
| opencv     | 4.5.2.52 |
| mediapipe  | 0.8.7.2  |
| numpy      | 1.19.3   |

## 代码调用

Windows配置anaconda，创建虚拟环境，激活虚拟环境

```gfm
conda create -n env
conda activate env
```

添加清华大学镜像源

```gfm
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
```

安装第三方库(`opencv`可以在官网获取最新版本)

```gfm
conda install numpy 
pip install opencv
pip install mediapipe
```

cd至合适的目录下

```gfm
git clone <link>
cd SemanticSegmentation
```

对图像集进行人像语义分割，在outputimage查看输出结果

```python
python SemanticSegmentation.py --choice image --bgcolor 100 150 200 --maskcolor 255 255 255
```

视频实时语义分割

```python
python SemanticSegmentation.py --choice video --bg_mode 0
```

> 以上两行命令中--bgcolor，--maskcolor，--bg_mode为可选参数，可调整
>
> ![parameters.png](https://i.loli.net/2021/09/04/dgTV5bak8JFSenA.png)

退出虚拟环境，回到base

```gfm
conda deactivate env
```

删除虚拟环境

```gfm
conda remove -n env --all
```

> Anaconda常用命令：https://www.cnblogs.com/wind-chaser/p/11325733.html#

## 文件结构

| 文件夹或文件            | 说明                 |
| ----------------------- | -------------------- |
| background              | 自定义背景图片       |
| inputimage              | 待分割的图像数据集   |
| outputimage             | 输出的分割后的图像集 |
| outputvideo             | 输出分割后的视频     |
| src                     | 参数说明             |
| README.md               | 解释文档             |
| SemanticSegmentation.py | 主程序，命令行运行   |

