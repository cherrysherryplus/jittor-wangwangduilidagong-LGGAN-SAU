### 0.环境要求
0.1 硬件要求
- 显存要求
**batch size** 为 **2** 时，要求不小于 **24**G，如 **A5000** / **RTX3090**
- 存储要求
不小于 **20**G

0.2 软件要求
```bash
# 按照 requirements.txt 安装对应版本的包
pip install -r requirements.txt
```

### 1.数据集准备
```bash
# 目标数据集目录结构
LGGAN-jittor-fid-v2 (项目目录)
--datasets
----landscape       (数据集目录)
------train         (训练数据)
--------imgs
--------labels
------val
--------imgs
--------labels
------testA         (A榜测试数据)
--------labels
------testB         (B榜测试数据)
--------labels
```
1.1 为构建上述数据集目录结构。首先进入项目目录，执行下述命令，下载训练、测试集并解压
```bash
# 假定数据集目录名为 landscape
cd datasets && mkdir landscape && cd landscape

# train
wget -O train.zip https://cloud.tsinghua.edu.cn/f/1d734cbb68b545d6bdf2/?dl=1
unzip -q train.zip

# testA
wget -O testA.zip https://cloud.tsinghua.edu.cn/f/70195945f21d4d6ebd94/?dl=1
unzip -q testA.zip -d testA
cd testA && mv val_A_labels_cleaned labels
cd ..

# testB
wget -O testB.zip https://cloud.tsinghua.edu.cn/f/980d8204f38e4dfebbc8/?dl=1
unzip -q testB.zip -d testB
cd testB && mv val_B-labels-clean labels
cd ..

# clean
rm *.zip
```

1.2 运行`split_dataset.ipynb`所有单元格，分割训练集和验证集


### 2.训练
2.1 风景图像生成模型。其训练分为了两个阶段
- 阶段 1，按照`train.sh`中的命令运行，训练到第 **90** 个epoch结束
- 阶段 2，按照`train_second_stage.sh`中的命令运行，从第 **91** 个epoch开始，到第 **120** 个epoch结束。

2.2 【可选，分数只差 *0.1* 分】 超分辨率缩放模型。其训练需要在原数据集上构造512\*384和256\*192两个版本的数据集
- 构造数据集。运行`construct_sr_dataset.ipynb`所有单元格构造训练RCAN的训练集和验证集。
- 训练 **100** epoch，选择`best.pkl`作为后续缩放使用的模型。


### 3.测试
```bash
# checkpoint存放目录的结构
项目目录
--checkpoint            (风景图像生成checkpoint保存的目录)
----jittor              (训练数据)
------latest_net_G.pkl  (生成器checkpoint)
------latest_net_D.pkl  (判别器checkpoint)
..
..
..
--RCAN                  (【可选】超分辨率checkpoint保存的目录)
----best.pkl            (超分辨率checkpoint)
..
..
```

3.1 **加载训练好的checkpoint**
###### 3.1.1 风景图像生成
1）下载`netG、netD`的模型文件（包括latest_net_G.pkl及latest_net_D.pkl）
```bash
latest_net_G.pkl
链接：https://pan.baidu.com/s/1YEOBWii6iZUxcXOwgRwIHg?pwd=znss 
提取码：znss


latest_net_D.pkl
链接：https://pan.baidu.com/s/1Q1eDx2OQ6qw0LgQIWY4mbA?pwd=jxwz 
提取码：jxwz
```
2）将`latest_net_G.pkl、latest_net_D.pkl`移动到`checkpoints/jittor`目录下。

###### 3.1.2 【可选】超分辨率缩放
1）下载`RCAN-jittor`的模型文件
```bash
best.pkl
链接：https://pan.baidu.com/s/1GmBD-bN-SkWh7jrJn4LxXg?pwd=kui7 
提取码：kui7
```
2）将模型文件移动到 `RCAN/` 目录下


3.2 **正式测试**
```bash
# 【若超分辨率缩放不符合要求，请选择0.4712版本】

# 得分：0.4712
# 使用PIL.Image.resize缩放到384 * 512
python test.py --input_path ./datasets/landscape/testB/labels --output_path ./results


# 得分：0.48
# 使用jittor-RCAN超分辨率模型缩放到384 * 512
python test.py --input_path ./datasets/landscape/testB/labels --output_path ./results --use_sr
```