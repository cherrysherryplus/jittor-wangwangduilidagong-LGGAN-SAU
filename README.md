### 1.数据集准备
```bash
# 数据集目录结构
datasets
--landscape     (数据集目录)
----train       (训练数据)
------imgs
------labels
----val
------imgs
------labels
----testA       (A榜测试数据)
------imgs
------labels
----testB       (B榜测试数据)
------imgs
------labels
```
1) 在`datasets`目录下，建立指向数据集目录的软链接，运行`ln -s <数据集目录> <软链接名>`，如`ln -s /datasets/landscape landscape`
2) 运行`split_dataset.ipynb`所有单元格，分割训练集和验证集


### 2.训练
2.1 风景图像生成模型。其训练分为了两个阶段
- 阶段 1，按照`train.sh`中的命令运行，训练到第 **90** 个epoch结束
- 阶段 2，按照`train_second_stage.sh`中的命令运行，从第 **91** 个epoch开始，到第 **120** 个epoch结束。

2.2 【可选，分数只差 *0.1* 分】 超分辨率缩放模型。其训练需要在原数据集上构造512\*384和256\*192两个版本的数据集
- 构造数据集。运行`construct_sr_dataset.ipynb`所有单元格构造训练集和验证集。
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