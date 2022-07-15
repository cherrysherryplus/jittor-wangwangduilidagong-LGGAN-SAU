# continue train at epoch 91
python train.py \
--input_path ./datasets/landscape/train \
--lambda_feat 10.0 \
--lambda_vgg 5.0 \
--lambda_l1 20.0 \
--lambda_class 5.0 \
--lambda_tv 4.0 \
--continue_train