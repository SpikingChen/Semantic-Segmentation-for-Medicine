# train data
python ./data_prepare.py \
--image_path '../Train/Image/' \
--label_path '../Train/Layer_Masks/' \
--save_dir './preprocess_train_data'


# val data
python ./data_prepare.py \
--image_path '../Validation/Image' \
--label_path None \
--save_dir './preprocess_val_data'