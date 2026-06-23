<!-- Chạy script để train -->

python train.py --train_csv data/train/train.csv --image_dir data/train/images --batch_size 16 --epochs 20 --device cuda --save_dir checkpoints

<!-- Chạy script để test từng ảnh -->

python demo/app.py --image data/test/images/TEST_0006.jpg --model checkpoints/model_epoch_25.pth

<!-- Chạy test chất lượng model -->

python evaluate.py --test_csv data/test/test.csv --image_dir data/test/images --model checkpoints/model_epoch_25.pth --device cuda

<!-- Chạy test chất lượng model theo kiểu n/x -->

python test_accuracy.py --test_csv data/test/test.csv --image_dir data/test/images --model checkpoints/model_epoch_25.pth --num_samples 50

python test_accuracy.py --test_csv data/datatest/Test1/test.csv --image_dir data/datatest/Test1/images --model checkpoints/model_epoch_25.pth --num_samples 50
# AI
# AI
