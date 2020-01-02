# 0
echo 'Section 0'
#sh scripts/test_saving_resnext101_sota.sh
TEST_DIR=../data/复赛/测试集B
OLD_DIR=`pwd`

# 1
echo 'Section 1'
#rm -r $TEST_DIR/fine_tune_0.1
#rm $TEST_DIR/fine_tune
#python3 prepare_fine_tune_data.py --thresh 0.1
cd $TEST_DIR
#ln -s -f fine_tune_0.1 fine_tune
cd $OLD_DIR
#sh scripts/ft_resnext101.sh
sh scripts/test_saving_resnext101.sh

# 2
echo 'Section 2'
rm -r $TEST_DIR/fine_tune_0.3
rm $TEST_DIR/fine_tune
python3 prepare_fine_tune_data.py --thresh 0.3
cd $TEST_DIR
ln -s -f fine_tune_0.3 fine_tune
cd $OLD_DIR
sh scripts/ft_resnext101.sh
sh scripts/test_saving_resnext101.sh

# 3
echo 'Section 3'
rm -r $TEST_DIR/fine_tune_0.5
rm $TEST_DIR/fine_tune
python3 prepare_fine_tune_data.py --thresh 0.5
cd $TEST_DIR
ln -s -f fine_tune_0.5 fine_tune
cd $OLD_DIR
sh scripts/ft_resnext101.sh
sh scripts/test_saving_resnext101.sh

# 3 repeat
echo 'Section 3 repeat'
rm -r $TEST_DIR/fine_tune_0.5
rm $TEST_DIR/fine_tune
python3 prepare_fine_tune_data.py --thresh 0.5
cd $TEST_DIR
ln -s -f fine_tune_0.5 fine_tune
cd $OLD_DIR
sh scripts/ft_resnext101.sh
sh scripts/test_saving_resnext101.sh

# 3 repeat 2
echo 'Section 3 repeat 2'
rm -r $TEST_DIR/fine_tune_0.5
rm $TEST_DIR/fine_tune
python3 prepare_fine_tune_data.py --thresh 0.5
cd $TEST_DIR
ln -s -f fine_tune_0.5 fine_tune
cd $OLD_DIR
sh scripts/ft_resnext101.sh
sh scripts/test_saving_resnext101.sh





