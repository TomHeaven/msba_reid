# 0
echo 'Section 0'
sh scripts/test_saving_resnext101_sota.sh

# 1
echo 'Section 1'
rm -r /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune_0.1
rm /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune
python3 prepare_fine_tune_data.py --thresh 0.1
ln -s -f  /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune_0.1 /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune
sh scripts/ft_resnext101.sh
sh scripts/test_saving_resnext101.sh

# 2
echo 'Section 2'
rm -r /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune_0.3
rm /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune
python3 prepare_fine_tune_data.py --thresh 0.3
ln -s -f  /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune_0.3 /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune
sh scripts/ft_resnext101.sh
sh scripts/test_saving_resnext101.sh

# 3
echo 'Section 3'
rm -r /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune_0.5
rm /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune
python3 prepare_fine_tune_data.py --thresh 0.5
ln -s -f  /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune_0.5 /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune
sh scripts/ft_resnext101.sh
sh scripts/test_saving_resnext101.sh

# 3 repeat
echo 'Section 3 repeat'
rm -r /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune_0.5
rm /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune
python3 prepare_fine_tune_data.py --thresh 0.5
ln -s -f  /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune_0.5 /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune
sh scripts/ft_resnext101.sh
sh scripts/test_saving_resnext101.sh

# 3 repeat2
echo 'Section 3 repeat2'
rm -r /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune_0.5
rm /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune
python3 prepare_fine_tune_data.py --thresh 0.5
ln -s -f  /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune_0.5 /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune
sh scripts/ft_resnext101.sh
sh scripts/test_saving_resnext101.sh




