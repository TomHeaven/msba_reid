# 4
echo 'Section 4'
rm -r /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune_0.6
rm /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune
python3 prepare_fine_tune_data.py --thresh 0.6
ln -s -f  /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune_0.6 /home/tomheaven/比赛/行人重识别2019/data/复赛/测试集A/fine_tune
sh scripts/ft_resnext101.sh
sh scripts/test_saving_resnext101.sh