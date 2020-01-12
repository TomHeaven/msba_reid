# ReID2019

Code From 摸奖的Nudters。本项目的sh脚本仅适合unix或者linux系统运行。

# 使用

+ 编译cython代码

```bash
cd csrc/eval_cylib; make
```

+ 修改config/defaults的_C.OUTPUT_DIR为本机合适目录，此目录用于保存模型权重和训练日志。
+ 修改config/defaults的_C.DATASETS.ROOT_DIR为本机数据路径。
+ 运行data_preprocess.py，将数据处理为程序可以接受的格式。注意修改数据路径为本机路径。
+ 修改scripts下test_saving开头的脚本参数TEST_WEIGHT的路径为本机路径。
+ 运行train_test_all.sh，将在logs目录下生成模型，在dist_mats目录下生成用于集成的距离矩阵，在submit目录下生成单模型提交结果。
+ 修改scripts/test_saving_multiple_model.sh，填入需要集成的距离矩阵文件路径（.h5），运行之。将在submit目录下生成集成结果。


