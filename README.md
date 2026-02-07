# OODLib

## load data
For DNN-oriented methods, ID training data, ID test data and OOD test data must be loaded.

For VLM-oriented methods, ID test data and OOD test data must be loaded, ID training data is optional (few-shot learning).

ID datasets: `cifar10`, `cifar100`, `ImageNet-1k`

OOD datasets: `cifar10`, `cifar100`, `ImageNet-1k`

## Baselines (sorted by date)
|Methods|File name|Paper|
|---|---|---|
|MSP|[msp.py](./baselines/dnn_based/msp.py)|[A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks](https://arxiv.org/pdf/1610.02136) [ICLR 2017]|
|ODIN|[odin.py](./baselines/dnn_based/odin.py)|[Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks](https://openreview.net/pdf?id=H1VGkIxRZ) [ICLR 2018]|
|Energy|[energy.py](./baselines/dnn_based/energy.py)|[Energy-based Out-of-distribution Detection](https://proceedings.neurips.cc/paper_files/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf) [NeurIPS 2020]|
|ReAct|[react.py](./baselines/dnn_based/react.py)|[ReAct: Out-of-distribution Detection With Rectified Activations](https://proceedings.neurips.cc/paper_files/paper/2021/file/01894d6f048493d2cacde3c579c315a3-Paper.pdf) [NeurIPS 2021]|
|MaxLogit|[maxlogit.py](./baselines/dnn_based/maxlogit.py)|[Scaling Out-of-Distribution Detection for Real-World Settings](https://proceedings.mlr.press/v162/hendrycks22a/hendrycks22a.pdf) [ICML 2022]|
|DICE|[dice.py](./baselines/dnn_based/dice.py)|[DICE: Leveraging Sparsification for Out-of-Distribution Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840680.pdf) [ECCV 2022]|
|GEN|[gen.py](./baselines/dnn_based/gen.py)|[GEN: Pushing the Limits of Softmax-Based Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.pdf)[CVPR 2023]|
|ASH|[ash.py](./baselines/dnn_based/ash.py)|[Extremely Simple Activation Shaping for Out-of-Distribution Detection](https://openreview.net/pdf?id=ndYXTEL6cZz)[ICLR 2023]|
|CADRef|[ash.py](./baselines/dnn_based/ash.py)|[CADRef: Robust Out-of-Distribution Detection via Class-Aware Decoupled Relative Feature Leveraging](https://openaccess.thecvf.com/content/CVPR2025/papers/Ling_CADRef_Robust_Out-of-Distribution_Detection_via_Class-Aware_Decoupled_Relative_Feature_Leveraging_CVPR_2025_paper.pdf)[CVPR 2025]|