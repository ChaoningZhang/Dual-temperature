# Dual Temperature Helps Contrastive Learning Without Many Negative Samples: Towards Understanding and Simplifying MoCo (Accepted by CVPR2022)

Chaoning Zhang, Kang Zhang, Trung X. Pham, Axi Niu, Zhinan Qiao, Chang D. Yoo, In So Kweon

This repository is the official implementation of ["Dual Temperature Helps Contrastive Learning Without Many Negative Samples: Towards Understanding and Simplifying MoCo"](https://arxiv.org/abs/2203.17248).

# Enviroment

Please refer [solo-learn](https://github.com/vturrisi/solo-learn) to install the enviroment.

# Training
To train SimCo, SimMoCo, and MoCoV2, use the script in folder `./bash_files`.


# Results
| Batch size | 64    | 128   | 256            | 512   | 1024  |
|------------|-------|-------|----------------|-------|-------|
| MoCo v2    | 52.58 | 54.40 | 53.28          | 51.47 | 48.90 |
| SimMoCo    | 54.02 | 54.93 | 54.11          | 52.45 | 49.70 |
| SimCo      | 58.04 | 58.29 | **58.35** | 57.08 | 55.34 |

This code is developed based on [solo-learn](https://github.com/vturrisi/solo-learn).

# Citation
```
@article{zhang2022dual,
  title={Dual temperature helps contrastive learning without many negative samples: Towards understanding and simplifying moco},
  author={Zhang, Chaoning and Zhang, Kang and Pham, Trung X and Niu, Axi and Qiao, Zhinan and Yoo, Chang D and Kweon, In So},
  journal={CVPR},
  year={2022}
}
```


# Acknowledgement

This work was partly supported by Institute for Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) under grant No.2019-0-01396 (Development of framework for analyzing, detecting, mitigating of bias in AI model and training data), No.2021-0-01381 (Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments) and No.2021-0-02068 (Artificial Intelligence Innovation Hub).
