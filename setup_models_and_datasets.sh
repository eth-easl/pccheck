# #!/bin/bash

## NVIDIA DeepLearning repo
cd ~
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples
git reset --hard 54e2fb4853ac0c393335f5187bd3b9aff4bbd765

# Transformer
cd ~
python3.9 -m pip install 'git+https://github.com/NVIDIA/dllogger'
cp pccheck/checkpoint_eval/models/transformer/getdata.sh DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/
cp pccheck/checkpoint_eval/models/transformer/train_checkfreq.py DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch/
cp pccheck/checkpoint_eval/models/transformer/train_gpm.py DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch/
cp pccheck/checkpoint_eval/models/transformer/train_pccheck.py DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch/
cp pccheck/checkpoint_eval/models/transformer/run_cfreq.py DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch/
cp pccheck/checkpoint_eval/models/transformer/run_gpm.py DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch/
cp pccheck/checkpoint_eval/models/transformer/run_pccheck.py DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch/
cd DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL
bash getdata.sh

# copy BERT
cd ~
cp pccheck/checkpoint_eval/models/bert/bertPrep.py DeepLearningExamples/PyTorch/LanguageModeling/BERT/data/
cp pccheck/checkpoint_eval/models/bert/create_datasets_from_start.sh DeepLearningExamples/PyTorch/LanguageModeling/BERT/data/
cp pccheck/checkpoint_eval/models/bert/modeling.py DeepLearningExamples/PyTorch/LanguageModeling/BERT/
cd DeepLearningExamples/PyTorch/LanguageModeling/BERT/data
bash create_datasets_from_start.sh
cd ~
cp pccheck/checkpoint_eval/models/bert/run_squad_chfreq.py DeepLearningExamples/PyTorch/LanguageModeling/BERT/
cp pccheck/checkpoint_eval/models/bert/run_squad_gpm.py DeepLearningExamples/PyTorch/LanguageModeling/BERT/
cp pccheck/checkpoint_eval/models/bert/run_squad_pccheck.py DeepLearningExamples/PyTorch/LanguageModeling/BERT/
cp pccheck/checkpoint_eval/models/bert/run_cfreq.py DeepLearningExamples/PyTorch/LanguageModeling/BERT/
cp pccheck/checkpoint_eval/models/bert/run_gpm.py DeepLearningExamples/PyTorch/LanguageModeling/BERT/
cp pccheck/checkpoint_eval/models/bert/run_pccheck.py DeepLearningExamples/PyTorch/LanguageModeling/BERT/

# ############################################################################################################################################

# ## HF Transformers
cd ~
git clone https://github.com/huggingface/transformers && cd transformers && git reset --hard ee88ae59940fd4b2c8fc119373143d7a1175c651 && python3.9 -m pip install -e .
cd ~ && python3.9 -m pip install git+https://github.com/huggingface/datasets#egg=datasets
cp pccheck/checkpoint_eval/models/opt/run_clm_cfreq.py transformers/examples/pytorch/language-modeling/
cp pccheck/checkpoint_eval/models/opt/run_clm_gpm.py transformers/examples/pytorch/language-modeling/
cp pccheck/checkpoint_eval/models/opt/run_clm_pccheck.py transformers/examples/pytorch/language-modeling/
cp pccheck/checkpoint_eval/models/opt/trainer_checkfreq.py transformers/src/transformers/
cp pccheck/checkpoint_eval/models/opt/trainer_gpm.py transformers/src/transformers/
cp pccheck/checkpoint_eval/models/opt/trainer_pccheck.py transformers/src/transformers/
cp pccheck/checkpoint_eval/models/opt/__init__.py transformers/src/transformers/

# ############################################################################################################################################

# ## DeepSpeed
cd ~
DEEPSPEED_PATH=$(python3.9 -c 'import deepspeed; print(deepspeed.__path__[0])' | tail -1)
cp pccheck/checkpoint_eval/deepspeed/__init__.py $DEEPSPEED_PATH/
cp pccheck/checkpoint_eval/models/llm_distr/trainer_pp.py transformers/src/transformers/
cp pccheck/checkpoint_eval/models/llm_distr/deepspeed.py transformers/src/transformers/
cp pccheck/checkpoint_eval/models/llm_distr/ds_config.json transformers/examples/pytorch/language-modeling/
cp pccheck/checkpoint_eval/models/llm_distr/bloom_ds.py transformers/examples/pytorch/language-modeling/
cp pccheck/checkpoint_eval/models/llm_distr/convert_to_ds.py transformers/examples/pytorch/language-modeling/
cp pccheck/checkpoint_eval/models/llm_distr/opt_ds.py transformers/examples/pytorch/language-modeling/
cp pccheck/checkpoint_eval/models/llm_distr/run_clm_pp_cfreq.py transformers/examples/pytorch/language-modeling/
cp pccheck/checkpoint_eval/models/llm_distr/run_clm_pp_gemini.py transformers/examples/pytorch/language-modeling/
cp pccheck/checkpoint_eval/models/llm_distr/run_clm_pp_gpm.py transformers/examples/pytorch/language-modeling/
cp pccheck/checkpoint_eval/models/llm_distr/run_clm_pp_pccheck.py transformers/examples/pytorch/language-modeling/
