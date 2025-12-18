
import os
import swift
from swift.llm import AppArguments, app_main, DeployArguments, run_deploy
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

with run_deploy(DeployArguments(model='static/model/checkpoint-39500-merged', verbose=False, log_interval=-1),return_url=True) as url:
    app_main(AppArguments(model='static/model/checkpoint-39500-merged', base_url=url, stream=True, max_new_tokens=2048))