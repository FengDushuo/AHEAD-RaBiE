import tornado.web
import tornado.escape
# import methods.readdb as mrd
from handlers.base import BaseHandler
import os
import swift
from swift.llm import AppArguments, app_main, DeployArguments, run_deploy
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class VosviewerHandler(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        # with run_deploy(DeployArguments(model='static/model/checkpoint-39500-merged', verbose=False, log_interval=-1, infer_backend='pt'),return_url=True) as url:
        #     app_main(AppArguments(model='static/model/checkpoint-39500-merged', base_url=url, stream=True, max_new_tokens=2048))
        self.redirect("http://127.0.0.1:7860")
