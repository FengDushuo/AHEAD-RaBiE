import tornado.web
import tornado.escape
import os
import handlers.config
import handlers.inputdata
# import methods.readdb as mrd
from handlers.base import BaseHandler
import json
import random
import handlers.config
import pandas as pd
import joblib
import numpy as np
import matplotlib as plot
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
import seaborn as sns
from methods.model_train_windows import run_models,do_predict,loadAbstractFile,getWord2Vec,getSentenceVec
import nltk
from nltk.corpus import stopwords
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import os

def ensure_sqlite_from_excel(excel_file, table_name='articles', outpath='static/data/db'):
    # 构建 SQLite 路径
    base = os.path.splitext(os.path.basename(excel_file))[0]
    sqlite_path = os.path.join(outpath, base + ".db")

    # 如果数据库不存在，则创建
    if not os.path.exists(sqlite_path):
        print(f"📂 未发现数据库，正在从 Excel 创建: {sqlite_path}")
        import_excel_to_sqlite(excel_file, sqlite_path, table_name)
    else:
        print(f"✅ 数据库已存在: {sqlite_path}")

    return sqlite_path

def import_excel_to_sqlite(excel_file, sqlite_path='static/data/db/articles.db', table_name='articles'):
    base, ext = os.path.splitext(excel_file)
    df = pd.read_excel(excel_file, engine='openpyxl' if ext.endswith('x') else None)
    
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    
    # ✅ 加入索引提升查询速度
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_year ON {table_name}([Publication Year])")
    conn.commit()
    conn.close()
    print(f"✅ 已保存到数据库文件并建立索引：{sqlite_path}")

class KnowledgeHandler(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        username = tornado.escape.json_decode(self.current_user)
        if handlers.config.global_data_path == '':
            directory_path = 'static/data/history/'
            first_file = next((f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))), None)
            history_path= 'static/data/history/'+first_file
            if os.path.exists(history_path):
                with open(history_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                handlers.config.global_data_json = json.dumps(json_data)
                timelines=json_data['timelines']
                researches=json_data['researches']
            else:
                self.write({
                    "status": "error",
                    "message": f"File  not found"
                })
        else:
            timelines = json.loads(handlers.config.global_data_json)['timelines']
            researches = json.loads(handlers.config.global_data_json)['researches']

                
        user_infos=[[99,username,"123456","123456@11.com"]]
        self.render("knowledge.html", user = user_infos[0], timelines=timelines,researches=researches)

    def post(self):
        username = tornado.escape.json_decode(self.current_user)
        user_keywords = user_input = self.get_argument('user_input', '')  
        timeline = self.get_argument("timeline")
        research = self.get_argument("research")
        if handlers.config.global_data_path == '':
            directory_path = 'static/data/history/'
            first_file = next((f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))), None)
            history_path= 'static/data/history/'+first_file
            if os.path.exists(history_path):
                with open(history_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                handlers.config.global_data_path = json_data["input_path"]
            else:
                self.write({
                    "status": "error",
                    "message": f"File not found"
                })
        filename = handlers.config.global_data_path
        if "upload/" not in filename:
            filename = "upload/"+filename        
        # 读取数据
        use_cols = ["Article Title", "DOI", "Keyword List", "Abstract", "180 Day Usage Count", "Since 2013 Usage Count", "Publication Year"]
        base, ext = os.path.splitext(filename)
        #data = pd.read_excel(filename, engine='openpyxl' if ext.endswith('x') else None, usecols=use_cols)
        sqlite_path = ensure_sqlite_from_excel(filename,outpath="static/data/db")
        conn = sqlite3.connect(sqlite_path)
        query = f"""
        SELECT [Article Title], [DOI], [Keyword List], [Abstract], 
            [180 Day Usage Count], [Since 2013 Usage Count], [Publication Year]
        FROM articles
        WHERE [Publication Year] = ?
        """
        data = pd.read_sql(query, conn, params=(int(timeline),))
        conn.close()
        # 用户输入的关键词
        if data.empty:
            self.write({"status": "empty", "message": f"未找到 {timeline} 年的文献数据"})
            return
        # 提取关键词和摘要进行匹配
        corpus = data['Keyword List'].fillna('') + " " + data['Abstract'].fillna('')
        # 使用 TF-IDF 计算文本相似度
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(corpus)
        user_tfidf = vectorizer.transform([user_keywords])
        cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
        # 计算推荐得分
        weights = {
            'similarity': 0.6,  # 关键词匹配的权重
            '180_day_usage': 0.2,  # 180天使用量的权重
            'since_2013_usage': 0.2  # 2013年以来使用量的权重
        }
        data['recommend_score'] = (weights['similarity'] * cosine_similarities) + \
                                (weights['180_day_usage'] * data['180 Day Usage Count']) + \
                                (weights['since_2013_usage'] * data['Since 2013 Usage Count'])

        # 按推荐得分排序，获取前10个推荐结果
        recommended_articles = data.sort_values(by='recommend_score', ascending=False).head(10)
        # 构造返回的推荐文章列表
        ret_list = []
        for _, item in recommended_articles.iterrows():
            ret_item = [str(item["Article Title"]), "http://dx.doi.org/" + str(item["DOI"])]
            ret_list.append(ret_item)
        # 生成Excel表格并保存
        recommended_articles = recommended_articles[['Article Title', 'DOI']]
        recommended_articles.to_excel("recommended_articles.xlsx", index=False)
        # 获取用户信息
        user_infos = [[99, username, "123456", "123456@11.com"]]        
        # 渲染页面并传递推荐结果
        self.render("knowledge_result.html", user=user_infos[0], recommends=ret_list)


