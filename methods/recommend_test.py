import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 数据加载与预处理
# 请将文件路径替换为实际文件路径
data = pd.read_excel(r'D:\a_work\1-phD\project\3-lin\data\literatures-data-102000\savedrecs1000.xls')  # 替换为你的 Excel 文件路径

# 查看数据结构
print(data.head())

# 检查缺失值
print(data.isnull().sum())

# 2. 用户输入的关键词
user_keywords = "nanozyme"  # 请替换为用户输入的关键词

# 3. 文本预处理：结合文章的关键词列表和摘要
corpus = data['Keyword List'].fillna('') + " " + data['Abstract'].fillna('')

# 4. 使用 TF-IDF 将文章内容转换为特征向量
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus)

# 5. 计算用户输入的关键词与文章的相似度
user_tfidf = vectorizer.transform([user_keywords])
cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

# 6. 获取与用户输入关键词最相关的前10篇文章
similar_articles_idx = cosine_similarities.argsort()[-10:][::-1]  # 获取相似度最高的10篇文章

# 7. 计算推荐得分
# 设置权重（可以根据需求进行调整）
weights = {
    'similarity': 0.6,  # 关键词匹配的权重
    '180_day_usage': 0.2,  # 180天使用量的权重
    'since_2013_usage': 0.2  # 2013年以来使用量的权重
}

# 8. 创建推荐得分列
data['recommend_score'] = (weights['similarity'] * cosine_similarities[similar_articles_idx]) + \
                           (weights['180_day_usage'] * data.loc[similar_articles_idx, '180 Day Usage Count']) + \
                           (weights['since_2013_usage'] * data.loc[similar_articles_idx, 'Since 2013 Usage Count'])

# 9. 排序并选择推荐结果
recommended_articles = data.loc[similar_articles_idx].sort_values(by='recommend_score', ascending=False)

# 10. 显示推荐的文章标题、关键词、摘要和使用量
recommended_articles_display = recommended_articles[['Article Title', 'Keyword List', 'Abstract', 
                                                    '180 Day Usage Count', 'Since 2013 Usage Count']]

print(recommended_articles_display)
