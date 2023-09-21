# embedding-projector-zh
[Embedding Projector](http://projector.tensorflow.org/)是Google开源的高维数据可视化工具，本项目基于这款交互式Web应用程序搭建一个可以进行高维数据分析的系统。部分参考了[TSNE-UMAP-Embedding-Visualisation](https://harveyslash.github.io/TSNE-UMAP-Embedding-Visualisation/)这个项目。

随着AI大模型时代的到来，传统的数据类型由结构化、非结构化和半结构化等形态逐渐向**Embedding向量化**转变，数据的存储形态发生了质的变化，随之而来的是对Embedding向量数据库需求的爆发式增长。

## 向量数据库

向量数据库是一种专门用于存储、 管理、查询、检索向量(Vector Embeddings)的数据库，主要应用于人工智能、机器学习、数据挖掘等领域。同传统数据库相比，向量数据库不仅能够完成基本的CRUD(添加、读取查询、更新、删除)、元数据过滤、水平缩放等操作，还能够对向量数据进行更快速的相似性搜索。目前AI主流的大模型如Transformer、GPT等均能够将文本、图像等非结构化数据转化为高维向量，而伴随大模型应用场景的扩展，这些高维向量数据的存储、检索将显著带动向量数据库的市场需求。

## 向量最近邻搜索

传统数据库有事务处理(OLTP)与数据分析(OLAP)两大核心应用场景。典型的事务处理场景包括：知识库，问答，推荐系统，人脸识别，图片搜索，等等等等。

- 知识问答：给出一个自然语言描述的问题，返回与这些输入最为接近的结果；
- 以图搜图：给定一张图片，找出与这张图片在逻辑上最接近的其他相关图片；

这些业务需求说到底都是一个共同的数学问题：**最近邻检索(KNN)**。通过机器学习模型将原始形式(文本、语音和图像)的数据映射到某个维度(dimension)下的数学向量表征，然后存储到向量数据库，最后进行**向量最近邻搜索**，就可以十分快速的完成上面的业务需求。

- 给定一个向量，找到距离此向量最近的其他向量。

![236f36cca40ce9b64ea114f99f6f9a01_modb_20230512_52a14140-f069-11ed-83f7-38f9d3cd240d](https://github.com/shangfr/embedding-projector-zh/assets/12015563/8d82a6b6-6552-4b53-ba01-0cd106e6ea15)

## Embedding Projector
[Embedding Projector](http://projector.tensorflow.org/)是Google开源的高维数据可视化工具，它提供了四种(UMAP、T-SNE、PCA、CUSTOM)常用的数据降维(data dimensionality reduction)方法：PCA、UMAP、t-SNE都是非监督的降维算法，可以用于发现高维数据中的结构。CUSTOM自定义线性投影可以帮助发现数据集中有意义的方向(direction)，比如一个语言生成模型中一种正式的语调和随意的语调之间的区别。


