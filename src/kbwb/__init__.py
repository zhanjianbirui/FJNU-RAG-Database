"""kbwb —— 本地运行的知识库搭建工作台。

模块划分对应变更 add-knowledge-base-workbench 中的能力边界：

- ``config``     配置加载、存储布局、站点配置 schema
- ``providers``  Chat 与 Embedding 的 provider 契约及内置实现
- ``acquire``    网页爬取与本地导入两条语料入口
- ``process``    正文提取、附件解析、Markdown 中间表示、语义切分
- ``profile``    语料画像与规则式配置推荐
- ``kernel``     KnowledgeKernel 契约与混合检索实现
- ``qa``         问答组装、拒答、单库导出服务
- ``server``     本地工作台 HTTP 服务与后台长任务
- ``eval``       评测集、指标与检索可达性体检
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
