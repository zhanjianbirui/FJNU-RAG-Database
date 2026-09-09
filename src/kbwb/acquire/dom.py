"""DOM 上的共用判断。

导航、侧栏、页脚这些结构性区域既干扰正文选择器的推断，也干扰详情页链接的
识别，因此判定逻辑集中在此处，由采样与推断共用。
"""

__all__ = ["NOISE_WORDS", "has_noise_ancestor", "is_noise_element", "text_of"]

#: class / id 中出现这些片段的元素视为结构性噪声。
NOISE_WORDS = (
    "nav", "menu", "header", "footer", "sidebar", "side",
    "breadcrumb", "copyright", "banner", "topbar", "crumb",
)


def text_of(element) -> str:
    return " ".join(element.text_content().split())


def is_noise_element(element) -> bool:
    marker = f"{element.get('class', '')} {element.get('id', '')}".lower()
    return any(word in marker for word in NOISE_WORDS)


def has_noise_ancestor(element) -> bool:
    """元素自身或任一祖先是结构性噪声。"""
    node = element
    while node is not None:
        if isinstance(node.tag, str) and is_noise_element(node):
            return True
        node = node.getparent()
    return False
