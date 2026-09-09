"""DOM 上的共用判断。

导航、侧栏、页脚这些结构性区域既干扰正文选择器的推断，也干扰详情页链接的
识别，因此判定逻辑集中在此处，由采样与推断共用。
"""

__all__ = [
    "NOISE_WORDS",
    "has_noise_ancestor",
    "is_noise_element",
    "selector_candidates",
    "text_of",
    "unique_selector",
]

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


def selector_candidates(element) -> list[str]:
    """为一个元素枚举可用于定位它的 CSS 选择器，由具体到宽泛。"""
    tag = element.tag if isinstance(element.tag, str) else ""
    if not tag:
        return []
    found: list[str] = []
    element_id = (element.get("id") or "").strip()
    if element_id and " " not in element_id:
        found.append(f"{tag}#{element_id}")
    classes = (element.get("class") or "").split()
    if len(classes) > 1:
        # 多个 class 连写比单个更具体，先试它
        found.append(tag + "".join(f".{name}" for name in classes))
    found.extend(f"{tag}.{name}" for name in classes)
    found.append(tag)
    return found


def unique_selector(document, element) -> str:
    """返回一个**首个匹配即该元素**的选择器。

    只保证"页面上存在某个元素满足条件"是不够的：抽取时取的是首个匹配，
    若选择器匹配多个元素，取到的很可能不是推断时看中的那个。因此这里逐个
    验证候选，选中第一个首匹配即目标的；都不满足时退回带位置的路径选择器。
    """
    for candidate in selector_candidates(element):
        try:
            matched = document.cssselect(candidate)
        except Exception:
            continue
        if matched and matched[0] is element:
            return candidate
    return _positional_selector(element)


def _positional_selector(element) -> str:
    """自根向下的位置路径，用于无 class/id 可依或选择器有歧义的元素。"""
    parts: list[str] = []
    node = element
    while node is not None and isinstance(node.tag, str):
        parent = node.getparent()
        if parent is None:
            parts.append(node.tag)
            break
        siblings = [c for c in parent if isinstance(c.tag, str) and c.tag == node.tag]
        index = siblings.index(node) + 1
        parts.append(f"{node.tag}:nth-of-type({index})" if len(siblings) > 1 else node.tag)
        node = parent
    return " > ".join(reversed(parts))
