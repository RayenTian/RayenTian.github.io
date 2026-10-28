---
title: 你好，新博客
pubDatetime: 2026-10-27T12:00:00+08:00
description: 记录博客的写作与发布流程，包含代码、数学公式和独立 HTML 页面示例。
lang: zh
translationKey: hello-world
tags:
  - blog
---

## 写一篇文章

在 `src/content/posts/` 下创建 Markdown 文件，填好 frontmatter，再推送到仓库。GitHub Actions 会自动构建并发布网站。

```md
---
title: 文章标题
pubDatetime: 2026-10-27T12:00:00+08:00
description: 在文章列表和搜索结果中显示的一句话摘要。
lang: zh
translationKey: my-post
tags:
  - blog
---

正文……
```

添加 `draft: true` 可以将文章保留为草稿。中英文版本使用相同的 `translationKey`，即可通过导航栏切换。

## 代码高亮

```python file=hello.py
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

## 数学公式

支持行内公式 $E = mc^2$，也支持独立公式：

$$
\mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

## 发布独立 HTML

`public/` 目录中的文件会原样发布。例如，[这个演示页面](/demos/hello.html) 对应 `public/demos/hello.html`，适合存放独立演示和工具生成的报告。
