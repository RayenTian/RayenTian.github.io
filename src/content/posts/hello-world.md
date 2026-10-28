---
title: Hello, New Blog
pubDatetime: 2026-10-27T12:00:00+08:00
description: The writing workflow of this blog, with math and code examples, and how to publish standalone HTML pages.
tags:
  - blog
---

## Table of contents

## Writing a post

Create a `.md` file under `src/content/posts/`, fill in the frontmatter, and `git push`. GitHub Actions builds and publishes the site automatically.

```md
---
title: Post title
pubDatetime: 2026-10-27T12:00:00+08:00
description: One-line summary shown in lists and search results.
tags:
  - tag
---

Body...
```

Add `draft: true` to keep a post unpublished.

## Code highlighting

```python file=hello.py
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

## Math

Inline math $E = mc^2$, and block math:

$$
\mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

## Publishing raw HTML

Files under `public/` are published as-is. For example, [this demo page](/demos/hello.html) is `public/demos/hello.html`. It is handy for tool-generated reports or standalone demos.
