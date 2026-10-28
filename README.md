# Stray Gradients

Personal technical blog built with [Astro](https://astro.build) and the [AstroPaper](https://github.com/satnaing/astro-paper) theme, deployed to GitHub Pages via GitHub Actions.

## Writing

Create a `.md` file under `src/content/posts/`:

```md
---
title: Post title
pubDatetime: 2026-10-27T12:00:00+08:00
description: One-line summary
tags:
  - tag
---
```

Posts with `draft: true` are not published; posts whose `pubDatetime` is in the future appear only after that time. Code highlighting and KaTeX math (`$...$`, `$$...$$`) are supported.

## Publishing standalone HTML

Files under `public/` are published as-is, e.g. `public/demos/foo.html` is served at `/demos/foo.html`.

## Development

```bash
pnpm install
pnpm dev        # http://localhost:4321
pnpm build      # build and generate the search index
```

## Deployment

Pushing to `main` triggers `.github/workflows/deploy.yml`. On first use, set Source to **GitHub Actions** in Settings → Pages.

Site settings live in `astro-paper.config.ts`.
