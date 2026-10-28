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

## Chinese and English

English keeps the existing URLs; Chinese pages use `/zh/`. The navigation
language switch opens the matching translation, or the target language's post
list when an article has no published translation. Lists, tags, archives, RSS,
and search are separated by language.

For bilingual posts, keep the English file at `src/content/posts/my-post.md`
and the Chinese file at `src/content/posts/_zh/my-post.md`. The `_zh` directory
is omitted from the public article slug. Give both versions the same
`translationKey`, and set `lang: zh` on the Chinese version (`lang` defaults to
`en` for existing posts):

```yaml
lang: zh
translationKey: my-post
```

Each version has its own title, description, body, dates, and draft status.
The language switch uses published translations only. The two files should use
the same relative path and filename after omitting `_zh`; different slugs can
also be paired with an explicit `translationKey`. Translation is authored in
Markdown; the switch does not machine-translate content at runtime.

Chinese UI strings are in `src/i18n/lang/zh.ts`, and the Chinese About page is
`src/content/pages/about-zh.md`. Chinese posts use the site's default social
preview image unless an `ogImage` is supplied (the generated image font only
covers Latin text).

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
