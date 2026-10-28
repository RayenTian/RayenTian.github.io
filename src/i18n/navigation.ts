import { getCollection } from "astro:content";
import { getRelativeLocaleUrl } from "astro:i18n";
import { getSortedPosts } from "@/utils/getSortedPosts";
import { getPostSlug, getPostUrl } from "@/utils/getPostPaths";
import { getUniqueTags } from "@/utils/getUniqueTags";
import { stripBase, stripLocale } from "@/utils/withBase";

/** Resolve real translations; a missing article translation links to the post list. */
export async function languageNavigation(pathname: string, locale: string) {
  const targetLocale = locale === "zh" ? "en" : "zh";
  const path =
    stripLocale(stripBase(pathname), locale).replace(/\/+$/, "") || "/";
  const posts = await getCollection("posts");
  const currentPosts = getSortedPosts(posts, locale);
  const targetPosts = getSortedPosts(posts, targetLocale);
  const currentPost = currentPosts.find(
    post => path === `/posts${getPostSlug(post.id, post.filePath)}`
  );
  const key = (post: (typeof posts)[number]) =>
    post.data.translationKey ?? getPostSlug(post.id, post.filePath);
  let translatedPath: string | undefined;
  let switchHref = getRelativeLocaleUrl(targetLocale, "posts");
  if (currentPost) {
    const translated = targetPosts.find(post => key(post) === key(currentPost));
    if (translated)
      translatedPath = getPostUrl(
        translated.id,
        translated.filePath,
        targetLocale
      );
  } else if (
    ["/", "/about", "/posts", "/tags", "/archives", "/search"].includes(path)
  ) {
    translatedPath = getRelativeLocaleUrl(targetLocale, path.slice(1));
  } else if (path === "/404" || path === "/404.html") {
    switchHref = getRelativeLocaleUrl(
      targetLocale,
      targetLocale === "en" ? "404.html" : "404"
    );
  } else if (path.startsWith("/tags/")) {
    const tag = path.split("/")[2];
    switchHref = getRelativeLocaleUrl(targetLocale, "tags");
    if (getUniqueTags(targetPosts).some(item => item.tag === tag)) {
      switchHref = getRelativeLocaleUrl(targetLocale, `tags/${tag}`);
    }
  }
  return {
    targetLocale,
    switchHref: translatedPath ?? switchHref,
    // Only actual equivalents belong in hreflang, never the list fallback.
    alternates: translatedPath
      ? [
          { lang: locale, href: pathname },
          { lang: targetLocale, href: translatedPath },
        ]
      : [],
  };
}
