import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import { getSortedPosts } from "@/utils/getSortedPosts";
import { getPostUrl } from "@/utils/getPostPaths";
import config from "@/config";

export async function localizedFeed(locale: string) {
  const posts = await getCollection("posts");
  const sortedPosts = getSortedPosts(posts, locale);

  return rss({
    title: config.site.title,
    description: locale === "zh" ? "技术积累与思考。" : config.site.description,
    site: config.site.url,
    items: sortedPosts.map(({ data, id, filePath }) => ({
      link: getPostUrl(id, filePath, locale),
      title: data.title,
      description: data.description,
      pubDate: new Date(data.modDatetime ?? data.pubDatetime),
    })),
  });
}

export async function GET() {
  return localizedFeed("en");
}
