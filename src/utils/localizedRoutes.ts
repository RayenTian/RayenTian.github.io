import type { GetStaticPathsOptions } from "astro";
import { getCollection } from "astro:content";
import { getSortedPosts } from "./getSortedPosts";
import { getUniqueTags } from "./getUniqueTags";
import { getPostSlug } from "./getPostPaths";
import { slugifyAll } from "./slugify";
import config from "@/config";

export async function articlePaths(locale: string) {
  const posts = getSortedPosts(await getCollection("posts"), locale);
  const adjacent = (index: number) => {
    const post = posts[index];
    return post
      ? { id: post.id, title: post.data.title, filePath: post.filePath }
      : null;
  };
  return posts.map((post, index) => ({
    params: { slug: getPostSlug(post.id, post.filePath) },
    props: {
      post,
      prevPost: adjacent(index + 1),
      nextPost: adjacent(index - 1),
    },
  }));
}

export async function postListPaths(
  locale: string,
  { paginate }: GetStaticPathsOptions
) {
  return paginate(getSortedPosts(await getCollection("posts"), locale), {
    pageSize: config.posts.perPage,
  });
}

export async function tagPaths(
  locale: string,
  { paginate }: GetStaticPathsOptions
) {
  const posts = getSortedPosts(await getCollection("posts"), locale);
  return getUniqueTags(posts).flatMap(({ tag, tagName }) =>
    paginate(
      posts.filter(({ data }) => slugifyAll(data.tags).includes(tag)),
      { params: { tag }, props: { tagName }, pageSize: config.posts.perPage }
    )
  );
}
