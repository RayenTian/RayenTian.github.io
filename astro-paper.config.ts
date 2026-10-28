import { defineAstroPaperConfig } from "./src/types/config";

export default defineAstroPaperConfig({
  site: {
    url: "https://rayentian.github.io/",
    title: "Stray Gradients",
    description: "Technical notes and thoughts.",
    author: "Rayen Tian",
    profile: "https://github.com/RayenTian",
    ogImage: "default-og.jpg",
    lang: "en",
    timezone: "Asia/Shanghai",
    dir: "ltr",
  },
  posts: {
    perPage: 10,
    perIndex: 5,
    scheduledPostMargin: 15 * 60 * 1000,
  },
  features: {
    lightAndDarkMode: true,
    dynamicOgImage: true,
    showArchives: true,
    showBackButton: true,
    editPost: {
      enabled: true,
      url: "https://github.com/RayenTian/RayenTian.github.io/edit/main/",
    },
    search: "pagefind",
  },
  socials: [{ name: "github", url: "https://github.com/RayenTian" }],
  shareLinks: [
    { name: "x", url: "https://x.com/intent/post?url=" },
    { name: "mail", url: "mailto:?subject=See%20this%20post&body=" },
  ],
});
