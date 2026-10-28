import { localizedFeed } from "../rss.xml";
export async function GET() {
  return localizedFeed("zh");
}
