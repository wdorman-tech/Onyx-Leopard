import { readdir, readFile } from "fs/promises";
import { join } from "path";

interface CompanyEntry {
  filename: string;
  name: string;
  industry: string;
  stage: string;
  node_count: number;
}

export async function GET() {
  const dir = join(process.cwd(), "public", "companies");

  let files: string[];
  try {
    files = await readdir(dir);
  } catch {
    return Response.json([]);
  }

  const oleoFiles = files.filter((f) => f.endsWith(".oleo"));
  const entries: CompanyEntry[] = [];

  for (const file of oleoFiles) {
    try {
      const raw = await readFile(join(dir, file), "utf-8");
      const data = JSON.parse(raw);
      const identity = data?.profile?.identity ?? {};
      entries.push({
        filename: file,
        name: identity.name || file.replace(".oleo", ""),
        industry: identity.industry || "",
        stage: identity.company_stage || "",
        node_count: data?.graph?.nodes?.length ?? 0,
      });
    } catch {
      // skip malformed files
    }
  }

  entries.sort((a, b) => a.name.localeCompare(b.name));
  return Response.json(entries);
}
