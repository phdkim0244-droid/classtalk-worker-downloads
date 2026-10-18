#!/usr/bin/env node
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { stat, writeFile } from "node:fs/promises";
import { resolve, join } from "node:path";

// Run after downloading the three successful native CI artifacts into one folder.
// This command only writes metadata; publishing is a separate maintainer action.
const [directory, version] = process.argv.slice(2);
if (!directory || !/^\d+\.\d+\.\d+$/.test(version ?? "")) {
  throw new Error("Usage: node scripts/make-worker-manifest.mjs <installer-directory> <version>");
}
const root = resolve(directory);
const names = {
  "windows-x64": `ClassTalk-Worker-${version}-windows-x64-Setup.exe`,
  "macos-arm64": `ClassTalk-Worker-${version}-macos-arm64.dmg`,
  "macos-x64": `ClassTalk-Worker-${version}-macos-x64.dmg`,
};
const downloads = [];
for (const [platform, filename] of Object.entries(names)) {
  const path = join(root, filename);
  const info = await stat(path);
  if (!info.isFile() || info.size === 0 || info.size > 2_147_483_648) throw new Error(`Invalid installer: ${filename}`);
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  // Signing must be verified separately before a maintainer changes this flag.
  downloads.push({ platform, filename, size: info.size, sha256: digest.digest("hex"), signed: false });
}
await writeFile(join(root, "manifest.json"), JSON.stringify({ schemaVersion: 1, version, downloads }, null, 2) + "\n");
await writeFile(join(root, "SHA256SUMS"), downloads.map(d=>`${d.sha256}  ${d.filename}`).join("\n") + "\n");
console.log(`Verified ${downloads.length} installers; wrote manifest.json and SHA256SUMS.`);
